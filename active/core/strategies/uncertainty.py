import torch
import torch.nn as nn
import ignite.distributed as idist

from transformers import PreTrainedModel
from .strategy import ScoreBasedStrategy
from ..utils.data import move_to_device

from abc import abstractmethod
from typing import Sequence, Any, Optional

class UncertaintyStrategy(ScoreBasedStrategy):
    """ Abstract Base Class for uncertainty-based sampling strategies.
           
        Args:
            model (PreTrainedModel): transformer-based model used for prediction
            ignore_labels (Optional[Sequence[int]]): list of labels to ignore for computation of uncertainty scores
            random_sample (Optional[bool]): 
                whether to random sample query according to distribution given by uncertainty scores.
                Defaults to False, meaning the elements with maximum uncertainty scores are selected.
    """

    def __init__(
        self,
        model:PreTrainedModel,
        ignore_labels:Optional[Sequence[int]] =[],
        random_sample:Optional[bool] =False
    ) -> None:
        # initialize strategy
        super(UncertaintyStrategy, self).__init__(
            random_sample=random_sample
        )
        # move model to available device(s)
        self.model = idist.auto_model(model)
        # save labels to ignore as tensor
        self.ignore_labels = torch.LongTensor(ignore_labels)

    @abstractmethod
    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError()

    def reduce_scores(self, scores:torch.FloatTensor, mask:torch.BoolTensor) -> torch.FloatTensor:
        """ Reduce uncertainty scores for token classification tasks.
            By default uncertainty strategies in the context of token
            classification tasks compute one uncertainty score per token.
            The most naive way of reducing these into a single score is
            by computing their average, which is implemented in this function.

            Args:
                scores (torch.FloatTensor): 
                    uncertainty scores of shape (B, S, ...) where
                    B is the batch size and S is the sequence length.
                    Note that for nested nested token classification
                    (i.e. multiple classifications per token) the shape
                    is three-dimensional where the last dimension represents
                    the number of classifiers
                mask (torch.BoolTensor):
                    mask of valid scores in the `scores` tensor. The shape
                    matches the shape of the `score` tensor.

            Returns:
                reduced_scores (torch.FloatTensor): reduced scores of shape (B,)
        """
        # compute valid sequence lengths and avoid division by zero
        lengths = mask.sum(dim=1, keepdims=True)
        lengths = torch.maximum(lengths, torch.ones_like(lengths))
        # average over sequence and entities in case of nested bio tagging
        return (scores / lengths).flatten(start_dim=1).sum(dim=1)

    @torch.no_grad()
    def process(self, batch:Any) -> torch.FloatTensor:
        # move batch to device
        batch = move_to_device(batch, device=idist.device())
        # apply model and get output logits
        self.model.eval()
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        # compute uncertainty scores
        scores = self.uncertainty_measure(probs)
        assert scores.size() == logits.size()[:scores.ndim]        

        # ignore labels
        _, labels = logits.max(dim=-1)
        ignore_mask = torch.isin(labels, self.ignore_labels.to(labels.device))
        scores[ignore_mask] = 0.0

        if scores.ndim == 1:
            # no reduction needed
            return scores

        # prepare mask for broadcasting with scores and ignore mask
        # this basically ensures that the mask applies to the first
        # dimensions, contrary to usual broadcasting which aligns
        # axes in reverse order
        mask = batch['attention_mask'].bool()
        mask = mask.reshape(mask.shape + (1,) * (scores.ndim - mask.ndim))
        assert mask.ndim == scores.ndim == ignore_mask.ndim

        # apply mask to scores
        scores.masked_fill_(~mask, 0.0)
        # reduce scores
        return self.reduce_scores(scores, mask & ~ignore_mask)

class LeastConfidence(UncertaintyStrategy):

    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        return 1.0 - probs.max(dim=-1).values

class PredictionEntropy(UncertaintyStrategy):

    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        return torch.special.entr(probs).sum(dim=-1)

class EntropyOverMax(UncertaintyStrategy):
    """ Uncertainty Sampling Strategy for token classification tasks. Computes the entropy over the maximum 
        predicted probability of each token. See Equation (4.1) in Joey Oehman (2021).

        Args:
            model (PreTrainedModel): transformer-based model used for prediction
            ignore_labels (Optional[Sequence[int]]): list of labels to ignore for computation of uncertainty scores

        References:
            - Joey Oehman, Active Learning for Named Entity Recognition with Swedish Language Models, 2021
    """    

    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        # only makes sense for (nested) token-classification tasks
        assert probs.ndim >= 3, "Entropy over Maxima Sampling only makes sense for token classification tasks!"
        # only take the maximum over label space here to match expected shapes
        # in process and reduce score functions
        return probs.max(dim=-1).values
    
    def reduce_scores(self, scores:torch.FloatTensor, mask:torch.BoolTensor) -> torch.FloatTensor:
        # take maximum over classifiers/entities in case of nested
        # token classification tasks, for other tasks this does nothing
        scores = scores.reshape(scores.size(0), scores.size(1), -1).max(dim=-1).values
        # note that scores at ~mask are already set to zero
        # and entropy is defined to be zero at x=0 
        # (see https://pytorch.org/docs/stable/special.html)
        return torch.special.entr(scores).sum(dim=-1)
