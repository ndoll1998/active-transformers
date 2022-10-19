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
        # compute valid sequence lengths and avoid division by zero
        lengths = mask.sum(dim=-1)
        lengths = torch.maximum(lengths, torch.ones_like(lengths))
        # average over sequence
        return scores.flatten(start_dim=1).sum(dim=1) / lengths

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
        
        # apply mask to scores
        mask = batch['attention_mask'].bool()
        scores[~mask, ...] = 0.0
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
        # only makes sense for token-classification tasks
        assert probs.ndim == 3, "Entropy over Maxima Sampling only makes sense for token classification tasks!"
        return probs.max(dim=-1).values
    
    def reduce_scores(self, scores:torch.FloatTensor, mask:torch.BoolTensor) -> torch.FloatTensor:
        # note that scores at ~mask are already set to zero
        # and entropy is defined to be zero at x=0 
        # (see https://pytorch.org/docs/stable/special.html)
        return torch.special.entr(scores).sum(dim=-1)
