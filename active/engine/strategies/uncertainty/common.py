import torch
import torch.nn as nn
import ignite.distributed as idist

from transformers import PreTrainedModel
from active.utils.data import move_to_device
from ..common.score import ScoreBasedStrategy

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
        if 'attention_mask' in batch:
            mask = batch['attention_mask'].bool()
            mask = mask.reshape(mask.shape + (1,) * (scores.ndim - mask.ndim))
            assert mask.ndim == scores.ndim == ignore_mask.ndim
            # apply mask to scores
            scores.masked_fill_(~mask, 0.0)
        else:
            mask = True

        # reduce scores
        return self.reduce_scores(scores, mask & ~ignore_mask)
