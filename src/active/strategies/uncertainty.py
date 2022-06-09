import torch
import torch.nn as nn
import ignite.distributed as idist

from transformers import PreTrainedModel
from .strategy import AbstractStrategy
from .utils import move_to_device

from abc import abstractmethod
from typing import Sequence, Any

class UncertaintyStrategy(AbstractStrategy):
    
    def __init__(
        self,
        model:PreTrainedModel
    ) -> None:
        # initialize strategy
        super(UncertaintyStrategy, self).__init__()
        # move model to available device(s)
        self.model = idist.auto_model(model)
    
    @abstractmethod
    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError()

    def reduce_scores(self, scores:torch.FloatTensor, mask:torch.BoolTensor) -> torch.FloatTensor:
        return scores.flatten(start_dim=1).sum(dim=1) / mask.sum(dim=1)

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
        
        if scores.ndim == 1:
            # no reduction needed
            return scores
        
        # apply mask to scores
        mask = batch['attention_mask']
        scores[~mask, ...] = 0.0
        # reduce scores
        return self.reduce_scores(scores, mask)

    def sample(self, output:torch.FloatTensor, query_size:int) -> Sequence[int]:
        # check shape of scores and get topk indices
        assert output.ndim == 1, "Expected scores to be one-dimensional but got shape %s" % str(tuple(output.size()))
        return output.topk(k=min(query_size, output.size(0))).indices


class LeastConfidence(UncertaintyStrategy):

    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        return 1.0 - probs.max(dim=-1).values


class PredictionEntropy(UncertaintyStrategy):

    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        return torch.special.entr(probs).sum(dim=-1)
