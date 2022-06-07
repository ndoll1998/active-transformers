import torch
from abc import abstractmethod
from .heuristic import ActiveHeuristic

class UncertaintyHeuristic(ActiveHeuristic):
    
    @torch.no_grad()
    def step(self, engine, batch):
        # move batch to device
        batch = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
        # apply model and get output logits
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        # compute uncertainty scores
        scores = self.uncertainty(probs)
        
        if scores.ndim == 1:
            # no average computations needed
            return scores
        # compute average over valid input tokens
        mask = batch['attention_mask']
        scores[~mask, ...] = 0.0
        return scores.flatten(start_dim=1).sum(dim=1) / mask.sum(dim=1)

    @abstractmethod
    def uncertainty(self, probs:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class LeastConfidence(UncertaintyHeuristic):

    def uncertainty(self, probs:torch.Tensor) -> torch.Tensor:
        return (1 - probs.max(dim=-1).values)

class PredictionEntropy(UncertaintyHeuristic):

    def uncertainty(self, probs:torch.Tensor) -> torch.Tensor:
        return torch.special.entr(probs).sum(dim=-1)
