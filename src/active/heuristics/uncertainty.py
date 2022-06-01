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
        return self.uncertainty(probs)

    @abstractmethod
    def uncertainty(self, probs:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class LeastConfidence(UncertaintyHeuristic):

    def uncertainty(self, probs:torch.Tensor) -> torch.Tensor:
        return (1 - probs.max(dim=-1).values)
