import torch
from .common import UncertaintyStrategy

class LeastConfidence(UncertaintyStrategy):
    """ Least Confidence Strategy """

    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        return 1.0 - probs.max(dim=-1).values
