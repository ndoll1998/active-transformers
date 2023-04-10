import torch
from .common import UncertaintyStrategy

class PredictionEntropy(UncertaintyStrategy):
    """ Prediction Entropy Strategy """

    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        return torch.special.entr(probs).sum(dim=-1)
