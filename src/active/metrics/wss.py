import torch
from ignite.metrics import Metric

class WorkSavedOverSampling(Metric):
    
    def reset(self):
        self.value = None

    def update(self, cm:torch.Tensor):
        # extract values from confusion matrix
        n = cm.sum()
        tp = torch.diag(cm)
        fn = cm.sum(dim=1) - tp
        fp = cm.sum(dim=0) - tp
        tn = n - tp - fp - fn
        # compute recall and wss score per class
        R = tp / (tp + fn + 1e-5)
        WSS = (tn + fn) / (n + 1e-5) - (1 - R)
        # average over classes and store
        self.value = WSS.mean()

    def compute(self) -> float:
        assert self.value is not None
        return self.value
