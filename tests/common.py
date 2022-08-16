import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from transformers.utils import ModelOutput
from types import SimpleNamespace

class PseudoModel(nn.Module):
    """ Pseudo Model used for testing Strategies. Returns input arguments as 
        namespace to allow access using dot-operator 
    """
    def forward(self, **kwargs):
        return SimpleNamespace(**kwargs)
        

class ClassificationModelOutput(ModelOutput):
    """ Output of classification model """
    logits: torch.FloatTensor =None
    loss: torch.FloatTensor =None

class ClassificationModel(nn.Module):
    """ Simple linear classification model used for testing """

    def __init__(self, *args, **kwargs):
        super(ClassificationModel, self).__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x:torch.FloatTensor, labels:torch.LongTensor =None):
        # predict and compute loss
        logits = self.linear(x)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        # return logits and labels as namespace
        return ClassificationModelOutput(
            logits=logits,
            loss=loss
        )

class NamedTensorDataset(TensorDataset):
    """ Helper Dataset similar to `TensorDataset` but returns dictionaries instead of tuples """

    def __init__(self, **named_tensors) -> None:
        self.names, tensors = zip(*named_tensors.items())
        super(NamedTensorDataset, self).__init__(*tensors)
    def __getitem__(self, idx) -> dict:
        tensors = super(NamedTensorDataset, self).__getitem__(idx)
        return dict(zip(self.names, tensors))
