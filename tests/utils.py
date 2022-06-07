import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

class ClassificationModel(nn.Linear):
    """ Simple linear classification model used for testing """

    def forward(self, x:torch.FloatTensor, labels:torch.LongTensor):
        # predict and compute loss
        logits = super(ClassificationModel, self).forward(x)
        loss = F.cross_entropy(logits, labels)
        # return logits and labels as dict
        return {
            'logits': logits,
            'loss': loss
        }

class NamedTensorDataset(TensorDataset):
    """ Helper Dataset similar to `TensorDataset` but returns dictionaries instead of tuples """

    def __init__(self, **named_tensors) -> None:
        self.names, tensors = zip(*named_tensors.items())
        super(NamedTensorDataset, self).__init__(*tensors)
    def __getitem__(self, idx) -> dict:
        tensors = super(NamedTensorDataset, self).__getitem__(idx)
        return dict(zip(self.names, tensors))
