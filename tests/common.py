import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from transformers import (
    AutoModel,
    PretrainedConfig, 
    PreTrainedModel
)
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

class ClassificationModelConfig(PretrainedConfig):
    """ Config of classification model """

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

class ClassificationModel(PreTrainedModel):
    """ Simple linear classification model used for testing """

    config_class=ClassificationModelConfig

    def __init__(self, config:ClassificationModelConfig):
        super(ClassificationModel, self).__init__(config)
        self.linear = nn.Linear(
            in_features=config.in_features,
            out_features=config.out_features
        )

    def init_weights(self):
        pass

    def forward(self, x:torch.FloatTensor, labels:torch.LongTensor =None, **kwargs):
        # predict and compute loss
        logits = self.linear(x)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        # return logits and labels as namespace
        return ClassificationModelOutput(
            logits=logits,
            loss=loss
        )

def register_classification_model():
    """ Register Classification Model in transformer AutoModel """
    AutoModel.register(ClassificationModelConfig, nn.Linear)
