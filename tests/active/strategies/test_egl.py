import torch
import torch.nn as nn
import torch.nn.functional as F
# import data helpers
from src.active.utils.data import NamedTensorDataset
# import common components
from tests.common import (
    ClassificationModel, 
    ClassificationModelConfig,
)
# import egl strategies
from src.active.strategies.egl import (
    EglByTopK,
    EglBySampling
)

class TestExpectedGradientLength:
    """ Test cases for Expected Gradient Length """

    def test_expected_grad_norm_by_top_k_labels(self):
        """ Test if EGL strategy runs without errors 
            Scenario:
                - Task: Sequence Classification
                - Strategy: EglByTopK
        """
        # create model and input data 
        model = ClassificationModel(ClassificationModelConfig(4, 2))
        x = torch.rand(8, 4)
        # create and run strategy
        strategy = EglByTopK(model, k=3)
        strategy.query(
            pool=NamedTensorDataset(x=x),
            query_size=4,
            batch_size=16
        )
    
    def test_expected_grad_norm_by_random_sampling(self):
        """ Test if EGL strategy runs without errors 
            Scenario:
                - Task: Token Classification
                - Strategy: EglBySampling
        """
        # create model and input data 
        model = ClassificationModel(ClassificationModelConfig(4, 2))
        x = torch.rand(8, 8, 4)
        # create and run strategy
        strategy = EglBySampling(model, k=3)
        strategy.query(
            pool=NamedTensorDataset(x=x),
            query_size=4,
            batch_size=16
        )
