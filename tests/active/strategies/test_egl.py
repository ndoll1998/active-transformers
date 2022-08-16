import torch
import torch.nn as nn
import torch.nn.functional as F
# import common components
from tests.common import ClassificationModel, NamedTensorDataset
# import egl strategy and gradient norm
from src.active.strategies.egl import (
    GoodfellowGradientNorm,
    EglByTopK,
    EglBySampling
)

class TestExpectedGradientLength:
    """ Test cases for Expected Gradient Length """

    def test_goodfellow_grad_norm(self):
        """ Test Gradient Norm Computation using Goodfellow """        
        # create sample model with linear components
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 4),
            nn.Sigmoid(),
            nn.Linear(4, 2)
        )
        # some sample input data
        x = torch.rand(8, 2)
        y = (torch.rand(8) > 0.5).long()

        # attach gradient norm hooks
        with GoodfellowGradientNorm(model) as grad_norm:

            model.zero_grad()
            # compute gradient norm using goodfellow
            # use summation to avoid average over batches
            # this matches the computation with batch size 1
            F.cross_entropy(model(x), y, reduction='sum').backward()
            goodfellow_norm = grad_norm.compute()

            # compute norms manually to compare
            for i in range(x.size(0)):
                # pass through model
                model.zero_grad()
                F.cross_entropy(model(x[i:i+1]), y[i:i+1]).backward()
                # compute norm and check
                manual_norm = sum((p.grad * p.grad).sum() for n, p in model.named_parameters())
                assert torch.allclose(manual_norm, goodfellow_norm[i]), "%.04f != %.04f" % (manual_norm, goodfellow_norm[i])
    
    def test_goodfellow_grad_norm_multiple_backward_passes(self):
        """ Test multi-backward pass of Goodfellow Norm Computation """
        # create sample model with linear components
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 4),
            nn.Sigmoid(),
            nn.Linear(4, 2)
        )
        # some sample input data
        x = torch.rand(8, 2)

        # attach gradient norm hooks
        with GoodfellowGradientNorm(model) as grad_norm:

            # define reductions
            # use only summations again to match the
            # computations done with batch size set to 1
            # (compare `test_goodfellow_grad_norm`)
            reductions = [
                lambda t: t.sum(),
                lambda t: t.sin().sum(),
                lambda t: t.exp().sum()
            ]

            model.zero_grad()
            # multiple backward passes with different reductions
            y = model.forward(x)
            for fn in reductions:
                fn(y).backward(retain_graph=True)
            # compute gradient norm using goodfellow
            goodfellow_norm = grad_norm.compute()

            # compute norms manually to compare
            for i in range(x.size(0)):
                for j, fn in enumerate(reductions):
                    # pass through model
                    model.zero_grad()
                    fn(model.forward(x[i:i+1])).backward()
                    # compute norm
                    manual_norm = sum((p.grad * p.grad).sum() for n, p in model.named_parameters())
                    assert torch.allclose(manual_norm, goodfellow_norm[i, j]), "%.04f != %.04f" % (manual_norm, goodfellow_norm[i, j])

    def test_expected_grad_norm_by_top_k_labels(self):
        """ Test if EGL strategy runs without errors 
            Scenario:
                - Task: Sequence Classification
                - Strategy: EglByTopK
        """
        # create model and input data 
        model = ClassificationModel(4, 2)
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
        model = ClassificationModel(4, 2)
        x = torch.rand(8, 8, 4)
        # create and run strategy
        strategy = EglBySampling(model, k=3)
        strategy.query(
            pool=NamedTensorDataset(x=x),
            query_size=4,
            batch_size=16
        )
