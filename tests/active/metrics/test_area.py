import numpy as np
from src.active.metrics import AreaUnderLearningCurve

class TestAreaUnderLearningCurve:
    """ Test for area under learning curve metric """

    def test_normalization(self):
        # linear sample curve
        x = np.arange(0, 100, step=5)
        y = np.linspace(0, 1, x.shape[0])
        # create metric
        area = AreaUnderLearningCurve()
        
        # reset, update with all pairs defining curve
        # and compute final metric score
        area.reset()
        list(map(area.update, zip(x, y)))
        valA = area.compute()
        
        # reset, update with all pairs defining curve
        # and compute final metric score
        area.reset()
        list(map(area.update, zip(x * 30, y)))
        valB = area.compute()
        
        assert valA == valB

    def test_linear(self):
        # linear sample curve
        x = np.arange(0, 100, step=5)
        y = np.linspace(0, 1, x.shape[0])
        # create metric
        area = AreaUnderLearningCurve()
        # reset, update with all pairs defining curve
        # and compute final metric score
        area.reset()
        list(map(area.update, zip(x, y)))
        val = area.compute()
        # should be 0.5 since all linear
        assert val == 0.5
        
    def test_quadratic(self):
        # quadratic curve
        x = np.linspace(0, 1, 100)
        y = x**2
        # create metric
        area = AreaUnderLearningCurve()
        # reset, update with all pairs defining curve
        # and compute final metric score
        area.reset()
        list(map(area.update, zip(x, y)))
        val = area.compute()
        # should be < 0.5 since "worse than" linear
        # note that we are in range [0, 1]
        assert val < 0.5
    
    def test_sqrt(self):
        # anti-quadratic curve
        x = np.linspace(0, 1, 100)
        y = np.sqrt(x)
        # create metric
        area = AreaUnderLearningCurve()
        # reset, update with all pairs defining curve
        # and compute final metric score
        area.reset()
        list(map(area.update, zip(x, y)))
        val = area.compute()
        # should be > 0.5 since "better than" linear
        assert val > 0.5
