# no cuda
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from active.engine.strategies.uncertainty.prediction_entropy import PredictionEntropy
from tests.utils.strategy import RunStrategyTests

class TestPredictionEntropy(RunStrategyTests):

    def create_strategy(self, model):
        return PredictionEntropy(model)
