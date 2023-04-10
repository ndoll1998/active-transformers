# no cuda
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from active.engine.strategies.uncertainty.entropy_over_max import (
    EntropyOverMax,
    BinaryEntropyOverMax
)
from tests.utils.strategy import RunTokenStrategyTests

class TestEntropyOverMax(RunTokenStrategyTests):

    def create_strategy(self, model):
        return EntropyOverMax(model)

class TestBinaryEntropyOverMax(RunTokenStrategyTests):

    def create_strategy(self, model):
        return BinaryEntropyOverMax(model)
