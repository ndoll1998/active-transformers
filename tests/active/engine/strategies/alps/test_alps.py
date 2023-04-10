import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from active.engine.strategies.alps.alps import Alps
from tests.utils.strategy import RunStrategyTests

class TestAlps(RunStrategyTests):

    def create_strategy(self, model):
        return Alps(model)
