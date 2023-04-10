import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from active.engine.strategies.alps.constant import AlpsConstantEmbeddings
from tests.utils.strategy import RunStrategyTests

class TestAlps(RunStrategyTests):

    def create_strategy(self, model):
        return AlpsConstantEmbeddings(model)
