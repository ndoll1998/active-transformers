import torch
import transformers
# import active learning components
from active.engine.loop import ActiveLoop
from active.engine.strategies.common.strategy import ActiveStrategy
# import utils and stuff
from active.utils.data import NamedTensorDataset
from abc import ABC, abstractmethod
from typing import List

class RunTokenStrategyTests(ABC):

    @abstractmethod
    def create_strategy(self, model:transformers.PreTrainedModel):
        pass

    def test_run_strategy_token_classification(self):
        # create distil bert model
        config = transformers.DistilBertConfig()
        model = transformers.DistilBertForTokenClassification(config)
        # create strategy
        strategy = self.create_strategy(model)
        _test_run_strategy(strategy, vocab_size=config.vocab_size)

class RunSequenceStrategyTests(ABC):

    @abstractmethod
    def create_strategy(self, model:transformers.PreTrainedModel):
        pass

    def test_run_strategy_sequence_classification(self):
        # create distil bert model
        config = transformers.DistilBertConfig()
        model = transformers.DistilBertForSequenceClassification(config)
        # create strategy
        strategy = self.create_strategy(model)
        _test_run_strategy(strategy, vocab_size=config.vocab_size)

class RunStrategyTests(RunSequenceStrategyTests, RunTokenStrategyTests):
    pass

def _test_run_strategy(
    strategy:ActiveStrategy,
    vocab_size:int,
    batch_size:int =4
):
    """ Helper function running a strategy on a small sample dataset """
    # create dataset
    data = NamedTensorDataset(input_ids=torch.randint(0, vocab_size, size=(4, 16)))
    data = torch.utils.data.Subset(data, list(range(len(data))))
    # run strategy
    strategy.query(data, query_size=2, batch_size=batch_size)

def _test_strategy_behavior(
    strategy:ActiveStrategy,
    pool:NamedTensorDataset,
    expected_order:List[int],
    batch_size:int =4
):
    """ Helper function to test the behaviour of a strategy.

        Args:
            strategy (AbstractStrategy): strategy to test
            pool (NamedTensorDataset): pool of data points
            expected_order (List[int]): expected selection order of data points in the pool
            batch_size (int): batch size used to process data pool. Defaults to 4.

        Throws AssertionError if strategy doesn't match expectation
    """
    # create active loop
    loop = ActiveLoop(
        pool=pool,
        batch_size=batch_size,
        query_size=1,
        strategy=strategy
    )
    # make sure sampling matches expectation
    for expected_idx, data in zip(expected_order, loop):
        sampled_idx = data.indices[0]
        assert expected_idx == sampled_idx
