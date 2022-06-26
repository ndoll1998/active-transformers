# import active learning components
from src.active.loop import ActiveLoop
from src.active.strategies.strategy import AbstractStrategy
# import utils and stuff
from tests.common import NamedTensorDataset
from typing import List

def _test_strategy_behavior( 
    strategy:AbstractStrategy,
    pool:NamedTensorDataset,
    expected_order:List[int]
):
    """ Helper function to test the behaviour of a strategy. 

        Args:
            strategy (AbstractStrategy): strategy to test
            pool (NamedTensorDataset): pool of data points
            expected_order (List[int]): expected selection order of data points in the pool

        Throws AssertionError if strategy doesn't match expectation
    """
    # create active loop
    loop = ActiveLoop(
        pool=pool,
        batch_size=len(pool),
        query_size=1,
        strategy=strategy,
        init_strategy=strategy
    )
    # make sure sampling matches expectation
    for expected_idx, data in zip(expected_order, loop):
        sampled_idx = data.indices[0]
        assert expected_idx == sampled_idx
