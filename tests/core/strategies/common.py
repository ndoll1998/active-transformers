# import active learning components
from active.core.loop import ActiveLoop
from active.core.strategies.strategy import AbstractStrategy
# import utils and stuff
from active.core.utils.data import NamedTensorDataset
from typing import List, Optional

def _test_strategy_behavior( 
    strategy:AbstractStrategy,
    pool:NamedTensorDataset,
    expected_order:List[int],
    batch_size:Optional[int] =4
):
    """ Helper function to test the behaviour of a strategy. 

        Args:
            strategy (AbstractStrategy): strategy to test
            pool (NamedTensorDataset): pool of data points
            expected_order (List[int]): expected selection order of data points in the pool
            batch_size (Optional[int]): batch size used to process data pool. Defaults to 4.

        Throws AssertionError if strategy doesn't match expectation
    """
    # create active loop
    loop = ActiveLoop(
        pool=pool,
        batch_size=batch_size,
        query_size=1,
        strategy=strategy,
        init_strategy=strategy
    )
    # make sure sampling matches expectation
    for expected_idx, data in zip(expected_order, loop):
        sampled_idx = data.indices[0]
        assert expected_idx == sampled_idx
