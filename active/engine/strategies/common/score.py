import torch
# import ignite events and abstract strategy
from ignite.engine import Events
from .strategy import ActiveStrategy
# other utils
from typing import Optional, Sequence

class ScoreBasedStrategy(ActiveStrategy):
    """ Abstract Base Class for strategies that assign a score to each element
        in the pool. Implements the sample function of the abstract strategy.
        Subclasses must define the following abstract method:
            - `process`: executed by the ignite engine

        Args:
            random_sample (Optional[bool]):
                whether to random sample query according to distribution given by scores computed in `process` function.
                Defaults to False, meaning the elements with the maximal scores are selected.

    """

    def __init__(
        self,
        random_sample:Optional[bool] =False
    ) -> None:
        # initialize abstract strategy
        super(ScoreBasedStrategy, self).__init__()
        # attach output check function
        self.add_event_handler(
            Events.ITERATION_COMPLETED,
            type(self)._check_output
        )
        # whether to random sample according to
        # scores, alternatively use arg-max
        self.random_sample = random_sample

    def _check_output(self) -> None:
        """ Event handler checking the output. Makes sure the output is one-dimensional,
            i.e. assignes a score to each element in the pool.
        """
        # make sure output is a single score for each item in the current batch
        assert self.state.output.ndim == 1, "Expected scores to be one-dimensional but got shape %s" % str(tuple(self.state.output.size()))

    def sample(self, scores:torch.FloatTensor, query_size:int) -> Sequence[int]:
        """ Query samples to label using the strategy

            Args:
                pool (Dataset): pool of data points from which to sample
                query_size (int): number of data points to sample from pool
                batch_size (int): batch size used to process the pool dataset

            Returns:
                indices (Sequence[int]): indices of selected data points in pool
        """
        # how many samples to draw
        k = min(scores.size(0), query_size)
        # draw samples from output scores
        return torch.multinomial(scores, k) if self.random_sample else scores.topk(k=k).indices
