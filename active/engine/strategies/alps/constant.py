from .alps import Alps
from torch.utils.data import Subset
from typing import Sequence

class AlpsConstantEmbeddings(Alps):

    def query(
        self,
        pool:Subset,
        query_size:int,
        batch_size:int
    ) -> Sequence[int]:
        """ Query samples to label using the strategy

            Args:
                pool (Subset): pool of data points from which to sample
                query_size (int): number of data points to sample from pool
                batch_size (int): batch size used to process the pool dataset

            Returns:
                indices (Sequence[int]): indices of selected data points in pool
        """

        assert isinstance(pool, Subset), "Pool must be a subset to infer selected samples from dataset"
        # compute emebddings only once at the first call
        if self.output is None:
            return super(AlpsConstantEmbeddings, self).query(
                pool=pool,
                query_size=query_size,
                batch_size=batch_size
            )

        # remove items that were sampled in previous runs
        assert len(pool.dataset) == self.output.size(0), "Mismatch between embeddings and dataset"
        embeds = self.output[pool.indices]

        # sample from remaining embeddings
        return self.sample(embeds, query_size)
