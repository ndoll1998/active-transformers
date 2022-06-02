# torch and transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel
# ignite
from ignite.engine import Engine
from ignite.handlers.stores import EpochOutputStore
import ignite.distributed as idist
# others
from abc import ABC, abstractmethod
from typing import Union

class ActiveHeuristic(Engine, ABC):

    def __init__(
        self, 
        model:PreTrainedModel
    ) -> None:
        # save model
        self.model = idist.auto_model(model) if model is not None else None
        # initialize engine
        super(ActiveHeuristic, self).__init__(self.step)
        # add output storage handler to self
        self.output_store = EpochOutputStore(
            # move scores to cpu for storing
            output_transform=lambda scores: scores.detach().cpu()
        )
        self.output_store.attach(self)

    @property
    def device(self) -> torch.device:
        return idist.device()

    @abstractmethod
    def step(self, engine, batch) -> torch.Tensor:
        """ Compute the ranking scores for elements of the given batch """
        raise NotImplementedError()

    def compute_scores(self, dataset:Dataset, batch_size:int) -> torch.Tensor:
        # create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        # reset output store
        self.output_store.reset()
        # run the engine on the given data
        self.run(loader)
        # get the ranking scores from the output store
        scores = self.output_store.data
        scores = torch.cat(scores, dim=0)
        assert scores.size(0) == len(dataset), "Size Mismatch between scores (%i) and dataset (%i)" % (scores.size(0), len(dataset))
        assert scores.ndim == 1, "Expected scores to have only a single dimension but got shape %s" % scores.size()
        # return
        return scores
