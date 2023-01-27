# disable cuda for tests
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import trainer
from active.helpers.trainer import Trainer, State, Events
from active.helpers.schedulers import LinearWithWarmup

# import others
import pytest
from active.core.utils.data import NamedTensorDataset
from tests.common import (
    ClassificationModel,
    ClassificationModelConfig,
    register_classification_model,
)

class TestTrainer:
    """ Test cases for the `Trainer` class """

    def create_random_loader(self, n:int, n_feats:int):
        # build sample dataset
        dataset = NamedTensorDataset(
            x=torch.rand(n, n_feats),
            labels=torch.zeros(n, dtype=int)
        )
        # train dataloader
        return DataLoader(dataset, batch_size=2, shuffle=True)

    @pytest.mark.parametrize('exec_number', range(5))
    def test_non_incremental_trainer(self, exec_number):
        """ Test non-incremental trainer setup, i.e. test if the model is reset
            on start of a new run.
        """
        # register classification model in transformers AutoModel
        # necessary because the auto-model functionality is used in
        # the trainer to extract the encoder from the model 
        register_classification_model()

        # create dataloader
        loader = self.create_random_loader(4, 2)

        # create model, optimizer and scheduler
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda *_: 1.0)

        # get initial weights of the model
        init_weight = model.linear.weight.clone()
        init_bias = model.linear.bias.clone()

        # create trainer
        trainer = Trainer(
            model=model,
            optim=optim,
            scheduler=scheduler,
        )

        # on start make sure the weights are the initial weights
        @trainer.on(Events.STARTED)
        def check_init_weights(engine):
            assert (model.linear.weight == init_weight).all()
            assert (model.linear.bias == init_bias).all()

        # on finish make sure the model weights are updated
        @trainer.on(Events.COMPLETED)
        def check_updated_weights(engine):
            assert (model.linear.weight != init_weight).any()
            assert (model.linear.bias != init_bias).any()

        # train model
        for _ in range(2):
            trainer.run(loader, max_epochs=2)

    @pytest.mark.parametrize('exec_number', range(5))
    def test_incremental_trainer(self, exec_number):
        """ Test incremental trainer setup, i.e. test if the model of the
            current run matches the output model of the previous one
        """
        # register classification model in transformers AutoModel
        # necessary because the auto-model functionality is used in
        # the trainer to extract the encoder from the model 
        register_classification_model()

        # create dataloader
        loader = self.create_random_loader(4, 2)

        # create model, optimizer and scheduler
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda *_: 1.0)

        # get initial weights of the model
        cur_weight = model.linear.weight.clone()
        cur_bias = model.linear.bias.clone()

        # create trainer
        trainer = Trainer(
            model=model,
            optim=optim,
            scheduler=scheduler,
            incremental=True,
        )

        # on start make sure the weights are the initial weights
        @trainer.on(Events.STARTED)
        def check_init_weights(engine):
            assert (model.linear.weight == cur_weight).all()
            assert (model.linear.bias == cur_bias).all()

        # on finish update the current weights
        @trainer.on(Events.COMPLETED)
        def update_weights(engine):
            cur_weight[:] = model.linear.weight[:]
            cur_bias[:] = model.linear.bias[:]

        # train model
        for _ in range(2):
            trainer.run(loader, max_epochs=2)

    def test_prepare_scheduler(self):
        """ Test if the learning rate scheduler is prepared correctly, i.e.
            is the number of training steps set
        """
        # register classification model in transformers AutoModel
        # necessary because the auto-model functionality is used in
        # the trainer to extract the encoder from the model 
        register_classification_model()

        # create model, optimizer and scheduler
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        # catch expected warning
        with pytest.warns(UserWarning, match="Number of training steps not set!"):
            scheduler = LinearWithWarmup(optim, warmup_proportion=0.1)

        # create trainer
        trainer = Trainer(
            model=model,
            optim=optim,
            scheduler=scheduler,
            incremental=True,
        )

        for (num_samples, num_epochs) in [(4, 1), (16, 1), (4, 4), (16, 4)]:
            # create dataloader
            loader = self.create_random_loader(num_samples, 2)
            trainer.run(loader, max_epochs=num_epochs)
            # test scheduler training steps
            assert scheduler.num_training_steps == len(loader) * num_epochs

    def test_min_epoch_length(self):

        # register classification model in transformers AutoModel
        # necessary because the auto-model functionality is used in
        # the trainer to extract the encoder from the model 
        register_classification_model()

        # create model and optimizer
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        # create trainer
        trainer = Trainer(model=model, optim=optim)

        # test behavior for epoch length induced by data
        state = trainer.run([0, 0, 0], max_epochs=0)
        assert state.epoch_length == 3, "Expected epoch length to match length of data"

        # check behavior for specified epoch length
        state = trainer.run([], max_epochs=0, epoch_length=5)
        assert state.epoch_length == 5, "Expected epoch length to match argument"

        # check behavior for specified epoch length lower than minimum epoch length
        state = trainer.run([], max_epochs=0, epoch_length=5, min_epoch_length=8)
        assert state.epoch_length == 8, "Expected epoch length to be overwritten by minimum epoch length"

        # check behavior for specified epoch length greater than minimum epoch length
        state = trainer.run([], max_epochs=0, epoch_length=5, min_epoch_length=3)
        assert state.epoch_length == 5, "Expected epoch length despite minimum epoch length set"

        # check behavior for specified induced epoch length lower than mininum epoch length
        state = trainer.run([0, 0, 0], max_epochs=0, min_epoch_length=8)
        assert state.epoch_length == 8, "Expected induced epoch length to be overwritten by minimum epoch length"

        # check behavior for specified induced epoch length greater than minimum epoch length
        state = trainer.run([0, 0, 0], max_epochs=0, min_epoch_length=2)
        assert state.epoch_length == 3, "Expected epoch length to induced epoch length despite minimum epoch length set"
