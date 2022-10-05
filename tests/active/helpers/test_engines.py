# disable cuda for tests
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import trainer
from src.active.helpers.engines import Trainer, Events
from src.active.helpers.schedulers import LinearWithWarmup

# import others
import pytest
from src.active.utils.tensor import NamedTensorDataset
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

    def test_non_incremental_trainer(self):
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

    def test_incremental_trainer(self):
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
