# disable cuda for tests
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
# import active learning components
from active.utils.data import NamedTensorDataset
from active.engine.engine import ActiveEngine
# import simple model for testing
from tests.utils.trainer import SimpleTrainer
from tests.utils.modeling import (
    ClassificationModel,
    ClassificationModelConfig,
    register_classification_model
)

class TestActiveLearningEngine:
    """ Test cases for the `ActiveLearningEngine` """

    def test_only_populated_train_data(self):

        # register classification model    
        register_classification_model()

        # create model and engine
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        engine = ActiveEngine(
            trainer=SimpleTrainer(model),
            train_val_split=1.0 # use all data for training
        )

        # create random dataset
        dataset = NamedTensorDataset(
            x=torch.rand(16, 2),
            labels=torch.randint(0, 2, size=(16,))
        )

        # step engine and check data sizes
        engine.step(dataset)
        assert engine.num_train_samples == 16
        assert engine.num_val_samples == 0

        # step engine and check data sizes
        engine.step(dataset)
        assert engine.num_train_samples == 32
        assert engine.num_val_samples == 0

    def test_populate_train_and_val_data(self):

        # register classification model    
        register_classification_model()

        # create model and optimizer
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        # create model and engine
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        engine = ActiveEngine(
            trainer=SimpleTrainer(model),
            train_val_split=0.8 # use 80% for training
        )

        # create random dataset
        dataset = NamedTensorDataset(
            x=torch.rand(16, 2),
            labels=torch.randint(0, 2, size=(16,))
        )

        # step engine and check data sizes
        engine.step(dataset)
        assert engine.num_train_samples == 13
        assert engine.num_val_samples == 3

        # step engine and check data sizes
        engine.step(dataset)
        assert engine.num_train_samples == 26
        assert engine.num_val_samples == 6

        # step engine and check data sizes
        engine.step(dataset)
        assert engine.num_train_samples == 38
        assert engine.num_val_samples == 10
