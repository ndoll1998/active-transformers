# disable cuda for tests
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
# import active learning components
from src.active.helpers.engines import Trainer
from src.active.engine import ActiveLearningEngine
from src.active.utils.tensor import NamedTensorDataset
# import simple model for testing
from tests.common import (
    ClassificationModel,
    ClassificationModelConfig,
    register_classification_model 
)

class TestActiveLearningEngine:
    """ Test cases for the `ActiveLearningEngine` """

    def test_only_populated_train_data(self):

        # register classification model    
        register_classification_model() 

        # create model and optimizer
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        # create al engine
        engine = ActiveLearningEngine(
            trainer=Trainer(model, optim, incremental=True),
            trainer_run_kwargs=dict(
                max_epochs=5
            ),
            train_val_ratio=1.0 # use all data for training
        )

        # create random dataset
        dataset = NamedTensorDataset(
            x=torch.rand(16, 2),
            labels=torch.randint(0, 2, size=(16,))
        )

        # run engine
        engine.run([dataset])
