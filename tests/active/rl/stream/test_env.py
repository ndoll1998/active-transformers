# disable cuda for tests
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch

from ignite.metrics import Fbeta

from src.active.strategies import Random
from src.active.helpers.engines import Evaluator, Trainer
from src.active.engine import ActiveLearningEngine
from src.active.rl.stream.env import StreamBasedEnv

from tests.common import (
    ClassificationModel,
    ClassificationModelConfig,
    register_classification_model, 
    NamedTensorDataset
)

class TestStreamBasedEnv:
    """ Test cases for the `StreamBasedEnv` """

    def test_reward_system(self):

        # register classification model    
        register_classification_model() 

        # create random dataset
        dataset = NamedTensorDataset(
            # actual input to classification model
            x=torch.rand(100, 2),
            # needed for environment
            input_ids=torch.randint(0, 100, size=(100, 1)),
            attention_mask=torch.ones((100, 1), dtype=bool),
            # needed for envornment but also used
            # for training classification model
            labels=torch.randint(0, 2, size=(100,))
        )

        # create model and optimizer
        model = ClassificationModel(ClassificationModelConfig(2, 2))
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        # create al engine
        engine = ActiveLearningEngine(
            trainer=Trainer(model, optim, incremental=True),
            trainer_run_kwargs=dict(
                max_epochs=5
            )
        )

        # create env
        env = StreamBasedEnv(
            budget=10,
            query_size=2,
            # AL engine and reward metric
            engine=engine,
            metric=Fbeta(beta=1.0, output_transform=Evaluator.get_logits_and_labels),
            # preselection query strategy
            query_strategy=Random(),
            # datasets
            policy_pool_data=dataset,
            model_pool_data=dataset,
            model_test_data=dataset,
            # only needed for observation space
            policy_sequence_length=1,
            model_sequence_length=1,
            max_num_labels=2
        )

        # test env for multiple runs/episodes
        for _ in range(10):

            # reset env
            env.reset()

            rewards = []
            # run a full episode and track rewards
            done = False
            while not done:
                _, r, done, _ = env.step(env.action_space.sample())
                rewards.append(r)

            # get final metric value after episode
            final_metric = env.state.prev_metric
            # check reward sum
            assert sum(rewards) == final_metric, "Rewards of an episode must sum up to the final reward metric!"
