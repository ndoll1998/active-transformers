# disable cuda for tests
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import pytest
# import reward metric
from ignite.metrics import Fbeta
# import active learning components
from active.core.strategies import Random
from active.core.utils.data import NamedTensorDataset
from active.helpers.trainer import Trainer
from active.helpers.evaluator import Evaluator
from active.helpers.engine import ActiveLearningEngine
from active.rl.stream.env import StreamBasedEnv
# import simple model for testing
from tests.common import (
    ClassificationModel,
    ClassificationModelConfig,
    register_classification_model 
)

class TestStreamBasedEnv:
    """ Test cases for the `StreamBasedEnv` """

    def create_sample_env(
        self,
        pool_size:int =100,
        budget:int =10,
        query_size:int =2,
    ) -> StreamBasedEnv:
        
        # register classification model    
        register_classification_model() 

        # create random dataset
        dataset = NamedTensorDataset(
            # actual input to classification model
            x=torch.rand(pool_size, 2),
            # needed for environment
            input_ids=torch.randint(0, 100, size=(pool_size, 1)),
            attention_mask=torch.ones((pool_size, 1), dtype=bool),
            # needed for environment but also used
            # for training classification model
            labels=torch.randint(0, 2, size=(pool_size,))
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
        return StreamBasedEnv(
            budget=budget,
            query_size=query_size,
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
    
    @pytest.mark.parametrize('exec_number', range(10))
    def test_run_until_pool_exhausted(self, exec_number):
 
        # create sample environment
        env = self.create_sample_env(
            pool_size=64,
            budget=100,    # don't stop env based on budget but only on pool exhausted
            query_size=16
        )

        # reset environment
        env.reset()

        # start at 1 as first pool item is already consumed by reset
        # stop one early last step is done seperately outside of loop
        for _ in range(1, 64 - 1):
            _, _, done, _ = env.step(env.action_space.sample())
            assert not done, "Environment reported to be done unexpectedly"

        # should be done as pool is exhaused
        _, _, done, _ = env.step(env.action_space.sample())
        assert done, "Environment should be done as pool should be exhausted"
    
    @pytest.mark.parametrize('exec_number', range(10))
    def test_reward_system(self, exec_number):

        # create sample environment
        env = self.create_sample_env()

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
