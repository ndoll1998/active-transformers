from __future__ import annotations
import os
# active
from active import helpers
from active import strategies
from active.rl import stream
from active.rl import extractors
# metrics and evaluator (only used for output transform)
from ignite.metrics import Fbeta
from active.core.metrics.area import AreaUnderLearningCurve
from active.helpers.evaluator import Evaluator
# configs
from active.scripts.run_train import Task, ModelConfig
from active.scripts.run_active import AlExperimentConfig
# ray
import ray
from ray.tune import Tuner
from ray.air import RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
# callbacks
from ray.rllib.algorithms.callbacks import MultiCallbacks
from active.rl.callbacks import (
    CustomMetricsFromEnvCallbacks,
    LoggingCallbacks
)
# others
import gym
from enum import Enum
from pydantic import BaseModel, validator
from typing import Type, List, Dict, Any, Union

def load_model_and_policy_pool_data(
    experiment:AlExperimentConfig,
    policy:FeatureExtractorConfig
):
    """ Helper function to load and preprocess pool datasets
        Note that this is non-trivial as the pool dataset has
        to be prepared for both the model and the policy's
        feature extractor while keeping the index-relationship
        between items (i.e. policy_pool[i] ~ model_pool[i]).
    """
    # get model and policy tokenizer
    model_tokenizer = experiment.model.tokenizer
    policy_tokenizer = policy.tokenizer
    # pad token ids
    model_pad_token_id = model_tokenizer.pad_token_id
    policy_pad_token_id = policy_tokenizer.pad_token_id   

    # disable filtering in data processing as this might lead to
    # differences in the model vs policy dataset
    model_data = experiment.data.copy(update={'min_sequence_length': 0})
    # create policy data config by updating only the maximum
    # sequence length but keeping minimum sequence length of zero set above
    policy_data = model_data.copy(update={'max_sequence_length': policy.max_sequence_length})

    # load and preprocess pool datasets
    model_pool = model_data.load_dataset(model_tokenizer, split={'pool': 'train'})['pool']
    policy_pool = policy_data.load_dataset(policy_tokenizer, split={'pool': 'train'})['pool']

    # undo type formatting to allow combining the datasets
    model_pool.set_format()
    policy_pool.set_format()
    # build combined dataset for filtering
    combined_pool = model_pool \
        .add_column('policy_input_ids', policy_pool['input_ids']) \
        .add_column('policy_attention_mask', policy_pool['attention_mask']) \
        .add_column('policy_labels', policy_pool['labels'])
    # use numpy arrays for filtering
    combined_pool.set_format(type='np')
    # make sure all pools are of the same size
    assert len(model_pool) == len(combined_pool) == len(policy_pool)
    
    # filter
    filter_ = lambda e: ((e['input_ids'] != model_pad_token_id).sum() > experiment.data.min_sequence_length) or \
        ((e['policy_input_ids'] != policy_pad_token_id).sum() > policy.min_sequence_length)
    combined_pool.filter(filter_, batched=False, desc='Filter')
    # split combined dataset back into model and policy datasets
    model_pool = combined_pool.remove_columns(['policy_input_ids', 'policy_attention_mask', 'policy_labels'])
    policy_pool = combined_pool.remove_columns(['input_ids', 'attention_mask', 'labels']) \
        .rename_column('policy_input_ids', 'input_ids') \
        .rename_column('policy_attention_mask', 'attention_mask') \
        .rename_column('policy_labels', 'labels')

    # set formats
    model_pool.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    policy_pool.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # return pool datasets
    return model_pool, policy_pool
    
class RayStreamBasedEnv(stream.env.StreamBasedEnv):
    """ Stream-based environment slightly modified to be
        compatible with ray.rllib 

        Main changes/features to the default stream-based env are:

            (1): `__init__` expects a dictionary containing all experiment configurations
                 and a feature-extractor configuration instance
            (2): the experiment configuration is selected by the worker index,
                 i.e. all environments of the same worker run the same experiment
            (3): model and data loading/preparation are delayed until first
                 call of `reset` to avoid overhead of driver copy
                 (see `Expensive Environments` at https://docs.ray.io/en/latest/rllib/rllib-env.html)
    """

    def __init__(self, env_config):
        # flag to mark the environment as initialized
        self._is_initialized = False
        # save configs for actual initialization in reset
        self.experiment = env_config['experiment_pool'][env_config.worker_index-1]
        self.feature_extractor = env_config['feature_extractor']
        # dummy initialize environment, set low-resource values
        # expecially those needed for the observation space
        super(RayStreamBasedEnv, self).__init__(
            # pass dummy arguments
            budget=0,
            query_size=0,
            engine=None,
            query_strategy=None,
            policy_pool_data=[],
            model_pool_data=[],
            model_test_data=[],
            metric=None,
            # as data is not given the layout needs to be set explicitly
            policy_sequence_length=self.feature_extractor.max_sequence_length,
            model_sequence_length=env_config['model_max_sequence_length'],
            max_num_labels=env_config['model_max_num_labels']
        )

    def reset(self):

        # initialize on first call
        if not self._is_initialized:
            # get the observation space before initialization
            obs_space = self.observation_space
            # load the pool datasets
            model_pool_data, policy_pool_data = load_model_and_policy_pool_data(
                experiment=self.experiment,
                policy=self.feature_extractor
            )
            # load the model test data
            # compared to the pool data this is easy as only
            # the model needs has access to the test dataset
            model_test_data = self.experiment.load_dataset(split={'test': 'test'})['test']
            # re-initialize with actual values
            super(RayStreamBasedEnv, self).__init__(
                budget=self.experiment.active.budget,
                query_size=self.experiment.active.query_size,
                # keep metric
                metric=Fbeta(beta=1, output_transform=Evaluator.get_logits_and_labels),
                # build active learning engine
                engine=helpers.engine.ActiveLearningEngine(
                    trainer=self.experiment.build_trainer(),
                    trainer_run_kwargs=self.experiment.trainer.run_kwargs,
                    train_batch_size=self.experiment.trainer.batch_size,
                    eval_batch_size=self.experiment.trainer.batch_size,
                    train_val_ratio=0.9
                ),
                # set query strategy
                query_strategy=strategies.Random(),
                # pass the preprocessed datasets
                policy_pool_data=policy_pool_data,
                model_pool_data=model_pool_data,
                model_test_data=model_test_data,
                # keep these values around to make sure the
                # observation space doesn't change due to re-initialization
                policy_sequence_length=self._policy_seq_len,
                model_sequence_length=self._model_max_seq_len,
                max_num_labels=self._model_max_num_labels
            )
            # add area metric to engine
            area = AreaUnderLearningCurve(
                output_transform=lambda _: (
                    # reusing the reward metric which
                    # is set to be the F-score
                    self.engine.state.iteration,
                    self.evaluator.state.metrics['__reward_metric']
                )
            )
            area.attach(self.engine, "Area(F)")
            # make sure the observation spaces match
            assert obs_space == self.observation_space, "Detected change in observation space after environment initialization"
            # mark as initialized
            self._is_initialized = True

        # reset
        return super(RayStreamBasedEnv, self).reset()

class Approach(Enum):
    """ Enum defining supported active learning approaches: 
            (1) stream:
                data is presented to the strategy/agent as a stream.
                Agent decides whether to annoate or to dicard datapoints.
            (2) pool:
                data is presented to strategy/agent as pool.
                Agent selects subset to annotate next.
    """
    STREAM = 'stream'
    POOL = 'pool'

    @property
    def env_type(self) -> Type[gym.Env]:
        if self == Approach.STREAM:
            return RayStreamBasedEnv
        elif self == Approach.POOL:
            raise NotImplementedError()

class FeatureExtractor(Enum):
    """ Enum defining supported feature extractors:
            (1) transformer:
                simple transformer model as feature extractor.
                Only uses input sequence and does not depend on model output.
    """
    TRANSFORMER = 'transformer'

    @property
    def type(self) -> Type[extractors.FeatureExtractor]:
        if self is FeatureExtractor.TRANSFORMER:
            return extractors.TransformerFeatureExtractor        

class Algorithm(Enum):
    """ Enum defining supported algorithms:
            (1) dqn: Deep-Q-Network
    """
    DQN = "dqn"
    PPO = "ppo"

    @property
    def type(self) -> Type[ray.rllib.algorithms.Algorithm]:
        if self is Algorithm.DQN:
            from ray.rllib.algorithms.dqn import DQN
            return DQN
        if self is Algorithm.PPO:
            from ray.rllib.algorithms.ppo import PPO
            return PPO

class Model(Enum):
    """ Enum definig supported models:
            (1) stream/dqn-model: simple Deep-Q-Learning Model for stream-based approach
    """
    STREAM_DQN_MODEL = "stream/dqn-model"

    @property
    def type(self) -> Type[ray.rllib.models.torch.torch_modelv2.TorchModelV2]:
        if self is Model.STREAM_DQN_MODEL:
            return stream.model.DQNModel

class FeatureExtractorConfig(ModelConfig):
    """ Feature Extractor Configuration Model """
    # task must be bio tagging
    task:Task =Task.BIO_TAGGING

    # feature extractor type used by policy
    feature_extractor_type:FeatureExtractor
    # sequence length
    max_sequence_length:int
    min_sequence_length:int

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            'feature_extractor_type': self.feature_extractor_type.type,
            'feature_extractor_config': {
                'pretrained_ckpt': self.pretrained_ckpt
            }
        }

class AlgorithmConfig(BaseModel):
    """ Algorithm Configuration Model"""
    # algorithm to use and algorithm specific parameters
    algo:Algorithm
    algo_config:Dict[str, Any]
    # general algorithm config
    rollout_fragment_length:int
    train_batch_size:int
    # number gpus the algorithm has access to
    num_gpus:int

    # convergence criteria
    max_timesteps:int

    @property
    def ray_config(self) -> Dict[str, Any]:
        return {
            "num_gpus": self.num_gpus,
            "rollout_fragment_length": self.rollout_fragment_length,
            "train_batch_size": self.train_batch_size,
            **self.algo_config
        }

class WorkersConfig(BaseModel):
    """ Workers Configuration Model """
    num_workers:int
    num_gpus_per_worker:float
    num_envs_per_worker:int
    remote_worker_envs:bool =False

    @property
    def ray_config(self) -> Dict[str, Any]:
        return self.dict()

def parse_experiment_config(cls, value) -> AlExperimentConfig:
    # check if value is a config instance
    if isinstance(value, AlExperimentConfig):
        return value
    # otherwise it should be a path to a valid config
    if not os.path.isfile(value):
        raise ValueError("Experiment configuration file not found: %s" % value)
    # load the config
    config = AlExperimentConfig.parse_file(value)
    # only supports bio tagging tasks
    if config.task is not Task.BIO_TAGGING:
        raise ValueError("Currently only bio tagging tasks are supported")   
    # return the loaded configuration
    return config 

class EvaluationConfig(BaseModel):
    """ Evaluation Configuration Model """
    # passed from RlExperiment
    _feature_extractor:FeatureExtractor =None
    # basic evaluation setup
    evaluation_interval:int
    evaluation_duration:int
    evaluation_num_workers:int
    # evaluation environemt
    env_config:Union[str, AlExperimentConfig]

    # validator to parse config
    _parse_env_config = validator('env_config', allow_reuse=True)(parse_experiment_config)

    @property
    def ray_config(self) -> Dict[str, Any]:
        return {
            # basic evaluation settings
            "evaluation_interval": self.evaluation_interval,
            "evaluation_duration": self.evaluation_duration,
            "evaluation_num_workers": self.evaluation_num_workers,
            # set evaluation environment
            "evaluation_config": {
                "env_config": {
                    "experiment": self.env_config,
                    "feature_extractor": self._feature_extractor
                }
            }
        }

class RlExperimentConfig(BaseModel):
    """ Reinforcement Learning Experiment Configuration Model"""
    # approach to use
    approach:Approach
    # algorithm and worker setup
    algorithm:AlgorithmConfig
    workers:WorkersConfig
    
    # model config
    model_type:Model
    feature_extractor:FeatureExtractorConfig
    # experiments -> environment setups
    env_configs:List[Union[str, AlExperimentConfig]]    

    # evaluation config
    evaluation:EvaluationConfig

    # validator to parse configs
    _parse_env_configs = validator('env_configs', each_item=True, allow_reuse=True)(parse_experiment_config)

    @validator('evaluation')
    def _pass_feature_extractor_to_evaluation(cls, value, values):
        assert 'feature_extractor' in values
        if isinstance(value, BaseModel):
            return value.copy(update={'_feature_extractor': values.get('feature_extractor')})
        elif isinstance(value, dict):
            return value | {'_feature_extractor': values.get('feature_extractor')}

    @validator('env_configs')
    def _check_one_env_config_per_worker(cls, v, values):
        assert 'workers' in values
        workers = values.get('workers')
        # make sure env configs align with number of workers
        if max(workers.num_workers, 1) != len(v):
            raise ValueError(
                "Must specify exactly one environment config (%i) per worker (%i)" % \
                (len(v), max(1, workers.num_workers))
            )
        # return env configs
        return v

    @property
    def ray_config(self) -> Dict[str, Any]:
        # get model maximum sequence length and number of labels over all
        # specified experiment configs including the evaluation experiment
        all_env_configs = self.env_configs + [self.evaluation.env_config]
        max_seq_length = max(config.data.max_sequence_length for config in all_env_configs)
        max_num_labels = max(map(len, (config.data.label_space for config in all_env_configs)))
        # build and return configuration
        return {
            # set framework to use
            "framework": "torch",
            # specify model
            "model": {
                "custom_model": self.model_type.type,
                "custom_model_config": self.feature_extractor.model_config
            },
            # set up environments
            "env": self.approach.env_type,
            "env_config": {
                "model_max_sequence_length": max_seq_length,
                "model_max_num_labels": max_num_labels,
                "feature_extractor": self.feature_extractor,
                "experiment_pool": self.env_configs
            },
            # add configs from all sub-models
            **self.algorithm.ray_config,
            **self.workers.ray_config,
            **self.evaluation.ray_config,
        }

def main():
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to valid RL experiment config")
    parser.add_argument("--seed", type=int, default=1337, help="Random Seed")
    parser.add_argument("--diable-env-checking", action="store_true", help="Disable environment checking before starting experiment")
    # parse arguments
    args = parser.parse_args()

    # parse configuration and apply overwrites
    config = RlExperimentConfig.parse_file(args.config)    
    print("Config:", config.json(indent=2))

    # build ray config and add callbacks
    ray_config = config.ray_config
    ray_config['callbacks'] = MultiCallbacks([
        CustomMetricsFromEnvCallbacks,
        LoggingCallbacks
    ])
    ray_config['log_level'] = "INFO"
    ray_config['disable_env_checking'] = args.disable_env_checking

    # connect to ray cluster
    ray.init(address='auto')

    # start training
    Tuner(
        config.algorithm.algo.type,
        param_space=ray_config,
        run_config=RunConfig(
            # stop config
            stop=dict(
                timesteps_total=config.algorithm.max_timesteps,
            ),
            # setup wandb callback
            callbacks=[
                WandbLoggerCallback(
                    project="rl-active-learning",
                    group=None,
                    log_config=False,
                    save_checkpoints=False,
                    config=config.dict()
                )
            ],
            verbose=1
        ),
    ).fit()
    # shutdown
    ray.shutdown()

if __name__ == '__main__':
    main()
