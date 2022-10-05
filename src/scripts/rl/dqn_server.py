import torch
# import ray
import ray
from ray.tune import Tuner
from ray.air import RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.algorithms.dqn import DQN
# import environment and model components
from src.active.rl.stream.env import StreamBasedEnv
from src.active.rl.stream.model import DQNModel
from src.active.rl.extractors.transformer import TransformerFeatureExtractor
# import argument constructors
from src.scripts.rl.ppo_server import (
    add_server_args, 
    add_policy_args
)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Start a DQN server providing the policy for connected clients")
    add_server_args(parser)
    add_policy_args(parser)
    # parse arguments
    args = parser.parse_args()

    # create a dummy environment to get
    # observation and action space
    env = StreamBasedEnv(
        budget=0,
        query_size=0,
        engine=None,
        metric=None,
        query_strategy=None,
        policy_pool_data=[],
        model_pool_data=[],
        model_test_data=[],
        # specify variables that degine observation space
        policy_sequence_length=args.policy_sequence_length,
        model_sequence_length=args.model_sequence_length,
        max_num_labels=args.model_num_labels
    )

    # build configuration
    config = dict(
        # specify framework and gpu usage
        framework='torch',
        num_gpus=1, # multi-gpu not supported

        # doesn't need an actual environment but interacts with
        # environments through connected clients
        env=None,
        # still the policy needs the observation space
        observation_space=env.observation_space,
        action_space=env.action_space,

        # use the policy server input to generate experiences
        input=lambda ioctx: PolicyServerInput(
            ioctx,
            address=args.address,
            port=args.port + ioctx.worker_index - 1,
            idle_timeout=3.0 # default is 3.0
        ),
        # number of workers, i.e. maximum number of clients
        # that connect to the server
        num_workers=args.num_workers,

        # disable off-policy estimation (OPE) as rollouts
        # are coming from clients which doens't allow off-policy
        off_policy_estimation_methods={},
        
        # algorithm parameters
        rollout_fragment_length=args.rollout_size,
        train_batch_size=args.policy_batch_size,

        # specify the policy model
        model=dict(
            # specify custom model
            custom_model=DQNModel,
            custom_model_config=dict(
                feature_extractor_type=TransformerFeatureExtractor,
                feature_extractor_config=dict(
                    pretrained_ckpt=args.policy_pretrained_ckpt
                ),
            )
        ),
        # learning rate
        lr=2e-5,
        # log level
        log_level="INFO",
    )

    # run tuner, i.e. train policy
    Tuner(
        DQN,
        param_space=config,
        run_config=RunConfig(
            # stop config
            stop=dict(
                # training_iteration=,
                timesteps_total=args.timesteps,
                # episode_reward_mean=
            ),
            # setup wandb callback
            callbacks=[
                WandbLoggerCallback(
                    project="rl-active-learning",
                    job_type="tune",
                    group=None,
                    log_config=False,
                    save_checkpoints=False,
                )
            ],
            verbose=2
        ),
    ).fit()
