# import ray
import ray
from ray.tune import Tuner
from ray.air import RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.algorithms.ppo import PPO
# import environment and model components
from src.active.rl.stream.env import StreamBasedEnv
from src.active.rl.stream.model import StreamBasedModel
from src.active.rl.extractors.transformer import TransformerFeatureExtractor
# import helper functions
from scripts.run_active import (
    add_data_args,
    add_model_and_training_args
)
from scripts.rl_client import (
    add_active_learning_args,
    build_stream_based_env
)

#
# Argument Parsing
#

def add_server_args(parser, group_name="Policy Server Arguments"):

    group = parser.add_argument_group(group_name)
    # server setup
    group.add_argument("--address", type=str, default="0.0.0.0", help="Host address of the policy server")
    group.add_argument("--port", type=int, default=9900, help="Base port for clients to connect to. Available ports are (port + worker_id - 1)")
    group.add_argument("--num-workers", type=int, default=1, help="Number of workers, that is maximum number of clients that can connect simultaneously.")
    # return argument group
    return group

def add_policy_args(parser, group_name="Policy Arguments"):
    
    group = parser.add_argument_group(group_name)
    # policy training arguments
    group.add_argument("--policy-batch-size", type=int, default=32, help="Batch size used to train policy model.")
    group.add_argument("--rollout-size", type=int, default=128, help="Size of the rollout buffer, i.e. number of steps used to train the policy model with.")
    group.add_argument("--timesteps", type=int, default=2048, help="Total number of timesteps to train the policy model for")
    # policy and model setup
    group.add_argument("--policy-pretrained-ckpt", type=str, default="distilbert-base-uncased", help="Pretrained Transformer model of the policy feature extractor")
    group.add_argument("--policy-sequence-length", type=int, default=32, help="Length of inputs for the policy transformer model. Needed to construct observation space.")
    group.add_argument("--model-sequence-length", type=int, default=32, help="Length of inputs for the prediction transformer model. Needed to construct observation space.")
    group.add_argument("--model-num-labels", type=int, default=9, help="Number of predicted labels (per token). Needed to construct observation space.")
    # return argument group
    return group


if __name__ == '__main__':

    # register stream based environment
    # this is used for evaluation
    from ray.tune.registry import register_env
    from types import SimpleNamespace
    register_env(
        "stream_based_env", 
        lambda config: build_stream_based_env(SimpleNamespace(**config))
    )

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Start a server providing the policy for connected clients")
    add_server_args(parser)
    add_policy_args(parser)
    add_data_args(parser, group_name="Evaluation Data Arguments")
    add_model_and_training_args(parser, group_name="Evaluation Model and Training Arguments")
    add_active_learning_args(parser, group_name="Evaluation Active Learning Arguments")
    # parse arguments
    args = parser.parse_args()

    # TODO: what should the query strategy for evaluation be
    # TODO: also probably shouldn't hard-code this
    args.query_strategy = "random"

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
        sgd_minibatch_size=args.policy_batch_size,
        # specify the policy model
        model=dict(
            custom_model=StreamBasedModel,
            custom_model_config=dict(
                feature_extractor_type=TransformerFeatureExtractor,
                feature_extractor_config=dict(
                    pretrained_ckpt=args.policy_pretrained_ckpt
                )
            )
        ),
        # log level
        log_level="INFO",
    
        # set up evaluation
        evaluation_interval=1,
        # evaluate for some episodes
        evaluation_duration=3,
        evaluation_duration_unit="episodes",
        evaluation_num_workers=3,
        # configuration for evaluation
        evaluation_config=dict(
            # disable exploration for evaluation
            # TODO: maybe this is a bad idea as policy gradient
            #       methods find optimal stochastic policies
            explore=False,
            # specify environment to use for evaluation
            # note that 'stream_based_env' was registered above
            env="stream_based_env",
            env_config=vars(args),
            # overwrite policy input server as for evaluation the
            # environment should be used
            input="sampler",
            # use one gpu per worker
            num_gpus_per_worker=1,
            # use the same base seed for evaluation
            # note that each worker still has a unique seed
            # but the seed is the same for all evaluation run
            seed=1337,
            # disable environment checking
            disable_env_checking=True
        )
    )

    # run tuner, i.e. train policy
    Tuner(
        PPO,
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
                    project="rl-active-learining",
                    group=None,
                    log_config=False,
                    save_checkpoints=False,
                )
            ],
            verbose=2
        ),
    ).fit()
