import torch
# import ray
import ray
from ray.tune import Tuner
from ray.air import RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.algorithms.ppo import PPO
# callbacks
from ray.rllib.algorithms.callbacks import MultiCallbacks
from active.rl.callbacks import (
    CustomMetricsFromEnvCallbacks,
    LoggingCallbacks
)
# import environment and model components
from active.rl.stream.env import StreamBasedEnv
from active.rl.stream.model import ActorCriticModel, RecurrentActorCriticModel
from active.rl.extractors.transformer import TransformerFeatureExtractor

def create_input_factory(
    address:str ="0.0.0.0",
    port:int =9900
):
    """ Policy Server Input Factory

        Args:
            address (str): Host address of the policy server
            port (int): 
                Base port for clients to connect to. Available
                ports are (port + worker_id - 1)
    """
    return lambda ioctx: PolicyServerInput(
        ioctx,
        address=address,
        port=port + ioctx.worker_index - 1,
        idle_timeout=3.0 # default is 3.0
    )

def create_eval_input_factory(
    address:str ="0.0.0.0",
    eval_port:int =8800
):
    """ Policy Server Input Factory for evaluation

        Args:
            address (str): Host address of the policy server
            eval_port (int):
                Base port for evaluation clients to connect to.
                Available ports are (port + eval_worker_id - 1)
    """
    return create_input_factory(
        address=address,
        port=eval_port
    )

def create_base_config(
    num_workers:int =1,
    rollout_size:int =1024,
    policy_batch_size:int =32,
    policy_sequence_length:int =32,
    model_sequence_length:int =32,
    model_num_labels:int =9,
    timesteps:int =2048
) -> dict:
    """ Build the base ray configuration

        Args:
            num_workers (int): 
                number of workers, i.e. maximum number of clients
                that can connect simultaneously.
            rollout_size (int): 
                Size of the rollout buffer per worker.
            policy_batch_size (int): 
                Batch size used for training the policy model.
            policy_sequence_length (int): 
                Length of the input sequences to the policy transformer model.
                Needed to construct the observation space.
            model_sequence_length (int):
                Length of the input sequences to the actual model.
                Needed to construct the observation space.
            model_num_labels (int):
                Number of predicted labels (per token). Needed to construct
                observation space.
            timesteps (int):
                total timesteps to train the policy for

        Returns:
            config (dict): base ray configuration
    """
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
        policy_sequence_length=policy_sequence_length,
        model_sequence_length=model_sequence_length,
        max_num_labels=model_num_labels
    )

    # build configuration
    return dict(
        # specify framework and gpu usage
        framework='torch',
        num_gpus=1, # multi-gpu not supported

        # doesn't need an actual environment but interacts with
        # environments through connected clients
        env=None,
        # input needs to be set to input factory later
        input=None,
        # still the policy needs the observation space
        observation_space=env.observation_space,
        action_space=env.action_space,

        # number of workers, i.e. maximum number of clients
        # that connect to the server
        num_workers=num_workers,

        # disable off-policy estimation (OPE) as rollouts
        # are coming from clients which doens't allow off-policy
        off_policy_estimation_methods={},
        
        # algorithm parameters
        rollout_fragment_length=rollout_size,
        train_batch_size=policy_batch_size,
        
        # log level
        log_level="WARN"
    )

def create_evaluation_config(
    evaluation_interval:int =1,
    evaluation_duration:int =5,
    num_eval_workers:int =1,
    eval_seed:int =1337,
):
    """ Build the evaluation configuration

        Args:
            evaluation_interval (int):
                interval in which to evaluate the policy.
            evaluation_duration (int):
                number of episodes to evaluate the policy for.
            num_eval_workers (int):
                number of evaluation workers, i.e. maximum number of
                evaluation clients that can connect simultaneously.
            eval_seed (int):
                seed to use for evaluation        

        Returns:
            config (dict): evaluation configuration
    """
    return dict(
        evaluation_interval=evaluation_interval,
        evaluation_duration=evaluation_duration,
        evaluation_parallel_to_training=True,
        evaluation_config=dict(
            input=None, # set later
            explore=True, # typical for ppo
            seed=eval_seed
        ),
        evaluation_num_workers=num_eval_workers,
        # keep_per_episode_custom_metrics=True
    )

def create_ppo_config(
    policy_pretrained_ckpt:str ="distilbert-base-uncased",
    policy_batch_size:int =32
):
    """ Build the ppo policy configuration

        Args:
            policy_pretrained_ckpt (str): 
                Pretrained Transformer model of the policy feature extractor.
            policy_batch_size (int): 
                Batch size used for training the policy model.

        Returns:
            config (dict): base ray configuration
    """
    return dict(
        # ppo specific hyperparameters
        num_sgd_iter=10,
        sgd_minibatch_size=policy_batch_size,
        entropy_coeff=0.001,
        vf_loss_coeff=1e-5,
        # specify the policy model
        model=dict(
            # specify custom model
            # custom_model=ActorCriticModel,
            custom_model=RecurrentActorCriticModel,
            custom_model_config=dict(
                feature_extractor_type=TransformerFeatureExtractor,
                feature_extractor_config=dict(
                    pretrained_ckpt=policy_pretrained_ckpt
                ),
            )
        ),
        # learning rate
        lr=2e-5
    )

def main():
    from defparse import ArgumentParser
    parser = ArgumentParser(description="Start a PPO server providing the policy for connected clients")
    # add arguments
    build_input_factory = parser.add_args_from_callable(create_input_factory, group="Server Arguments")
    build_eval_input_factory = parser.add_args_from_callable(create_eval_input_factory, group="Server Arguments")
    build_ppo_config = parser.add_args_from_callable(create_ppo_config, group="Policy Arguments")
    build_base_config = parser.add_args_from_callable(create_base_config, group="Base Arguments")
    build_eval_config = parser.add_args_from_callable(create_evaluation_config, group="Evaluation Arguments")
    # parse arguments
    args = parser.parse_args()

    # build full configuration
    config = dict(
        **build_base_config(),
        **build_eval_config(),
        **build_ppo_config()
    )
    config['input'] = build_input_factory()
    config['evaluation_config']['input'] = build_eval_input_factory()
    config['callbacks'] = MultiCallbacks([
        CustomMetricsFromEnvCallbacks,
        LoggingCallbacks
    ])

    # run tuner, i.e. train policy
    Tuner(
        PPO,
        param_space=config,
        run_config=RunConfig(
            # stop config
            stop=dict(
                timesteps_total=args.timesteps,
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

if __name__ == '__main__':
    main()
