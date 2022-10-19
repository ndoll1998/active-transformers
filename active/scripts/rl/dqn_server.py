import torch
# import ray
import ray
from ray.tune import Tuner
from ray.air import RunConfig
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.algorithms.dqn import DQN
# callbacks
from ray.rllib.algorithms.callbacks import MultiCallbacks
from active.rl.callbacks import (
    CustomMetricsFromEnvCallbacks,
    LoggingCallbacks
)
# import environment and model components
from active.rl.stream.env import StreamBasedEnv
from active.rl.stream.model import DQNModel
from active.rl.extractors.transformer import TransformerFeatureExtractor
# import basic policy server functions
from active.scripts.rl.ppo_server import (
    create_input_factory, 
    create_eval_input_factory,
    create_base_config
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
            explore=False,
            seed=eval_seed
        ),
        evaluation_num_workers=num_eval_workers,
        # keep_per_episode_custom_metrics=True
    )

def create_dqn_config(
    policy_pretrained_ckpt:str ="distilbert-base-uncased",
    policy_batch_size:int =32
):
    """ Build the dqn policy configuration

        Args:
            policy_pretrained_ckpt (str): 
                Pretrained Transformer model of the policy feature extractor.
            policy_batch_size (int): 
                Batch size used for training the policy model.

        Returns:
            config (dict): base ray configuration
    """
    return dict(
        # dqn specific config
        num_atoms=1,
        # dueling dqn
        dueling=False,
        v_min=-1.0,
        v_max=1.0,
        # noisy network for exploration
        noisy=True,
        sigma0=0.5,
        # others
        double_q=True,
        n_step=1,
        # replay buffer config
        replay_buffer_config=dict(
            capacity=64000,
            # slightly prioritize samples with higher td-error
            prioritized_replay_alpha=0.6,
            prioritized_replay_beta=0.4,
            prioritized_replay_eps=1e-6
        ),
        # specify the policy model
        model=dict(
            # specify custom dqn model
            custom_model=DQNModel,
            custom_model_config=dict(
                feature_extractor_type=TransformerFeatureExtractor,
                feature_extractor_config=dict(
                    pretrained_ckpt=policy_pretrained_ckpt
                )
            )
        ),
        # learning rate
        lr=2e-5
    )


def main():
    from defparse import ArgumentParser
    parser = ArgumentParser(description="Start a DQN server providing the policy for connected clients")
    build_input_factory = parser.add_args_from_callable(create_input_factory, group="Server Arguments")
    build_eval_input_factory = parser.add_args_from_callable(create_eval_input_factory, group="Server Arguments")
    build_dqn_config = parser.add_args_from_callable(create_dqn_config, group="Policy Arguments")
    build_base_config = parser.add_args_from_callable(create_base_config, group="Base Arguments")
    build_eval_config = parser.add_args_from_callable(create_evaluation_config, group="Evaluation Arguments")
    # parse arguments
    args = parser.parse_args()

    # build full configuration
    config = dict(
        **build_base_config(),
        **build_eval_config(),
        **build_dqn_config()
    )
    config['input'] = build_input_factory()
    config['evaluation_config']['input'] = build_eval_input_factory()
    config['callbacks'] = MultiCallbacks([
        CustomMetricsFromEnvCallbacks,
        LoggingCallbacks
    ])

    # run tuner, i.e. train policy
    Tuner(
        DQN,
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
