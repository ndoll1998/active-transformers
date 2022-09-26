import os
import torch
import numpy as np
# import ray policy client
import ray
from ray.rllib.env.policy_client import PolicyClient
# import environment and run helper
from src.active.rl.utils import client_run_episode
from src.active.metrics import AreaUnderLearningCurve
# import active learning setup helpers
from scripts.run_active import (
    add_data_args,
    add_model_and_training_args,
    load_and_preprocess_datasets,
    build_engine_and_loop
)
from scripts.rl_client import (
    add_client_args,
    add_reinforcement_learning_args,
    add_active_learning_args,
    build_stream_based_env
)
# logging
import wandb
# exception
from requests.exceptions import ConnectionError


def add_evaluation_args(parser, group_name="Evaluation Arguments"):
    
    group = parser.add_argument_group(group_name)
    group.add_argument("--ckpt-dir", type=str, default="output/rl/", help="Directory to store policy checkpoint in if metric improved")
    group.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes to run. Each episode will have a unique seed. The final metric is averaged out.")

    # return argument group
    return group


if __name__ == '__main__':
    
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Start reinforcement learning client for evaluation of current policy only")
    add_client_args(parser)    
    add_data_args(parser)
    add_model_and_training_args(parser)
    add_reinforcement_learning_args(parser)
    add_active_learning_args(parser)
    add_evaluation_args(parser)
    
    # parse arguments
    args = parser.parse_args()

    # create checkpoint directory
    os.make_dirs(args.ckpt_dir, exists_ok=True)
    
    # build environment
    env = build_stream_based_env(args)
    # add evaluation metrics to env
    AreaUnderLearningCurve(
        output_transform=lambda _: (
            # point of learning curve given
            # by iteration and reward metric
            env.engine.state.iteration,
            env.evaluator.state.metrics['__reward_metric']
        )
    ).attach(env.engine, "test/Area(F)")
    
    # create policy client 
    client = PolicyClient(
        address=args.server_address,
        # use local policy network to use avoid updates mid-evaluation
        inference_mode='local',
        # only update policy on explicit request
        update_interval=None
    )

    # create a wandb run for each evaluation episode
    run = wandb.init(
        project="rl-active-learning",
        group="PPO",
        job_type="eval"
    )

    # keep track of current best average metric
    cur_best_metric = float('-inf')

    # run evaluation forever (i.e. as long as no error is thrown)
    while True:
    
        # set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
        try:
            
            metrics = []
            for i in range(args.episodes):
            
                # update local policy model
                assert client.local
                print("POLLING MODEL WEIGHTS")
                client.update_policy_weights()
            
                print("RUNNING EVALUATION EPISODE %i" % i)
                # run episode
                client_run_episode(
                    env=env,
                    client=client,
                    training_enabled=False
                )
                
                metrics.append(env.engine.state.metrics['test/Area(F)'])
            
            # compute and log current average metric
            avg_metric = sum(metrics) / len(metrics)
            wandb.log({"test/Area(F)": avg_metric})

            # check if metric improved over previous best
            if avg_metric > cur_best_metric:
                # update metric
                cur_best_metric = avg_metric
                # save model parameters
                torch.save(
                    client.rollout_worker.get_weights(),
                    os.path.join(args.ckpt_dir, "model.bin"),
                )

        except ConnectionError:
            print("LOST CONNECTION TO POLICY INPUT SERVER!")
            break

    # finish weights and biases runs
    for run in runs:
        run.finish()
