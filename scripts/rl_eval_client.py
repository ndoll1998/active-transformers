import os
import json
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
    group.add_argument("--resume", action='store_true', help="Whether to resume the previous checkpoint stored in `ckpt_dir`. Alternatively the previous checkpoint will be overwritten.")

    # return argument group
    return group

class Checkpoint(object):
    """ Helper class to handle checkpointing over multiple calls to the script """

    def __init__(self, ckpt_dir:str):
        # checkpoint metric and history
        self.prev_metric = float('-inf')
        self.history = {}
        # save checkpoint directory
        self.ckpt_dir = ckpt_dir

    def handle(self, metrics, weights):
        
        # add metrics to history
        self.history[len(self.history)] = metrics 
        # compute average metric
        cur_metric = sum(metrics) / len(metrics)

        if cur_metric > self.prev_metric:
            # save weights
            torch.save(weights, os.path.join(self.ckpt_dir, "model.bin"))
            self.prev_metric = cur_metric

    def load(self):
        with open(os.path.join(self.ckpt_dir, "ckpt.json"), 'r') as f:
            data = json.loads(f.read())
        self.prev_metric = data['prev_metric']
        self.history = data['history']

    def save(self):
        with open(os.path.join(self.ckpt_dir, "ckpt.json"), 'w+') as f:
            f.write(json.dumps({
                'prev_metric': self.prev_metric,
                'history': self.history
            }, indent=4))

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
            
            wandb.log({"test/Area(F)": sum(metrics) / len(metrics)})

            # load previous checkpoint if exists
            ckpt = Checkpoint(args.ckpt_dir)
            if args.resume and os.path.isfile(os.path.join(args.ckpt_dir, "ckpt.json")):
                ckpt.load()
            else:
                # create checkpoint directory if not exists
                os.makedirs(args.ckpt_dir, exist_ok=True)
            
            # handle
            ckpt.handle(metrics, client.rollout_worker.get_weights())
            # save checkpoint
            ckpt.save()

        except ConnectionError:
            print("LOST CONNECTION TO POLICY INPUT SERVER!")
            break

    # finish weights and biases runs
    for run in runs:
        run.finish()
