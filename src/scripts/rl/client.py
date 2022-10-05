import torch
import numpy as np
# import ray policy client
import ray
from ray.rllib.env.policy_client import PolicyClient
# import environment and active learning components
from src.active.rl.stream.env import StreamBasedEnv
from src.active.rl.utils import client_run_episode
from src.active.helpers.engines import Evaluator
from src.active.engine import ActiveLearningEvents
# import ignite
from ignite.metrics import Fbeta
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# transformers tokenizer
from transformers import AutoTokenizer
# import active learning setup helpers
from src.scripts.run_active import (
    add_data_args,
    add_model_and_training_args,
    load_and_preprocess_datasets,
    build_engine_and_loop
)
# import exception
from requests.exceptions import ConnectionError

#
# Argument Parsing
#

def add_client_args(parser, group_name="Policy Client Arguments"):

    group = parser.add_argument_group(group_name)
    # server and client setup
    group.add_argument("--server-address", type=str, default="http://0.0.0.0:9900", help="URI to server running the policy")
    # return argument group
    return group

def add_reinforcement_learning_args(parser, group_name="Reinforcement Learning Arguments"):

    group = parser.add_argument_group(group_name)
    # reinforcement learning params
    group.add_argument("--policy-pretrained-ckpt", type=str, default="distilbert-base-uncased", help="Pretrained checkpoint of the policy transformer model. Only used for tokenization.")
    group.add_argument("--query-strategy", type=str, default="random", help="Strategy used for preselection of query elements from dataset. Query elements are passed to the agent for final selection.")
    # return argument group
    return group

def add_active_learning_args(parser, group_name="Active Learning Arguments"):
    
    group = parser.add_argument_group(group_name)
    # server and client setup
    group.add_argument("--query-size", type=int, default=25, help="Number of data points to query from pool at each AL step")
    group.add_argument("--steps", type=int, default=-1, help="Number of Active Learning Steps. Defaults to -1 meaning the whole dataset will be processed.")
    # return argument group
    return group

#
# Environment
#

def build_stream_based_env(args):
    
    # only token classification tasks are allowed
    assert args.task == "token", "Only token classification tasks are supported"
    # set strategy to query strategy, this is useful since the strategy
    # generated for the loop in `build_engine_and_loop` is exactly the
    # query strategy but is used differently. Besides the function expects
    # the strategy field to be set anyways
    args.strategy = args.query_strategy

    assert args.pretrained_ckpt == args.policy_pretrained_ckpt, "Preprocessing not implemented yet!"
    # load and preprocess datasets
    # TODO: what if policy and model need different tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    ds = load_and_preprocess_datasets(args, tokenizer=tokenizer)
    # build active learning engine
    al_engine, loop = build_engine_and_loop(args, ds)

    # attach progress bar to trainer
    ProgressBar(ascii=True).attach(al_engine.trainer, output_transform=lambda out: {'loss': Evaluator.get_loss(out)})

    # attach log event handler
    @al_engine.on(ActiveLearningEvents.DATA_SAMPLING_COMPLETED)
    def log_handler(engine):
        # log active learning step
        print("AL Step:              %i" % engine.state.iteration)
        print("Train Data Size:      %i" % len(engine.train_dataset))
        print("Validation Data Size: %i" % len(engine.val_dataset))

    # create environment
    return StreamBasedEnv(
        # active learning setup
        budget=(args.steps * args.query_size) if args.steps > -1 else float('inf'),
        query_size=args.query_size,
        engine=al_engine,
        # reward metric and query strategy
        metric=Fbeta(beta=1.0, output_transform=Evaluator.get_logits_and_labels),
        query_strategy=loop.strategy,
        # data
        policy_pool_data=ds['train'],
        model_pool_data=ds['train'],
        model_test_data=ds['test']        
    )

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Start reinforcement learning client running a stream-based AL environment")
    add_client_args(parser)    
    add_data_args(parser)
    add_model_and_training_args(parser)
    add_reinforcement_learning_args(parser)
    add_active_learning_args(parser)

    # parse arguments
    args = parser.parse_args()
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build environment
    env = build_stream_based_env(args)

    env.engine.max_convergence_retries=1

    # create policy client
    client = PolicyClient(
        address=args.server_address,
        inference_mode='remote', # or local
        # only update policy on explicit request
        update_interval=None
    )

    i = 0
    # run for n episodes
    while True:
        i += 1

        try:
            # update local policy model
            if client.local:
                print("POLLING MODEL WEIGHTS")
                client.update_policy_weights()

            print("RUNNING EPISODE %i" % i)
            # run episode
            client_run_episode(
                env=env,
                client=client,
                training_enabled=True
            )

        except ConnectionError:
            print("LOST CONNECTION TO POLICY INPUT SERVER!")
            break
