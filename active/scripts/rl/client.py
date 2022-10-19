import torch
import numpy as np
# import ray policy client
import ray
from ray.rllib.env.policy_client import PolicyClient
# import environment and active learning components
from active.rl.stream.env import StreamBasedEnv
from active.helpers.evaluator import Evaluator
from active.core.metrics import AreaUnderLearningCurve
from active.helpers.engine import (
    ActiveLearningEngine, 
    ActiveLearningEvents
)
from active.core.loop import ActiveLoop
# import ignite
from ignite.engine import Events
from ignite.metrics import Fbeta
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# transformers tokenizer
from transformers import AutoTokenizer
# import active learning setup helpers
from active.scripts.run_train import attach_metrics
from active.scripts.run_active import (
    load_and_preprocess_datasets,
    create_trainer,
    create_engine_and_loop
)
# import exception
from requests.exceptions import ConnectionError
# argparse helpers
from typing import Literal
from defparse import Ignore

def create_client(
    server_address:str ="0.0.0.0:9900",
    mode:Literal["remote", "local"] ='remote',
    training_disabled:bool =False
):
    """ Create Policy Client

        Args:
            server_address (str): URI to the server running the policy
            mode (str): inference mode
            training_disabled (bool):
                whether to use observations from this client for
                policy updates        

        Returns:
            client (PolicyClient): client instance
    """
    return PolicyClient(
        address=server_address,
        inference_mode=mode,
        # only update policy on explicit request
        update_interval=None
    )

def create_env(
    ds:Ignore[dict],
    engine:Ignore[ActiveLearningEngine],
    loop:Ignore[ActiveLoop],
    # active learning args
    query_strategy:str ="random",
    query_size:int =25,
    steps:int =-1,
    # policy and model
    pretrained_ckpt:str ="distilbert-base-uncased",
    policy_pretrained_ckpt:str ="distilbert-base-uncased"
) -> StreamBasedEnv:
    """ Build Stream-based Active Learning Environment

        Args:
            ds (dict): dictionary containing datasets
            engine (ActiveLearningEngine): active learning engine instance
            loop (ActiveLearningLoop): 
                active learning loop, actually only the strategy
                within is of interest
            query_strategy (str):
                Strategy used for preselection of query elements from
                dataset. Query elements are passed to the agent for
                final selection.
            query_size (int): 
                Number of data points to query from pool at each AL step
            steps (int): 
                number of Active Learning Steps. Defaults to -1
                meaning the whole pool of data will be processed.
            pretrained_ckpt (str):
                pre-trained tranformer checkpoint to use.
            policy_pretrained_ckpt (str): 
                Pretrained Transformer model of the policy feature extractor.
    
        Returns:
            env (StreamBasedEnv): stream-based active learning environment
    """

    assert pretrained_ckpt == policy_pretrained_ckpt, "Preprocessing not implemented yet!"
    # attach progress bar to trainer
    ProgressBar(ascii=True).attach(engine.trainer, output_transform=lambda out: {'loss': out['loss']})

    # attach log event handler
    @engine.on(ActiveLearningEvents.DATA_SAMPLING_COMPLETED)
    def log_handler(e):
        # log active learning step
        print("AL Step:              %i" % engine.state.iteration)
        print("Train Data Size:      %i" % len(engine.train_dataset))
        print("Validation Data Size: %i" % len(engine.val_dataset))

    # create environment
    env = StreamBasedEnv(
        # active learning setup
        budget=(steps * query_size) if steps > -1 else float('inf'),
        query_size=query_size,
        engine=engine,
        # reward metric and query strategy
        metric=Fbeta(beta=1.0, output_transform=Evaluator.get_logits_and_labels),
        query_strategy=loop.strategy,
        # data
        policy_pool_data=ds['train'],
        model_pool_data=ds['train'],
        model_test_data=ds['test']        
    ) 

    @engine.on(Events.ITERATION_COMPLETED)
    def log_reward_metric(e):
        print("Reward Metric:        %.02f" % env.state.prev_metric)

    # reset env once to instantiate evaluator
    env.reset()
    # attach metrics to active learning engine
    attach_metrics(env.evaluator)
    AreaUnderLearningCurve(
        output_transform=lambda _: (
            # point of learning curve given
            # by iteration and reward metric
            env.engine.state.iteration,
            env.evaluator.state.metrics['F']
        )
    ).attach(env.engine, "Area(F)")

    @engine.on(Events.COMPLETED)
    def log_area_metric(e):
        print("Area Under Curve:     %.02f" % env.engine.state.metrics["Area(F)"])

    return env

def main():
    from defparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Start reinforcement learning client running a stream-based AL environment")
    # add arguments
    build_client = parser.add_args_from_callable(create_client, group="Client Arguments")
    build_datasets = parser.add_args_from_callable(load_and_preprocess_datasets, group="Dataset Arguments", ignore=["task"])
    build_trainer = parser.add_args_from_callable(create_trainer, group="Trainer Arguments", ignore=["task"])
    build_engine_and_loop = parser.add_args_from_callable(create_engine_and_loop, group="Active Learning Arguments", ignore=["task", "strategy"])
    build_env = parser.add_args_from_callable(create_env, group="Environment Arguments")
    # parse arguments
    args = parser.parse_args()

    # build environment
    env = build_env(
        build_datasets(task="token"),
        *build_engine_and_loop(
            trainer=build_trainer(task="token"),
            pool=[], # net needed here
            strategy=args.query_strategy
        )
    )

    try:
        # create policy client
        client = build_client()
    except ConnectionError:
        print("FAILED TO OPEN CONNECTION TO POLICY INPUT SERVER!")
        exit()

    i = 0
    # run for n episodes
    while True:
        i += 1

        try:
            # update local policy model
            # this also updates global variables including random states
            if client.local:
                print("POLLING MODEL WEIGHTS")
                client.update_policy_weights()

            print("RUNNING EPISODE %i" % i)
    
            # reset environment and start episode
            obs = env.reset()
            eid = client.start_episode(
                training_enabled=not args.training_disabled
            )

            done = False
            while not done:
                # get action from client
                action = client.get_action(eid, obs)
                # apply actions and observe returns
                obs, reward, done, info = env.step(action)
                # log returns
                client.log_returns(eid, reward, info)

            # end episode
            client.end_episode(eid, obs)

        except ConnectionError:
            print("LOST CONNECTION TO POLICY INPUT SERVER!")
            break

if __name__ == '__main__':
    main()
