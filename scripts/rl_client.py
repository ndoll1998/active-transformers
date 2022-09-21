import torch
import numpy as np
# hugginface
import datasets
import transformers
# import ray policy client
import ray
from ray.rllib.env.policy_client import PolicyClient
# import environment and run helper
from src.active.rl.stream.env import StreamBasedEnv
from src.active.rl.utils import client_run_episode
from src.active.helpers.engines import Evaluator
from src.active.engine import ActiveLearningEvents
# import ignite
from ignite.engine import Events
from ignite.metrics import Fbeta
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# import active learning setup helpers
from scripts.run_active import (
    load_and_preprocess_datasets,
    build_engine_and_loop
)


if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Start reinforcement learning client running a stream-based AL environment")
    # server and client setup
    parser.add_argument("--server-address", type=str, default="http://0.0.0.0:9900", help="URI to server running the policy")
    parser.add_argument("--training-disabled", action='store_true', help="Flag to run environment without training the policy")
    # reinforcement learning params
    parser.add_argument("--policy-pretrained-ckpt", type=str, default="distilbert-base-uncased", help="Pretrained checkpoint of the policy transformer model. Only used for tokenization.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to run the environemnt for.")
    parser.add_argument("--strategy", type=str, default="random", help="Strategy used for preselection of query elements from dataset. Query elements are passed to the agent for final selection.")
    # data setup
    parser.add_argument("--dataset", type=str, default="conll2003", help="Dataset to run active learning experiment on")
    parser.add_argument("--label-column", type=str, default="ner_tags", help="Dataset column containing target labels.")
    parser.add_argument('--min-length', type=int, default=0, help="Minimum sequence length an example must fulfill. Samples with less tokens will be filtered from the dataset.")
    parser.add_argument("--max-length", type=int, default=32, help="Maximum length of input sequences")
    # specify active learning parameters
    parser.add_argument("--query-size", type=int, default=25, help="Number of data points to query from pool at each AL step")
    parser.add_argument("--steps", type=int, default=-1, help="Number of Active Learning Steps. Defaults to -1 meaning the whole dataset will be processed.")
    # specify model and learning hyperparameters
    parser.add_argument("--pretrained-ckpt", type=str, default="distilbert-base-uncased", help="The pretrained model checkpoint")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate used by optimizer")
    parser.add_argument("--weight-decay", type=float, default=1.0, help="Weight decay rate")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs to train within a single AL step")
    parser.add_argument("--epoch-length", type=int, default=None, help="Number of update steps of an epoch")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size to use during training and evaluation")
    # specify convergence criteria
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping Patience")
    parser.add_argument("--acc-threshold", type=float, default=0.98, help="Early Stopping Accuracy Threshold")
    # others
    parser.add_argument("--use-cache", action='store_true', help="Use cached datasets if present")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed")
    
    # parse arguments
    args = parser.parse_args()
    # also only token classification tasks are allowed the entry
    # is needed to select the correct model type in `build_engine_and_loop`
    args.task = "token"

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert args.pretrained_ckpt == args.policy_pretrained_ckpt, "Preprocessing not implemented yet!"
    # load and preprocess datasets
    # TODO: what if policy and model need different tokenization
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.pretrained_ckpt)
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
    env = StreamBasedEnv(
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

    # create policy client
    client = PolicyClient(
        address=args.server_address,
        inference_mode='remote', # alternatively local to pull policy model from server
        update_interval=None
    )

    # run for n episodes
    for i in range(args.num_episodes):

        print("RUNNING EPISODE %i" % i)
        # run episode
        client_run_episode(
            env=env,
            client=client,
            training_enabled=not args.training_disabled
        )

        print("EPISODE COMPLETE")
        # update model if inference mode is local
        if client.local:
            print("POLLING MODEL WEIGHTS")
            client.update_policy_weights()

    
