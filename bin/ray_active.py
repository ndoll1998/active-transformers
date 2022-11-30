import os
import ray
import json
from tqdm import trange
from itertools import product
from active.scripts.run_active import active

@ray.remote(num_gpus=1)
class RemoteActiveExperiment(object):

    def __init__(self, config:str, strategy:str, seed:int):
        # save config and seed
        self.config = config
        self.strategy = strategy
        self.seed = seed
        # extract experiment name from config
        with open(self.config, 'r') as f:
            self.name = json.loads(f.read())['name']

    def run(self, use_cache:bool):
        return active(self.config, self.seed, self.strategy, None, None, use_cache, True)

    def __repr__(self) -> str:        
        return "%s(%s)" % (self.name, self.strategy)

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks.")
    parser.add_argument("--configs", type=str, nargs='+', required=True, help="Path to a valid experiment configurations")
    parser.add_argument("--strategies", type=str, nargs='+', default=["random"], help="Active Learning Strategies to apply")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1337], help="Random Seeds")
    parser.add_argument("--use-cache", action='store_true', help="Load cached preprocessed datasets if available")
    parser.add_argument("--ray-address", type=str, default='auto', help="Address of ray cluster head node")
    parser.add_argument("-v", "--verbose", action="store_true", help="Forwards logs to head node")
    # parse arguments
    args = parser.parse_args()

    # connect to ray cluster
    ray.init(
        address=args.ray_address,
        runtime_env={
            'working_dir': os.getcwd(),
            'env_vars': {
                "WANDB_MODE": os.environ.get("WANDB_MODE", "disabled"),
                "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "dump")
            }
        },
        # setup logging
        logging_level="WARN",
        log_to_driver=args.verbose
    )
    
    experiments = [
        RemoteActiveExperiment.remote(config, strategy, seed)
        for config, strategy, seed in product(args.configs, args.strategies, args.seeds)
    ]
    # create/schedule all remote jobs
    futures = [experiment.run.remote(args.use_cache) for experiment in experiments]

    # wait for jobs to finish
    for _ in trange(len(futures)):
        # block until one job finished
        _, futures = ray.wait(futures, num_returns=1, fetch_local=False)

    # cleanup
    ray.shutdown()
