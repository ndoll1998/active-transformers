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

    def run(self, budget:int, query_size:int, use_cache:bool):
        return active(self.config, self.seed, self.strategy, budget, query_size, use_cache, True)

    def __repr__(self) -> str:        
        return "%s(%s)" % (self.name, self.strategy)

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks.")
    parser.add_argument("--configs", type=str, nargs='+', required=True, help="Path to a valid experiment configurations")
    parser.add_argument("--strategies", type=str, nargs='+', default=["random"], help="Active Learning Strategies to apply")
    parser.add_argument("--query-size", type=int, default=None, help="Query size to use for all experiments. Defaults to value specified in each config.")
    parser.add_argument("--budget", type=int, default=None, help="Annotation budget to use for all experiments. Defaults to value specified in each config.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1337], help="Random Seeds")
    parser.add_argument("--use-cache", action='store_true', help="Load cached preprocessed datasets if available")
    parser.add_argument("--ray-address", type=str, default='auto', help="Address of ray cluster head node")
    parser.add_argument("-v", "--verbose", action="store_true", help="Forwards logs to head node")
    # parse arguments
    args = parser.parse_args()
    
    # check if all configs exist
    for config in args.configs:
        assert os.path.isfile(config), "Config file %s not found!" % config

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
        logging_level="INFO",
        log_to_driver=args.verbose
    )
    
    experiments = [
        RemoteActiveExperiment.remote(config, strategy, seed)
        for config, strategy, seed in product(args.configs, args.strategies, args.seeds)
    ]
    # create/schedule all remote jobs
    futures = [experiment.run.remote(args.budget, args.query_size, args.use_cache) for experiment in experiments]
    task_ids = [f.task_id() for f in futures]

    # wait for jobs to finish
    for _ in trange(len(futures)):
        # block until one job finished
        objs, futures = ray.wait(futures, num_returns=1, fetch_local=False)
        # delete actor corresponding to finished job
        # to free cluster resource and allow execution
        # of next scheduled task
        i = task_ids.index(objs[0].task_id())
        del experiments[i], task_ids[i]

    # cleanup
    ray.shutdown()
