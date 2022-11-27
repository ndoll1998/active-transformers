import ray
from functools import partial
from active.scripts.run_train import train

@ray.remote(num_gpus=1)
def remote_train(config:str, seed:int, use_cache:bool):
    return train(config, seed, use_cache, True)

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks.")
    parser.add_argument("--config", type=str, required=True, help="Path to a valid experiment configuration")
    parser.add_argument("--use-cache", action='store_true', help="Load cached preprocessed datasets if available")
    parser.add_argument("--seed", type=int, default=1337, help="Random Seed")
    parser.add_argument("--ray-address", type=str, default='auto', help="Address of ray cluster head node")
    # parse arguments
    args = parser.parse_args()

    ray.init(address=args.ray_address)
    # schedule training run and block until completion
    state = remote_train.remote(args.config, args.seed, args.use_cache)
    state = ray.get(state)   

    ray.shutdown()
