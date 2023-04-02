import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import active learning components
from active import (
    ActiveLoop,
    ActiveEngine,
    ActiveEvents,
    strategies
)
# utils
from active.utils.model import get_encoder_from_model
from active.utils.data import NamedTensorDataset
from active.engine.strategy import ActiveStrategy
# import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# dimensionality reduction
from sklearn.manifold import TSNE
# others
import transformers
from math import ceil
from matplotlib import pyplot as plt
from pydantic import BaseModel, validator
# base config and helper function
from active.scripts.run_train_v2 import Task, ExperimentConfig

def visualize_strategy_output(strategy):
    # get selected indices and processing output of unlabeled pool
    idx = strategy.selected_indices
    output = strategy.output
    output = output if output.ndim == 2 else \
        output.reshape(-1, 1) if output.ndim == 1 else \
        output.flatten(start_dim=1)
    # reduce dimension for visualization using t-sne    
    X = TSNE(
        n_components=2,
        perplexity=50,
        learning_rate='auto',
        init='random'
    ).fit_transform(
        X=output
    )
    # plot
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=1.0, color="blue", alpha=0.1)
    ax.scatter(X[idx, 0], X[idx, 1], s=2.0, color='red', alpha=1.0)
    ax.legend(['pool', 'query'])
    # return axes and figure
    return ax, fig

class ActiveLearningConfig(BaseModel):
    """ Active Learning Configuration Model """
    task:Task =None
    # strategy to use
    strategy:str
    # budget and query size
    budget:int
    query_size:int
    # validation split
    train_val_split:float =0.9

    @property
    def num_steps(self):
        return ceil(self.budget / self.query_size)

    @property
    def strategy_kwargs(self) -> dict:
        # parse keyword arguments from strategy identifier string
        if ('[' in self.strategy) and (']' in self.strategy):
            kwargs = self.strategy.split('[', 1)[1].rsplit(']', 1)[0]
            return eval(f"dict({kwargs})")
        else:
            # no argument supplied
            return dict()

    def build_strategy(self, model:transformers.PreTrainedModel) -> ActiveStrategy:
        if self.strategy.startswith('random'):
            return strategies.Random(**self.strategy_kwargs)
        elif self.strategy.startswith('least-confidence'):
            return strategies.LeastConfidence(model, **self.strategy_kwargs)
        elif self.strategy.startswith('prediction-entropy'):
            return strategies.PredictionEntropy(model, **self.strategy_kwargs)
        elif self.strategy.startswith('badge') and self.task == Task.SEQUENCE:
            return strategies.BadgeForSequenceClassification(
                get_encoder_from_model(model),
                model.classifier,
                **self.strategy_kwargs
            )
        elif self.strategy.startswith('badge') and self.task in [Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING]:
            return strategies.BadgeForTokenClassification(
                get_encoder_from_model(model),
                model.classifier,
                **self.strategy_kwargs
            )
        elif self.strategy.startswith('alps'):
            return strategies.AlpsConstantEmbeddings(model, **self.strategy_kwargs)
        elif self.strategy.startswith('egl-topk'):
            return strategies.EglByTopK(model, **self.strategy_kwargs)
        elif self.strategy.startswith('egl-sampling'):
            return strategies.EglBySampling(model, **self.strategy_kwargs)
        elif self.strategy.startswith('layer-egl-topk'):
            return strategies.LayerEglByTopK(model, **self.strategy_kwargs)
        elif self.strategy.startswith('layer-egl-sampling'):
            return strategies.LayerEglBySampling(model, **self.strategy_kwargs)
        elif self.strategy.startswith('entropy-over-max'):
            return strategies.EntropyOverMax(model, **self.strategy_kwargs)
        elif self.strategy.startswith('binary-entropy-over-max'):
            return strategies.BinaryEntropyOverMax(model, **self.strategy_kwargs)

        raise ValueError("Unrecognized strategy: %s" % self.strategy)

class AlExperimentConfig(ExperimentConfig):
    """ Active Learning Experiment Configuration """

    # add active learning config to experiment config
    active:ActiveLearningConfig

    @validator('active', pre=True)
    def _pass_task_to_active_config(cls, v, values):
        assert 'task' in values
        if isinstance(v, BaseModel):
            return v.copy(update={'task': values.get('task')})
        elif isinstance(v, dict):
            return v | {'task': values.get('task')}

def active(config:str, seed:int, strategy:str, budget:int, query_size:int, use_cache:bool, disable_tqdm:bool =False, save_samples:bool =False, visualize_embeds:bool =False):
    import wandb

    # parse configuration and apply overwrites
    config = AlExperimentConfig.parse_file(config)
    config.trainer.seed = seed
    config.trainer.data_seed = seed
    config.trainer.disable_tqdm |= disable_tqdm
    config.active.strategy = strategy or config.active.strategy
    config.active.budget = budget or config.active.budget
    config.active.query_size = query_size or config.active.query_size
    print("Config:", config.json(indent=2))

    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # pre-compute some helpers used later on
    if config.task in (Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING):
        begin_tag_ids = torch.LongTensor(
            [1] if config.task is Task.NESTED_BIO_TAGGING else \
            [
                i for i, tag in enumerate(config.data.label_space)
                if tag.startswith(config.data.begin_tag_prefix)
            ]
        )

    # load datasets
    ds = config.load_dataset(use_cache=use_cache)
    train_data = NamedTensorDataset.from_dataset(ds['train'])
    test_data = NamedTensorDataset.from_dataset(ds['test'])

    # keep tokenizer around as it is used in event handler
    tokenizer = config.model.tokenizer
    # create trainer and strategy
    trainer = config.build_trainer()
    strategy = config.active.build_strategy(trainer.model)
    # attach progress bar to strategy
    if not disable_tqdm:
        ProgressBar(ascii=True, desc='Strategy').attach(strategy)

    # create active learning loop
    loop = ActiveLoop(
        pool=train_data,
        strategy=strategy,
        batch_size=config.trainer.per_device_eval_batch_size * config.trainer._n_gpu,
        query_size=config.active.query_size,
        init_strategy=strategy if isinstance(strategy, strategies.Alps) else strategies.Random()
    )
    # create active learning engine
    al_engine = ActiveEngine(
        trainer=trainer,
        train_val_split=config.active.train_val_split
    )

    @al_engine.on(Events.ITERATION_STARTED)
    def on_started(engine):
        # log active learning step
        i = engine.state.iteration
        print("-" * 8, "AL Step %i" % i, "-" * 8)

    @al_engine.on(ActiveEvents.DATA_SAMPLING_COMPLETED)
    def save_selected_samples(engine):
        # check whether to save samples
        if not save_samples:
            return
        # create path to save samples in
        path = os.path.join(config.trainer.output_dir, "samples")
        os.makedirs(path, exist_ok=True)
        # log
        print("Saving selected samples to `%s`" % path)
        # get selected samples and re-create input texts
        data = engine.state.batch
        texts = tokenizer.batch_decode(
            sequences=[sample['input_ids'] for sample in data],
            skip_special_tokens=True
        )
        # save selected samples to file
        with open(os.path.join(path, "step-%i.txt" % engine.num_total_samples), 'w+') as f:
            f.write('\n'.join(texts))

    @al_engine.on(Events.ITERATION_COMPLETED)
    def test_and_log(engine):

        # evaluate on validation and test set
        val_metrics = trainer.evaluate(engine.val_dataset, metric_key_prefix="val")
        test_metrics = trainer.evaluate(test_data, metric_key_prefix="test")
        # print metrics
        print("Step:", engine.state.iteration, "-" * 15)
        print("Val   Entity F-Score:", val_metrics['val_weighted-avg_F'])
        print("Test  Entity F-Score:", test_metrics['test_weighted-avg_F'])

        # get total time spend in strategy
        strategy_time = loop.strategy.state.times[Events.COMPLETED.name]
        strategy_time = {'times/strategy': strategy_time} if strategy_time is not None else {}
        # log number of train and validation samples
        num_samples = {
            '#docs/train': engine.num_train_samples,
            '#docs/val': engine.num_val_samples
        }
        # log total number of tokens in train and test dataset
        num_tokens = {
            '#tokens/train': sum(item['attention_mask'].sum() for item in engine.train_dataset),
            '#tokens/val': sum(item['attention_mask'].sum() for item in engine.val_dataset)
        }
        # log total number of entity annotations in current datasets
        # number of entities matches number of begin tags in datasets
        num_annotations = {
            '#annotations/train': sum(
                torch.isin(item['labels'], begin_tag_ids).sum()
                for item in engine.train_dataset
            ),
            '#annotations/val': sum(
                torch.isin(item['labels'], begin_tag_ids).sum()
                for item in engine.val_dataset
            )
        } if config.task in (Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING) else {}
        # log all remaining metrics to weights and biases
        wandb.log(
            data=(engine.state.metrics | test_metrics | strategy_time | num_samples | num_tokens | num_annotations),
            step=engine.num_total_samples
        )

        # check if there is an output to visualize
        # score-based strategies only have score outputs, no need to visualize them
        if visualize_embeds and (loop.strategy.output is not None) and (not isinstance(loop.strategy, strategies.ScoreBasedStrategy)):
            print("Visualizing strategy output")
            ax, fig = visualize_strategy_output(loop.strategy)
            ax.set(title="%s embedding iteration %i (t-SNE)" % (config.active.strategy, engine.state.iteration))
            wandb.log({"Embedding": wandb.Image(fig)}, step=engine.num_total_samples)
            # close figure
            plt.close(fig)

    # initialize wandb
    wandb_config = config.dict()
    wandb_config['data']['dataset'] = config.data.dataset_info.builder_name
    wandb_config['seed'] = seed
    wandb.init(
        config=wandb_config,
        project=os.environ.get("WANDB_PROJECT", "active-final"),
        group=config.name,
        job_type=config.active.strategy
    )

    # run active learning experiment
    state = al_engine.run(loop, steps=config.active.num_steps)
    print("Active Learning Metrics:", state.metrics)
    # log active learning metric scores
    wandb.run.summary.update(state.metrics)

    # run finished
    wandb.finish(quiet=True)

def main():

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or bio-taging classification tasks using active learning.")
    # add arguments
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks.")
    parser.add_argument("--config", type=str, required=True, help="Path to a valid experiment configuration")
    parser.add_argument("--use-cache", action='store_true', help="Load cached preprocessed datasets if available")
    parser.add_argument("--seed", type=int, default=1337, help="Random Seed")
    parser.add_argument("--strategy", type=str, default=None, help="Overwrite strategy specified in config")
    parser.add_argument("--budget", type=int, default=None, help="Overwrite budget specified in config")
    parser.add_argument("--query-size", type=int, default=None, help="Overwrite query size specified in config")
    parser.add_argument("--save-samples", action='store_true', help="Save samples selected by the strategy to disk")
    parser.add_argument("--visualize-embeds", action='store_true', help="Visualize strategy embedding for non-score-based strategies")
    # parse arguments
    args = parser.parse_args()

    # run active learning experiment
    active(**vars(args))

if __name__ == '__main__':
    main()
