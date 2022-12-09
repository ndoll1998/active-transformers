import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import active learning components
from active.core import strategies
from active.core.loop import ActiveLoop
from active.core.metrics import AreaUnderLearningCurve, WorkSavedOverSampling
from active.helpers.engine import ActiveLearningEvents, ActiveLearningEngine
from active.helpers.evaluator import Evaluator
from active.core.utils.model import get_encoder_from_model
# import ignite
from ignite.engine import Events
from ignite.metrics import ConfusionMatrix
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# dimensionality reduction
from sklearn.manifold import TSNE
# others
import transformers
from math import ceil
from pydantic import BaseModel, validator
# base config and helper function
from active.scripts.run_train import Task, ExperimentConfig

def visualize_embeds(strategy):
    from matplotlib import pyplot as plt
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

    @property
    def num_steps(self):
        return ceil(self.budget / self.query_size)

    def build_strategy(self, model:transformers.PreTrainedModel) -> strategies.AbstractStrategy:    
        if self.strategy == 'random':
            return strategies.Random()
        elif self.strategy == 'least-confidence':
            return strategies.LeastConfidence(model)
        elif self.strategy == 'prediction-entropy':
            return strategies.PredictionEntropy(model)
        elif self.strategy == 'badge' and self.task == Task.SEQUENCE:
            return strategies.BadgeForSequenceClassification(
                get_encoder_from_model(model), 
                model.classifier
            )
        elif self.strategy == 'badge' and self.task in [Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING]:
            return strategies.BadgeForTokenClassification(
                get_encoder_from_model(model), 
                model.classifier
            )
        elif self.strategy == 'alps':
            return strategies.AlpsConstantEmbeddings(model, mlm_prob=0.15)
        elif self.strategy == 'egl':
            return strategies.EglByTopK(model, k=5)
        elif self.strategy == 'egl-sampling':
            return strategies.EglBySampling(model, k=8)
        elif self.strategy == 'entropy-over-max':
            return strategies.EntropyOverMax(model)
        elif self.strategy == 'entropy-over-max-ignore':
            return strategies.EntropyOverMax(model, ignore_labels=[0])
        elif self.strategy == 'entropy-over-max-sample':
            return strategies.EntropyOverMax(model, random_sample=True)

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

def active(config:str, seed:int, strategy:str, budget:int, query_size:int, use_cache:bool, disable_tqdm:bool =False):
    import wandb

    # parse configuration and apply overwrites
    config = AlExperimentConfig.parse_file(config)
    config.active.strategy = strategy or config.active.strategy
    config.active.budget = budget or config.active.budget
    config.active.query_size = query_size or config.active.query_size
    print("Config:", config.json(indent=2))
    
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load datasets
    ds = config.load_dataset(use_cache=use_cache)
    
    # keep tokenizer around as it is used in event handler
    tokenizer = config.model.tokenizer
    # load model and build strategy
    model = config.load_model()
    strategy = config.active.build_strategy(model)
    # attach progress bar to strategy
    if not disable_tqdm:
        ProgressBar(ascii=True, desc='Strategy').attach(strategy)
    
    # create trainer, validator and tester engines
    trainer = config.trainer.build_trainer(model) 
    validator = Evaluator(model)
    tester = Evaluator(model)
    # attach metrics
    config.data.attach_metrics(trainer, tag="train")
    config.data.attach_metrics(validator, tag="val")
    config.data.attach_metrics(tester, tag="test")
    # attach progress bar
    if not disable_tqdm:
        ProgressBar(ascii=True).attach(trainer, output_transform=lambda output: {'L': output['loss']})
        ProgressBar(ascii=True, desc='Validating').attach(validator)
        ProgressBar(ascii=True, desc='Testing').attach(tester)
    # attach confusion matrix metric to tester
    # needed for some active learning metrics
    ConfusionMatrix(
        # for nested bio tagging the label space is the entity
        # types, however the confusion matrix is computed over B,I,O tags
        num_classes=3 if config.task is Task.NESTED_BIO_TAGGING else len(config.data.label_space),
        output_transform=tester.get_logits_and_labels
    ).attach(tester, "cm")

    # create active learning loop
    loop = ActiveLoop(
        pool=ds['train'],
        strategy=strategy,
        batch_size=config.trainer.batch_size,
        query_size=config.active.query_size,
        init_strategy=strategy if isinstance(strategy, strategies.Alps) else strategies.Random()
    )
    # create active learning engine
    al_engine = ActiveLearningEngine(
        trainer=trainer,
        trainer_run_kwargs=config.trainer.run_kwargs,
        train_batch_size=config.trainer.batch_size,
        eval_batch_size=config.trainer.batch_size,
        train_val_ratio=0.8
    )
    
    @al_engine.on(Events.ITERATION_STARTED)
    def on_started(engine):
        # log active learning step
        i = engine.state.iteration
        print("-" * 8, "AL Step %i" % i, "-" * 8)
        
    @al_engine.on(ActiveLearningEvents.DATA_SAMPLING_COMPLETED)
    def save_selected_samples(engine):
        # create path to save samples in
        path = os.path.join(
            "output",
            os.environ.get("WANDB_RUN_GROUP", "default"),
            os.environ.get("WANDB_NAME", "default-run"),
        )
        os.makedirs(path, exist_ok=True)
        # get selected samples and re-create input texts
        data = engine.state.batch
        texts = tokenizer.batch_decode(
            sequences=[sample['input_ids'] for sample in data],
            skip_special_tokens=True
        )
        # save selected samples to file
        with open(os.path.join(path, "step-%i.txt" % len(engine.train_dataset)), 'w+') as f:
            f.write('\n'.join(texts))

    @al_engine.on(Events.ITERATION_COMPLETED)
    def evaluate_and_log(engine):
        
        # create validation and test dataloaders
        val_loader = DataLoader(engine.val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(ds['test'], batch_size=64, shuffle=False)
        # evaluate on val and test set
        val_metrics = validator.run(val_loader).metrics
        test_metrics = tester.run(test_loader).metrics
        # don't log test confusion matrix
        test_metrics = test_metrics.copy()
        test_metrics.pop('cm')
        # log both train and test metrics
        print("Step:", engine.state.iteration, "-" * 15)
        print("Training Converged:", trainer.converged)
        if config.task is Task.SEQUENCE:
            print("Train F-Score:", trainer.state.metrics['train/F'])
            print("Val   F-Score:", val_metrics['val/F'])
            print("Test  F-Score:", test_metrics['test/F'])
        elif config.task in (Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING):
            print("Train Entity F-Score:", trainer.state.metrics['train/entity/weighted avg/F'])
            print("Val   Entity F-Score:", val_metrics['val/entity/weighted avg/F'])
            print("Test  Entity F-Score:", test_metrics['test/entity/weighted avg/F'])
        
        # get total time spend in strategy
        strategy_time = loop.strategy.state.times[Events.COMPLETED.name]
        strategy_time = {'times/strategy': strategy_time} if strategy_time is not None else {}
        # log all remaining metrics to weights and biases
        wandb.log(
            data=(trainer.state.metrics | val_metrics | test_metrics | strategy_time),
            step=len(engine.train_dataset)
        )
    
        # check if there is an output to visualize
        if loop.strategy.output is not None:
            ax, fig = visualize_embeds(loop.strategy)
            ax.set(title="%s embedding iteration %i (t-SNE)" % (config.active.strategy, engine.state.iteration))
            wandb.log({"Embedding": wandb.Image(fig)}, step=len(engine.train_dataset))
            # close figure
            plt.close(fig)
    
    # add work saved over sampling metric
    wss = WorkSavedOverSampling(output_transform=lambda _: tester.state.metrics['cm'])
    wss.attach(al_engine, "test/wss")
   
    # add area under learning curve metrics 
    if config.task is Task.SEQUENCE:
        # area under f-score curve
        area = AreaUnderLearningCurve(
            output_transform=lambda _: (
                al_engine.state.iteration,
                tester.state.metrics['test/F']
            )
        )
        area.attach(al_engine, "test/Area(F)")

    elif config.task in (Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING):
        # area under token-level f-score curve
        area = AreaUnderLearningCurve(
            output_transform=lambda _: (
                al_engine.state.iteration,
                tester.state.metrics['test/token/F']
            )
        )
        area.attach(al_engine, "test/token/Area(F)")
        # area under entity-level f-score curve
        area = AreaUnderLearningCurve(
            output_transform=lambda _: (
                al_engine.state.iteration,
                tester.state.metrics['test/entity/weighted avg/F']
            )
        )
        area.attach(al_engine, "test/entity/weighted avg/Area(F)")

    # initialize wandb
    wandb_config = config.dict()
    wandb_config['data']['dataset'] = ds['train'].info.builder_name
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
    # parse arguments
    args = parser.parse_args()

    # run active learning experiment
    active(**vars(args))

if __name__ == '__main__':
    main()
