import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# datasets and transformers
import datasets
from transformers import AutoTokenizer
# import active learning components
from active.core.loop import ActiveLoop
from active.core.strategies import *
from active.core.metrics import AreaUnderLearningCurve, WorkSavedOverSampling
from active.helpers.engine import ActiveLearningEvents, ActiveLearningEngine
from active.helpers.trainer import Trainer
from active.helpers.evaluator import Evaluator
from active.core.utils.model import get_encoder_from_model
# import ignite
from ignite.engine import Events
from ignite.metrics import ConfusionMatrix
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# dimensionality reduction
from sklearn.manifold import TSNE
# others
import wandb
from matplotlib import pyplot as plt
# helper functions
from active.scripts.run_train import (
    load_and_preprocess_datasets,
    create_trainer,
    attach_metrics
)
# argparse helpers
from typing import Literal
from defparse import Ignore


#
# Build Strategy and AL-Engine
#

def build_strategy(model, strategy, task):
    if strategy == 'random': return Random()
    elif strategy == 'least-confidence': return LeastConfidence(model)
    elif strategy == 'prediction-entropy': return PredictionEntropy(model)
    elif strategy == 'badge' and task == 'sequence':
        return BadgeForSequenceClassification(get_encoder_from_model(model), model.classifier)
    elif strategy == 'badge' and task == 'token':
        return BadgeForTokenClassification(get_encoder_from_model(model), model.classifier)
    elif strategy == 'alps': return AlpsConstantEmbeddings(model, mlm_prob=0.15)
    elif strategy == 'egl': return EglByTopK(model, k=5)
    elif strategy == 'egl-sampling': return EglBySampling(model, k=8)
    elif strategy == 'entropy-over-max': return EntropyOverMax(model)
    elif strategy == 'entropy-over-max-ignore': return EntropyOverMax(model, ignore_labels=[0])
    elif strategy == 'entropy-over-max-sample': return EntropyOverMax(model, random_sample=True)

def create_engine_and_loop(
    trainer:Ignore[Trainer],
    pool:Ignore[Dataset],
    # strategy
    task:Literal["token", "sequence"] ="token",
    strategy:str ="random",
    query_size:int =25,
    # engine run args
    steps:int =-1,
    # trainer run args
    epochs:int =50,
    epoch_length:int =None,
    min_epoch_length:int =16,
    batch_size:int =64
):
    """ Build the Active Learning Engine and Loop

        Args:
            trainer (Trainer): trainer instance to use
            pool (Dataset): unlabeled pool dataset
            task (str):
                The learning task to solve. Either `sequence` or 
                `token` classification.
            strategy (str): Active Learning Strategy to use
            steps (int): 
                number of Active Learning Steps. Defaults to -1
                meaning the whole pool of data will be processed.
            query_size (int): 
                Number of data points to query from pool at each AL step
            epochs (int):
                Maximum number of epochs to run the trainer.
            epoch_length (int):
                Number of update steps of a single epochs.
            min_epoch_length (int):
                Minimum number of update steps of a single epoch.
            batch_size (int):
                Batch size to use during training and evaluation.

        Returns:
            engine (ActiveLearningEngine): active learning engine instance
            loop (ActiveLoop): active learning loop instance
    """
    # create strategy and attach progress bar to strategy
    strategy = build_strategy(trainer.unwrapped_model, strategy, task)
    # create active learning loop
    loop = ActiveLoop(
        pool=pool,
        strategy=strategy,
        batch_size=batch_size,
        query_size=query_size,
        init_strategy=strategy if isinstance(strategy, Alps) else Random()
    )

    # create active learning engine
    al_engine = ActiveLearningEngine(
        trainer=trainer,
        trainer_run_kwargs=dict(
            max_epochs=epochs,
            epoch_length=epoch_length,
            min_epoch_length=min_epoch_length
        ),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        train_val_ratio=0.9
    )

    # return engine and loop
    return al_engine, loop

#
# Visualization
#

def visualize_embeds(strategy):
    # get selected indices and processing output of unlabeled pool
    idx = loop.strategy.selected_indices
    output = loop.strategy.output
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

def main():
    from defparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks using active learning.")
    # add arguments
    build_datasets = parser.add_args_from_callable(load_and_preprocess_datasets, group="Dataset Arguments")
    build_trainer = parser.add_args_from_callable(create_trainer, group="Trainer Arguments")
    build_engine_and_loop = parser.add_args_from_callable(create_engine_and_loop, group="Active Learning Arguments")
    # parse arguments
    args = parser.parse_args()

    # build datasets
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    ds = build_datasets()

    config = vars(args)
    config['dataset'] = ds['train'].info.builder_name
    # initialize wandb, project and run name are set
    # as environment variables (see `experiments/run-active.sh`)
    wandb.init(config=config)
    
    # build engine and loop
    al_engine, loop = build_engine_and_loop(
        trainer=build_trainer(),
        pool=ds['train']
    )
    
    # attach progress bar to strategy
    ProgressBar(ascii=True, desc='Strategy').attach(loop.strategy)
    
    # get trainer and create validater and tester
    trainer = al_engine.trainer
    validator = Evaluator(trainer.unwrapped_model)
    tester = Evaluator(trainer.unwrapped_model)
    # attach metrics
    attach_metrics(trainer, tag="train")
    attach_metrics(validator, tag="val")
    attach_metrics(tester, tag="test")
    # attach progress bar
    ProgressBar(ascii=True).attach(trainer, output_transform=lambda output: {'L': output['loss']})
    ProgressBar(ascii=True, desc='Validating').attach(validator)
    ProgressBar(ascii=True, desc='Testing').attach(tester)
    # attach confusion matrix metric to tester
    # needed for some active learning metrics
    ConfusionMatrix(
        num_classes=trainer.unwrapped_model.config.num_labels,
        output_transform=tester.get_logits_and_labels
    ).attach(tester, "cm")

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
        # print
        print(
            "Training Converged:", trainer.converged, 
            "Train F-Score: %.03f" % trainer.state.metrics['train/F']
        )
        # create validation and test dataloaders
        val_loader = DataLoader(engine.val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(ds['test'], batch_size=64, shuffle=False)
        # run on validation data
        val_metrics = validator.run(val_loader).metrics
        print("Validation Metrics:", val_metrics)

        # run on test data
        test_metrics = tester.run(test_loader).metrics
        print("Test Metrics:", test_metrics)
        # don't log test confusion matrix
        test_metrics = test_metrics.copy()
        test_metrics.pop('cm')

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
            ax.set(title="%s embedding iteration %i (t-SNE)" % (args.strategy, engine.state.iteration))
            wandb.log({"Embedding": wandb.Image(fig)}, step=len(engine.train_dataset))
            # close figure
            plt.close(fig)
    
    # add metrics to active learning engine
    wss = WorkSavedOverSampling(output_transform=lambda _: tester.state.metrics['cm'])
    area = AreaUnderLearningCurve(
        output_transform=lambda _: (
            # point of learning curve given
            # by iteration and accuracy value
            al_engine.state.iteration,
            tester.state.metrics['test/F']
        )
    )
    wss.attach(al_engine, "test/wss")
    area.attach(al_engine, "test/Area(F)")

    # run active learning experiment
    state = al_engine.run(loop, steps=args.steps)
    print("Active Learning Metrics:", state.metrics)
    # log active learning metric scores
    wandb.run.summary.update(state.metrics)

    # run finished
    wandb.finish(quiet=True)

if __name__ == '__main__':
    main()
