# import torch
import torch
from torch.utils.data import DataLoader
# import active learning loop and utils
from src.active.loop import ActiveLoop
from src.active.engine import ConvergenceRetryEvents, ActiveLearningEngine
from src.active.metrics import AreaUnderLearningCurve, WorkSavedOverSampling
from src.active.utils.engines import Trainer, Evaluator
# import ignite
from ignite.engine import Events
from ignite.metrics import Recall, Precision, Average,Fbeta, Accuracy, ConfusionMatrix
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
# dimensionality reduction
from sklearn.manifold import TSNE
# others
import wandb
from itertools import islice
from matplotlib import pyplot as plt
from argparse import ArgumentParser

def build_argument_parser() -> ArgumentParser:
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence classification tasks using active learning.")
    parser.add_argument("--dataset", type=str, default='ag_news', help="The text classification dataset to use. Must have features 'text' and 'label'.")
    parser.add_argument("--label-column", type=str, default="label", help="Dataset column containing target labels.")
    parser.add_argument("--pretrained-ckpt", type=str, default="bert-base-uncased", help="The pretrained model checkpoint")
    parser.add_argument("--strategy", type=str, default="random", help="Active Learning Strategy to use")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate used by optimizer")
    parser.add_argument("--lr-decay", type=float, default=1.0, help="Layer-wise learining rate decay")
    parser.add_argument("--weight-decay", type=float, default=1.0, help="Weight decay rate")
    parser.add_argument("--steps", type=int, default=-1, help="Number of Active Learning Steps. Defaults to -1 meaning the whole dataset will be processed.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs to train within a single AL step")
    parser.add_argument("--epoch-length", type=int, default=None, help="Number of update steps of an epoch")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size to use during training and evaluation")
    parser.add_argument("--query-size", type=int, default=25, help="Number of data points to query from pool at each AL step")
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping Patience")
    parser.add_argument("--acc-threshold", type=float, default=0.98, help="Early Stopping Accuracy Threshold")
    parser.add_argument('--min-length', type=int, default=0, help="Minimum sequence length an example must fulfill. Samples with less tokens will be filtered from the dataset.")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum length of input sequences")
    parser.add_argument("--use-cache", action='store_true', help="Use cached datasets if present")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed")
    # return parser
    return parser

def attach_metrics(engine, tag):
    # create metrics
    L = Average(output_transform=type(engine).get_loss)
    A = Accuracy(output_transform=type(engine).get_logits_and_labels)
    R = Recall(output_transform=type(engine).get_logits_and_labels, average=True)
    P = Precision(output_transform=type(engine).get_logits_and_labels, average=True)
    F = Fbeta(beta=1.0, output_transform=type(engine).get_logits_and_labels, average=True)
    # attach metrics
    L.attach(engine, '%s/L' % tag)
    A.attach(engine, '%s/A' % tag)
    R.attach(engine, '%s/R' % tag)
    P.attach(engine, '%s/P' % tag)
    F.attach(engine, '%s/F' % tag)

def visualize_embeds(strategy):
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

def run_active_learning(args, loop, model, optim, scheduler, ds) -> None:
    
    config = vars(args)
    config['dataset'] = ds['train'].info.builder_name
    # initialize wandb, project and run name are set
    # as environment variables (see `run.sh`)
    wandb.init(config=config)
    
    # create the trainer, validater and tester
    trainer = Trainer(model, optim, scheduler, args.acc_threshold, args.patience, incremental=True)
    validator = Evaluator(model)
    tester = Evaluator(model)
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
        num_classes=model.config.num_labels,
        output_transform=tester.get_logits_and_labels
    ).attach(tester, "cm")
    
    # create active learning engine
    al_engine = ActiveLearningEngine(
        trainer=trainer,
        trainer_run_kwargs=dict(
            max_epochs=args.epochs,
            epoch_length=args.epoch_length
        ),
        train_batch_size=args.batch_size,
        eval_batch_size=64,
        max_convergence_retries=3,
        train_val_ratio=0.9
    )
    
    @al_engine.on(Events.ITERATION_STARTED)
    def on_started(engine):
        i = engine.state.iteration
        print("-" * 8, "AL Step %i" % i, "-" * 8)

    @al_engine.on(ConvergenceRetryEvents.CONVERGED | ConvergenceRetryEvents.CONVERGENCE_RETRY_COMPLETED)
    def on_converged(engine):
        print("Training Converged:", engine.trainer.converged, "Train F-Score: %.03f" % trainer.state.metrics['train/F'])
        # print("Final Train Accuracy:", engine.trainer.train_accuracy)

    @al_engine.on(Events.ITERATION_COMPLETED)
    def evaluate_and_log(engine):
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
        # log all remaining metrics to weights and biases
        wandb.log(
            data=(trainer.state.metrics | val_metrics | test_metrics),
            step=len(engine.train_dataset)
        )
    
        # check if there is an output to visualize
        if loop.strategy.output is not None:
            ax, fig = visualize_embeds(loop.strategy)
            ax.set(title="%s embedding iteration %i (t-SNE)" % (args.strategy, engine.state.iteration))
            wandb.log({"Embedding": wandb.Image(fig)}, step=len(engine.train_dataset))
    
    # add metrics to active learning engine
    wss = WorkSavedOverSampling(output_transform=lambda _: tester.state.metrics['cm'])
    area = AreaUnderLearningCurve(
        output_transform=lambda _: (
            # point of learning curve given
            # by iteration and accuracy value
            al_engine.state.iteration,
            tester.state.metrics['test/A']
        )
    )
    wss.attach(al_engine, "test/wss")
    area.attach(al_engine, "test/Area(Accuracy)")

    # run active learning experiment
    state = al_engine.run(loop, steps=args.steps)
    print("Active Learning Metrics:", state.metrics)
    # log active learning metric scores
    wandb.run.summary.update(state.metrics)

    # run finished
    wandb.finish(quiet=True)
