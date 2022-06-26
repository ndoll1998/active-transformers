# import torch
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
# import active learning loop and utils
from src.active.loop import ActiveLoop
from src.active.utils.engines import Trainer, Evaluator
# import ignite
from ignite.engine import Events
from ignite.metrics.recall import Recall
from ignite.metrics.precision import Precision
from ignite.metrics import Average, Fbeta, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
# others
from itertools import islice
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
    parser.add_argument("--steps", type=int, default=20, help="Number of Active Learning Steps")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs to train within a single AL step")
    parser.add_argument("--epoch-length", type=int, default=None, help="Number of update steps of an epoch")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size to use during training and evaluation")
    parser.add_argument("--query-size", type=int, default=25, help="Number of data points to query from pool at each AL step")
    parser.add_argument("--patience", type=int, default=5, help="Early Stopping Patience")
    parser.add_argument("--acc-threshold", type=float, default=0.98, help="Early Stopping Accuracy Threshold")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum length of input sequences")
    parser.add_argument("--use-cache", action='store_true', help="Use cached datasets if present")
    parser.add_argument("--model-cache", type=str, default="/tmp/model-cache", help="Cache to save the intermediate models at.")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed")
    # return parser
    return parser

def attach_metrics(engine):
    # create metrics
    L = Average(output_transform=type(engine).get_loss)
    A = Accuracy(output_transform=type(engine).get_logits_and_labels)
    R = Recall(output_transform=type(engine).get_logits_and_labels, average=True)
    P = Precision(output_transform=type(engine).get_logits_and_labels, average=True)
    F = Fbeta(beta=1.0, output_transform=type(engine).get_logits_and_labels, average=True)
    # attach metrics
    L.attach(engine, 'L')
    A.attach(engine, 'A')
    R.attach(engine, 'R')
    P.attach(engine, 'P')
    F.attach(engine, 'F')

def run_active_learning(args, loop, model, optim, scheduler, ds) -> None:
    # create the trainer, validater and tester
    trainer = Trainer(model, optim, scheduler, args.acc_threshold, args.patience, incremental=True, cache_dir=args.model_cache)
    validater = Evaluator(model)
    tester = Evaluator(model)
    # attach metrics
    attach_metrics(trainer)
    attach_metrics(tester)
    attach_metrics(validater)
    # attach progress bar
    ProgressBar().attach(trainer, output_transform=lambda output: {'L': output['loss']})
    ProgressBar(desc='Validating').attach(validater)
    ProgressBar(desc='Testing').attach(tester)
    
    config = vars(args)
    config['dataset'] = ds['train'].info.builder_name
    # create wandb logger, project and run name are set
    # as environment variables (see `run.sh`)
    logger = WandBLogger(config=config)
    # log train metrics after each train run
    logger.attach_output_handler(
        trainer,
        event_name=Events.COMPLETED,
        tag="train",
        metric_names='all',
        output_transform=lambda *_: {'steps': trainer.state.iteration},
        global_step_transform=lambda *_: len(trainer.state.dataloader.dataset)
    )
    # log validation metrics
    logger.attach_output_handler(
        validater,
        event_name=Events.COMPLETED,
        tag="val",
        metric_names='all',
        global_step_transform=lambda *_: len(trainer.state.dataloader.dataset)
    )
    # log test metrics
    logger.attach_output_handler(
        tester,
        event_name=Events.COMPLETED,
        tag="test",
        metric_names='all',
        global_step_transform=lambda *_: len(trainer.state.dataloader.dataset)
    )
     
    # run active learning loop
    train_data, val_data = [], []
    for i, samples in enumerate(islice(loop, args.steps), 1):
        print("-" * 8, "AL Step %i" % i, "-" * 8)

        # split into train and validation samples
        train_samples, val_samples = random_split(
            samples, [
                int(len(samples) * 0.9),
                len(samples) - int(len(samples) * 0.9)
            ]
        )
        # create datasets
        train_data.append(train_samples)
        val_data.append(val_samples)

        # create dataloaders
        train_loader = DataLoader(ConcatDataset(train_data), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(ConcatDataset(val_data), batch_size=64, shuffle=False)
        test_loader = DataLoader(ds['test'], batch_size=64, shuffle=False)
        # udpate validation loader
        trainer.val_loader = val_loader

        # try to train to convergence at most three times
        for _ in range(3):
            # train model
            state = trainer.run(train_loader, max_epochs=args.epochs, epoch_length=args.epoch_length)
            print("Training Converged:", trainer.converged)
            print("Train Metrics:", state.metrics)
            print("Final Train Accuracy:", trainer.train_accuracy)
            # check for convergence
            if trainer.converged:
                break

        # validate model
        state = validater.run(val_loader)
        print("Validation Metrics:", state.metrics)

        # test model
        state = tester.run(test_loader)
        print("Test Metrics:", state.metrics)
