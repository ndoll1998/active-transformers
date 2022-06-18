import os
import re
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    random_split
)
import numpy as np
# datasets and transformers
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
# import active learning components
from src.active.loop import ActiveLoop
from src.active.strategies.random import Random
from src.active.strategies.uncertainty import (
    LeastConfidence,
    PredictionEntropy
)
from src.active.strategies.badge import BadgeForSequenceClassification
from src.active.strategies.alps import Alps
from src.active.strategies.egl import (
    EglByTopK,
    EglBySampling,
    EglFastByTopK,
    EglFastBySampling
)
# import utilities
from src.utils.engines import Trainer, Evaluator
from src.utils.schedulers import LinearWithWarmup
from src.utils.params import TransformerParameterGroups
# import ignite metrics and handlers
from ignite.engine import Events
from ignite.metrics.recall import Recall
from ignite.metrics.precision import Precision
from ignite.metrics import Average, Fbeta, Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
# others
import wandb
from itertools import islice

def prepare_seq_cls_datasets(ds, tokenizer, max_length):
    # prepare datasets
    ds = {
        key: dataset \
            # encode texts
            .map(
                lambda example: tokenizer(
                    text=example['text'],
                    max_length=max_length,
                    truncation=True,
                    padding='max_length',
                    return_token_type_ids=False
                ),
                batched=True,
                desc=key
            ) \
            # rename label column to match classifier naming convention
            .rename_column('label', 'labels')
        # prepare each dataset in the dict
        for key, dataset in ds.items()
    }
    # set formats
    for dataset in ds.values():
        dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )

    return ds

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

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on the Conll2003 dataset. No active learning involved.")
    parser.add_argument("--dataset", type=str, default='ag_news', help="The text classification dataset to use. Must have features 'text' and 'label'.")
    parser.add_argument("--pretrained-ckpt", type=str, default="bert-base-uncased", help="The pretrained model checkpoint")
    parser.add_argument("--strategy", type=str, default="random", 
        choices=['random', 'least-confidence', 'prediction-entropy', 'badge', 'alps', 'egl', 'egl-fast', 'egl-sampling', 'egl-fast-sampling'], 
        help="Active Learning Strategy to use"
    )
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
    parser.add_argument("--seed", type=int, default=2022, help="Random seed")
    # parse arguments
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    # load and prepare dataset
    ds = datasets.load_dataset(args.dataset, split={'train': 'train', 'test': 'test'})
    ds = prepare_seq_cls_datasets(ds, tokenizer, args.max_length)
    num_labels = len(ds['train'].info.features['labels'].names)

    # load model and create optimizer
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_ckpt, num_labels=num_labels)
    optim = torch.optim.AdamW(TransformerParameterGroups(model, lr=args.lr, lr_decay=args.lr_decay, weight_decay=args.weight_decay))
    scheduler = LinearWithWarmup(optim, warmup_proportion=0.1)

    # create the trainer, validater and tester
    trainer = Trainer(model, optim, scheduler, args.acc_threshold, args.patience, incremental=True)
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
    # create wandb logger
    # project and run name are set as environment variables
    # (see `run.sh`)
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

    # create strategy
    if args.strategy == 'random': strategy = Random()
    elif args.strategy == 'least-confidence': strategy = LeastConfidence(model)
    elif args.strategy == 'prediction-entropy': strategy = PredictionEntropy(model)
    elif args.strategy == 'badge': strategy = BadgeForSequenceClassification(model.bert, model.classifier)
    elif args.strategy == 'alps': strategy = Alps(model, mlm_prob=0.15)
    elif args.strategy == 'egl': strategy = EglByTopK(model, k=3)
    elif args.strategy == 'egl-fast': strategy = EglFastByTopK(model, k=3)
    elif args.strategy == 'egl-sampling': strategy = EglBySampling(model, k=5)
    elif args.strategy == 'egl-fast-sampling': strategy = EglFastBySampling(model, k=5)
    # attach progress bar to strategy
    ProgressBar(desc='Strategy').attach(strategy)

    # list of all strategies that can be also used as initial strategies,
    # i.e. strategies that do necessarily depend on a finetuned model
    valid_init_strategies = (
        Alps,
        EglByTopK,
        EglBySampling,
        EglFastBySampling
    )
    # create active learning loop
    loop = ActiveLoop(
        pool=ds['train'],
        strategy=strategy,
        batch_size=64,
        query_size=args.query_size,
        init_strategy=strategy if isinstance(strategy, valid_init_strategies) else Random()
    )
    
    # active learning loop
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
