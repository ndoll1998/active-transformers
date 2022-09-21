import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
# datasets and transformers
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
# import trainer and evaluator engines
from src.active.helpers.engines import Trainer, Evaluator
# import data processor for token classification tasks
from src.data.processor import TokenClassificationProcessor
# import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Recall, Precision, Average, Fbeta, Accuracy
# others
import wandb

#
# Data Preparation
#

def build_data_processor(tokenizer, dataset_info, args):

    if args.task == 'sequence':
        # tokenize inputs and carry label
        return lambda example: tokenizer(   
            text=example['text'],
            max_length=args.max_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        ) | {'labels': example[args.label_column]}

    if args.task == 'token':
        # create sequence tagging processor
        return TokenClassificationProcessor(
            tokenizer=tokenizer,
            dataset_info=dataset_info,
            tags_field=args.label_column,
            max_length=args.max_length
        )

    # task not recognized
    raise ValueError("Unknown task: %s" % args.task)
 
def build_data_filter(tokenizer, args):
    # filter out all examples that don't satify the minimum length
    return lambda example: len(example['input_ids']) -  example['input_ids'].count(tokenizer.pad_token_id) > args.min_length

def prepare_datasets(ds, processor, filter_, load_from_cache=False):
    # process and filter datasets
    ds = {
        key: dataset \
            .map(processor, batched=False, desc=key, load_from_cache_file=load_from_cache)
            .filter(filter_, batched=False, desc=key, load_from_cache_file=load_from_cache)
        for key, dataset in ds.items()
    }
    # set data formats
    for dataset in ds.values():
        dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )

    return ds

def load_and_preprocess_datasets(args, tokenizer):
    
    # load datasets
    ds = datasets.load_dataset(args.dataset, split={'train': 'train', 'test': 'test'})
    dataset_info=next(iter(ds.values())).info

    # load tokenizer and create task-specific data processor and filter
    processor = build_data_processor(tokenizer, dataset_info, args)
    filter_ = build_data_filter(tokenizer, args)

    # prepare dataset
    return prepare_datasets(ds, processor, filter_)


#
# Model and Metrics
#

def create_model_optim_scheduler(args, ds):
    
    dataset_info=next(iter(ds.values())).info
    # get number of labels in data
    num_labels = \
        dataset_info.features[args.label_column].num_classes if args.task == 'sequence' else \
        dataset_info.features[args.label_column].feature.num_classes
    
    ModelTypes = {
        'sequence': AutoModelForSequenceClassification,
        'token': AutoModelForTokenClassification
    }
    # load model and create optimizer
    model = ModelTypes[args.task].from_pretrained(args.pretrained_ckpt, num_labels=num_labels)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None

    return model, optim, scheduler

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

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks.")
    # specify task and data
    parser.add_argument("--task", type=str, default=None, choices=['sequence', 'token'], help="The learning task to solve. Either sequence or token classification.")
    parser.add_argument("--dataset", type=str, default=None, help="The dataset to use. Sequence Classification datasets must have feature 'text'. Token Classification datasets must have feature 'tokens'.")
    parser.add_argument("--label-column", type=str, default="label", help="Dataset column containing target labels.")
    parser.add_argument('--min-length', type=int, default=0, help="Minimum sequence length an example must fulfill. Samples with less tokens will be filtered from the dataset.")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum length of input sequences")
    # specify model and learning hyperparameters
    parser.add_argument("--pretrained-ckpt", type=str, default="bert-base-uncased", help="The pretrained model checkpoint")
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

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # load datasets
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    ds = load_and_preprocess_datasets(args, tokenizer=tokenizer)
    # load model, optimizer and scheduler
    model, optim, scheduler = create_model_optim_scheduler(args, ds)
    
    train_data, test_data = ds['train'], ds['test']
    # split validation dataset from train set
    train_data, val_data = random_split(train_data, [
        int(0.8 * len(train_data)),
        len(train_data) - int(0.8 * len(train_data))
    ])
    # create dataloaders
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
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
    
    # add event handler for testing and logging
    @trainer.on(Events.EPOCH_COMPLETED)
    def test_and_log(engine):
        # evaluate on val and test set
        val_state = validator.run(val_loader)
        test_state = tester.run(test_loader)
        # log both train and test metrics
        print("Step:", engine.state.iteration, "-" * 15)
        print("Train F-Score:", engine.state.metrics['train/F'])
        print("Val   F-Score:", val_state.metrics['val/F'])
        print("Test  F-Score:", test_state.metrics['test/F'])
        wandb.log(
            test_state.metrics | val_state.metrics | engine.state.metrics, 
            step=engine.state.iteration
        )

    # run training
    trainer.run(
        data=train_loader,
        max_epochs=args.epochs, 
        epoch_length=args.epoch_length
    )
