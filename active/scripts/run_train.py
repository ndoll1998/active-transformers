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
from active.helpers.trainer import Trainer
from active.helpers.evaluator import Evaluator
# import data processor for token classification tasks
from active.helpers.processor import TokenClassificationProcessor
# import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Recall, Precision, Average, Fbeta, Accuracy
# others
import wandb
from typing import Literal

def load_and_preprocess_datasets(
    pretrained_ckpt:str ='distilbert-base-uncased',
    task:Literal["token", "sequence"] ='token',
    dataset:str ='conll2003',
    label_column:str ='ner_tags',
    min_length:int =0,
    max_length:int =32,
    use_cache:bool =False
):
    """ Load and Preprocess train and test datasets.

        Args:
            pretrained_ckpt (str):
                pre-trained tranformer checkpoint to use.
            task (str): 
                The learning task to solve.
            dataset (str):
                The dataset to use. Must match the `task` argument.
            label_column (str):
                Dataset column containing target labels.
            min_length (int): 
                Minimum sequence length an example must fulfill.
                Samples with less tokens will be filtered from dataset.
            max_length (int):
                Maximum length of input sequences. Samples with more
                tokens will be truncated.
            use_cache (bool):
                Whether to use cached preprocessed datasets or do
                the preprocessing explicitly.

        Returns:
            ds (dict): dictionary containing the train and test dataset.
    """

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
    # load datasets
    ds = datasets.load_dataset(dataset, split={'train': 'train', 'test': 'test'})
    dataset_info=next(iter(ds.values())).info

    if task == 'sequence':
        # tokenize inputs and carry label
        processor = lambda example: tokenizer(   
            text=example['text'],
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False
        ) | {'labels': example[label_column]}

    elif task == 'token':
        # create sequence tagging processor
        processor = TokenClassificationProcessor(
            tokenizer=tokenizer,
            dataset_info=dataset_info,
            tags_field=label_column,
            max_length=max_length
        )

    else:
        # task not recognized
        raise ValueError("Unknown task: %s" % task)
    
    # filter out all examples that don't satify the minimum length
    filter_ = lambda example: len(example['input_ids']) - example['input_ids'].count(tokenizer.pad_token_id) > min_length
        
    # process and filter datasets
    ds = {
        key: dataset \
            .map(processor, batched=False, desc=key, load_from_cache_file=use_cache)
            .filter(filter_, batched=False, desc=key, load_from_cache_file=use_cache)
        for key, dataset in ds.items()
    }
    # set data formats
    for dataset in ds.values():
        dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )

    return ds

def create_trainer(
    pretrained_ckpt:str ='distilbert-base-uncased',
    task:Literal["token", "sequence"] ='token',
    dataset:str ='conll2003',
    label_column:str ='ner_tags',
    lr:float =2e-5,
    weight_decay:float =0.01,
    acc_threshold:float =2.0,
    patience:int =15,
    incremental:bool =True,
    # additional arguments not used in the function
    # itself but needed for running the trainer
    epochs:int =50,
    epoch_length:int =None,
    min_epoch_length:int =16,
    batch_size:int =32,
    seed:int =1337
):
    """ Create the transformer model, optimizer and scheduler

        Args:
            pretrained_ckpt (str):
                pre-trained tranformer checkpoint to use.
            task (str): 
                The learning task to solve. Either `sequence` or 
                `token` classification.
            dataset (str):
                The dataset to use. Must match the `task` argument.
            label_column (str):
                Dataset column containing target labels.
            lr (float): 
                Learning rate used by optimizer.
            weight_decay (float): 
                Weight decay rate used by optimizer.
            acc_threshold (float):
                Accuracy threshold to detect convergence.
            patience (int):
                Early Stopping Patience used by trainer.
            incremental (bool):
                Whether to use an incremental trainer or to reset the
                model before each active learning step.
            epochs (int):
                Maximum number of epochs to run the trainer.
            epoch_length (int):
                Number of update steps of a single epochs.
            min_epoch_length (int):
                Minimum number of update steps of a single epoch.
            batch_size (int):
                Batch size to use during training and evaluation.
            seed (int):
                random seed to use

        Returns:
            trainer (Trainer): trainer instance
    """
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # get dataset info
    builder = datasets.load_dataset_builder(dataset)
    dataset_info = builder._info()
    # get number of labels in data
    num_labels = \
        dataset_info.features[label_column].num_classes if task == 'sequence' else \
        dataset_info.features[label_column].feature.num_classes
    
    ModelTypes = {
        'sequence': AutoModelForSequenceClassification,
        'token': AutoModelForTokenClassification
    }
    # load model and create optimizer
    model = ModelTypes[task].from_pretrained(pretrained_ckpt, num_labels=num_labels)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # build trainer
    return Trainer(
        model=model,
        optim=optim,
        scheduler=None,
        acc_threshold=acc_threshold,
        patience=patience,
        incremental=incremental
    )

def attach_metrics(engine, tag=None):
    # create metrics
    L = Average(output_transform=lambda out: out['loss'])
    A = Accuracy(output_transform=type(engine).get_logits_and_labels)
    R = Recall(output_transform=type(engine).get_logits_and_labels, average=True)
    P = Precision(output_transform=type(engine).get_logits_and_labels, average=True)
    F = Fbeta(beta=1.0, output_transform=type(engine).get_logits_and_labels, average=True)
    # attach metrics
    L.attach(engine, 'L' if tag is None else ('%s/L' % tag))
    A.attach(engine, 'A' if tag is None else ('%s/A' % tag))
    R.attach(engine, 'R' if tag is None else ('%s/R' % tag))
    P.attach(engine, 'P' if tag is None else ('%s/P' % tag))
    F.attach(engine, 'F' if tag is None else ('%s/F' % tag))

def main():
    from defparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks.")
    build_datasets = parser.add_args_from_callable(load_and_preprocess_datasets, group="Dataset Arguments")
    build_trainer = parser.add_args_from_callable(create_trainer, group="Trainer Arguments", ignore=["incremental"])
    # parse arguments
    args = parser.parse_args()

    # load datasets and build trainer
    ds = build_datasets()
    trainer = build_trainer()
    
    train_data, test_data = ds['train'], ds['test']
    # split validation dataset from train set
    train_data, val_data = random_split(train_data, [
        int(0.8 * len(train_data)),
        len(train_data) - int(0.8 * len(train_data))
    ])

    # create dataloaders
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    config = vars(args)
    config['dataset'] = ds['train'].info.builder_name
    # initialize wandb, project and run name are set
    # as environment variables (see `run.sh`)
    wandb.init(config=config)

    # create the validater and tester
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
        epoch_length=args.epoch_length,
        min_epoch_length=args.min_epoch_length
    )
   
if __name__ == '__main__':
    main() 
