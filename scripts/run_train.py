import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
# datasets and transformers
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
# import trainer and evaluator engines
from src.active.helpers.engines import Trainer, Evaluator
# import ignite
from ignite.engine import Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# others
import wandb
from scripts.run_active import (
    prepare_datasets, 
    build_data_processor, 
    build_data_filter,
    attach_metrics
)


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
    ds = datasets.load_dataset(args.dataset, split={'train': 'train', 'test': 'test'})
    dataset_info=next(iter(ds.values())).info
    
    # get number of labels in data
    num_labels = \
        dataset_info.features[args.label_column].num_classes if args.task == 'sequence' else \
        dataset_info.features[args.label_column].feature.num_classes

    # load tokenizer and create task-specific data processor and filter
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    processor = build_data_processor(tokenizer, dataset_info, args)
    filter_ = build_data_filter(tokenizer, args)

    # prepare dataset
    ds = prepare_datasets(ds, processor, filter_)
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
    
    ModelTypes = {
        'sequence': AutoModelForSequenceClassification,
        'token': AutoModelForTokenClassification
    }
    # load model and create optimizer
    model = ModelTypes[args.task].from_pretrained(args.pretrained_ckpt, num_labels=num_labels)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None

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
