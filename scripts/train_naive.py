import torch
# import dataset
import datasets
from torch.utils.data import DataLoader
from src.data.processor import Conll2003Processor
# import ignite
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics.recall import Recall
from ignite.metrics.precision import Precision
from ignite.metrics import Fbeta, Average
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
# and ignite helpers
from src.nn.engines import Evaluator, Trainer
# import model
from transformers import (
    DistilBertTokenizer,
    DistilBertForTokenClassification
)

def attach_metrics(engine):
    # create metrics
    L = Average(output_transform=type(engine).get_loss)
    R = Recall(output_transform=type(engine).get_logits_and_labels, average=True)
    P = Precision(output_transform=type(engine).get_logits_and_labels, average=True)
    F = Fbeta(beta=1.0, output_transform=type(engine).get_logits_and_labels, average=True)
    # attach metrics
    L.attach(engine, 'L')
    R.attach(engine, 'R')
    P.attach(engine, 'P')
    F.attach(engine, 'F')
    # return metric names
    return ['L', 'R', 'P', 'F']

def prepare_conll2003(
    dataset:datasets.Dataset,
    processor:Conll2003Processor,
    use_cache:bool =False
) -> None:
    # filter out empty sampels
    dataset = dataset.filter(
        lambda e: len(e['tokens']) > 0,
        batched=False, 
        load_from_cache_file=use_cache, 
        desc="Filtering"
    )
    # preprocess
    dataset = dataset.map(
        processor, 
        batched=False, 
        load_from_cache_file=use_cache, 
        desc="Processing"
    )
    # and set format
    dataset.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'labels']
    )
    return dataset

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on the Conll2003 dataset. No active learning involved.")
    parser.add_argument("--lang", type=str, default='en', choices=['en', 'de'], help="Use the english or german Conll2003 dataset")
    parser.add_argument("--pretrained-ckpt", type=str, default="distilbert-base-uncased", help="The pretrained model checkpoint")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate used by optimizer")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train")
    parser.add_argument("--epoch-length", type=int, default=64, help="The maximum number of steps of an epoch")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size to use during training and evaluation")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum length of input sequences")
    parser.add_argument("--use-cache", type=bool, default=True, help="Use cached datasets if present")
    # parse arguments
    args = parser.parse_args()

    # load english/german conll2003 dataset
    if args.lang == 'en':
        ds = datasets.load_dataset('conll2003', download_mode=None if args.use_cache else 'force_redownload')
    elif args.lang == 'de':
        ds = datasets.load_dataset("src/data/conll2003.py", download_mode=None if args.use_cache else 'force_redownload')

    # print dataset sizes
    train_dataset, val_dataset, test_dataset = ds['train'], ds['validation'], ds['test']
    print("Train Size:     ", len(train_dataset))
    print("Validation Size:", len(val_dataset))
    print("Test Size:      ", len(test_dataset))

    # create data processor
    tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_ckpt)
    processor = Conll2003Processor(
        tokenizer=tokenizer, 
        dataset_info=train_dataset.info,
        tags_field='ner_tags',
        max_length=args.max_length
    )
    # prepare datasets and build dataloaders
    train_loader = DataLoader(prepare_conll2003(train_dataset, processor, args.use_cache), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(prepare_conll2003(val_dataset, processor, args.use_cache), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(prepare_conll2003(test_dataset, processor, args.use_cache), batch_size=args.batch_size, shuffle=False)

    # create model and optimizer
    model = DistilBertForTokenClassification.from_pretrained(args.pretrained_ckpt, num_labels=processor.num_labels)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # create trainer
    trainer = Trainer(model, optim)
    train_metrics = attach_metrics(trainer)
    ProgressBar().attach(trainer)
    # create evaluator
    evaluator = Evaluator(model)
    val_metrics = attach_metrics(evaluator)
    ProgressBar().attach(evaluator)
    # create tester
    tester = Evaluator(model)
    test_metrics = attach_metrics(tester)
    ProgressBar().attach(tester)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        print("-" * 7, "Train Metrics", "-" * 8)
        for name, val in engine.state.metrics.items():
            print(("%s:" % name).ljust(10), "%.04f" % val)
        print("-" * 30)

    # attach evaluation handler    
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        # evaluate and log validation metrics
        state = evaluator.run(val_loader, epoch_length=args.epoch_length)
        print("-" * 5, "Validation Metrics", "-" * 5)
        for name, val in state.metrics.items():
            print(("%s:" % name).ljust(10), "%.04f" % val)
        print("-" * 30)

    # create wandb logger
    logger = WandBLogger(
        project='master-thesis',
        name="naive-conll-%s" % args.pretrained_ckpt,
        config=vars(args),
        group="naive",
        mode="online"
    )

    # log train metrics to wandb
    logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="train",
        metric_names=train_metrics,
        global_step_transform=lambda *_: trainer.state.iteration
    )
    # log validation metrics to wandb
    logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="val",
        metric_names=val_metrics,
        global_step_transform=lambda *_: trainer.state.iteration
    )
    # log test metrics to wandb
    logger.attach_output_handler(
        tester,
        event_name=Events.EPOCH_COMPLETED,
        tag="test",
        metric_names=test_metrics,
        global_step_transform=lambda *_: trainer.state.iteration
    )

    # run trainer and test model
    state = trainer.run(train_loader, max_epochs=args.epochs, epoch_length=args.epoch_length)
    state = tester.run(test_loader, epoch_length=args.epoch_length)
    # log test metrics
    print("-" * 8, "Test Metrics", "-" * 8)
    for name, val in state.metrics.items():
        print(("%s:" % name).ljust(10), "%.04f" % val)
    print("-" * 30)
