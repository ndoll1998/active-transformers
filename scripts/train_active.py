import os
import torch
# import dataset
import datasets
from torch.utils.data import ConcatDataset, DataLoader
from src.data.processor import Conll2003Processor
# import active learning components
from src.active.loop import ActiveLoop
from src.active.heuristics import LeastConfidence, Random
# import ignite
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics.recall import Recall
from ignite.metrics.precision import Precision
from ignite.metrics import Fbeta, Average
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
# and ignite helpers
from src.utils.engines import Evaluator, Trainer
# import model
from transformers import (
    DistilBertTokenizer,
    DistilBertForTokenClassification
)
# import helpers
from copy import deepcopy
from itertools import islice
from tempfile import TemporaryDirectory
from scripts.train_naive import attach_metrics, load_conll2003

def create_trainer(model, optim, logger, patience, tmpdir, dataset_size):
    # create trainer
    trainer = Trainer(model, optim)
    train_metrics = attach_metrics(trainer)
    ProgressBar(desc="Training").attach(trainer)
    # early stopping based on training metrics
    stopper = EarlyStopping(
        patience=patience,
        score_function=lambda e: e.state.metrics['F'],
        trainer=trainer
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, stopper)
    # add logger
    logger.attach_output_handler(
        trainer,
        event_name=Events.COMPLETED,
        tag="train",
        # log train metrics
        metric_names=train_metrics,
        # and number of train steps until convergence
        output_transform=lambda *_: {'steps': trainer.state.iteration},
        global_step_transform=lambda *_: dataset_size
    )
    # add model checkpoint handler
    ckpt = Checkpoint(
        to_save={'model': model},
        save_handler=DiskSaver(dirname=tmpdir, require_empty=False),
        score_function=lambda e: e.state.metrics['F'],
        n_saved=1
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, ckpt)
    # load best model after training finished
    @trainer.on(Events.COMPLETED)
    def load_best(engine):
        ckpt.load_objects(
            to_load={'model': model},
            checkpoint=os.path.join(tmpdir, ckpt.last_checkpoint)
        )
        
    # return trainer
    return trainer
    
def create_evaluator(model, logger, dataset_size):
    # create evaluator
    evaluator = Evaluator(model)
    val_metrics = attach_metrics(evaluator)
    ProgressBar(desc="Validating").attach(evaluator)
    # add logger
    logger.attach_output_handler(
        evaluator,
        event_name=Events.COMPLETED,
        tag="val",
        metric_names=val_metrics,
        global_step_transform=lambda *_: dataset_size
    )

    return evaluator

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on the Conll2003 dataset. No active learning involved.")
    parser.add_argument("--lang", type=str, default='en', choices=['en', 'de'], help="Use the english or german Conll2003 dataset")
    parser.add_argument("--pretrained-ckpt", type=str, default="distilbert-base-uncased", help="The pretrained model checkpoint")
    parser.add_argument("--heuristic", type=str, default="random", choices=['random', 'least-confidence'], help="Active Learning Heuristic to use")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate used by optimizer")
    parser.add_argument("--steps", type=int, default=10, help="Number of Active Learning Steps")
    parser.add_argument("--epochs", type=int, default=45, help="Maximum number of epochs to train within a single AL step")
    parser.add_argument("--epoch-length", type=int, default=64, help="Number of update steps of an epoch")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size to use during training and evaluation")
    parser.add_argument("--query-size", type=int, default=8, help="Number of data points to query from pool at each AL step")
    parser.add_argument("--patience", type=int, default=2, help="Early Stopping Patience")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum length of input sequences")
    parser.add_argument("--use-cache", type=bool, default=True, help="Use cached datasets if present")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # parse arguments
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)

    # create tokenizer and load datasets
    tokenizer = DistilBertTokenizer.from_pretrained(args.pretrained_ckpt)
    ds, processor = load_conll2003(args.lang, tokenizer, args.max_length, args.use_cache)    

    # create validation loader as it is constant through AL steps
    train_dataset, val_dataset = ds['train'], ds['validation']
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # create model and save initial parameters
    model = DistilBertForTokenClassification.from_pretrained(args.pretrained_ckpt, num_labels=processor.num_labels)
    init_ckpt = deepcopy(model.state_dict())
    
    # create heuristic
    if args.heuristic == 'random':
        heuristic = Random()
    elif args.heuristic == 'least-confidence':
        heuristic = LeastConfidence(model)
        ProgressBar(desc="Heuristic").attach(heuristic)

    # active learning loop
    loop = ActiveLoop(
        pool=train_dataset,
        heuristic=heuristic,
        batch_size=args.batch_size * 8,
        query_size=args.query_size
    )

    # create temporary directory to store model checkpoint
    with TemporaryDirectory() as tmpdir:
        
        # create wandb logger
        logger = WandBLogger(
            project='master-thesis',
            name="AL-conll-%s-%s" % (args.heuristic, args.pretrained_ckpt),
            config=vars(args),
            group="AL",
            mode="online"
        )

        datasets = []
        # run active learning loop
        for i, new_data in enumerate(islice(loop, args.steps), 1):

            print("-" * 8, "AL Step %i" % i, "-" * 8)

            # add new data to list
            datasets.append(new_data)
            size = sum(map(len, datasets))
            # create datalaoder
            loader = DataLoader(
                ConcatDataset(datasets), 
                batch_size=args.batch_size, 
                shuffle=True
            )

            # load initial model checkpoint and create optimizer
            model.load_state_dict(init_ckpt)
            optim = torch.optim.Adam(model.parameters(), lr=args.lr)

            # train the model
            trainer = create_trainer(model, optim, logger, args.patience, tmpdir, size)
            state = trainer.run(loader, max_epochs=args.epochs, epoch_length=args.epoch_length)
            print("Train Metrics:", state.metrics)
            
            # evaluate model
            evaluator = create_evaluator(model, logger, size)
            state = evaluator.run(val_loader)
            print("Val Metrics:", state.metrics)
