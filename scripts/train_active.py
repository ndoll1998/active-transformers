import torch
# import dataset
import datasets
from torch.utils.data import DataLoader
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
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
# and ignite helpers
from src.nn.engines import Evaluator, Trainer
# import model
from transformers import (
    DistilBertTokenizer,
    DistilBertForTokenClassification
)
# import helpers
from copy import deepcopy
from scripts.train_naive import attach_metrics, prepare_conll2003

class EmptyDataset(torch.utils.data.Dataset):
    """ Helper Dataset """
    def __len__(self) -> int:
        return 0

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on the Conll2003 dataset. No active learning involved.")
    parser.add_argument("--lang", type=str, default='en', choices=['en', 'de'], help="Use the english or german Conll2003 dataset")
    parser.add_argument("--pretrained-ckpt", type=str, default="distilbert-base-uncased", help="The pretrained model checkpoint")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate used by optimizer")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train")
    parser.add_argument("--epoch-length", type=int, default=64, help="The maximum number of steps of an epoch")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size to use during training and evaluation")
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
    # prepare datasets and get a random split for initial model training
    dataset = prepare_conll2003(test_dataset, processor, use_cache=True)

    # create model and save initial parameters
    model = DistilBertForTokenClassification.from_pretrained(args.pretrained_ckpt, num_labels=processor.num_labels)
    init_ckpt = deepcopy(model.state_dict())

    # create heuristic
    heuristic = LeastConfidence(model)
    ProgressBar("Heuristic").attach(heuristic)

    # active learning loop
    loop = ActiveLoop(
        pool=dataset,
        heuristic=Random(), #heuristic,
        batch_size=args.batch_size,
        query_size=4
    )

    data = EmptyDataset()
    for i, new_data in enumerate(loop):

        # add new data to data and create dataloader
        data = data + new_data
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

        # load initial model checkpoint and create optimizer
        model.load_state_dict(init_ckpt)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        # create trainer
        trainer = Trainer(model, optim)
        attach_metrics(trainer)
        ProgressBar(desc="Training").attach(trainer)
        # create evaluator
        evaluator = Evaluator(model)
        attach_metrics(evaluator)
        ProgressBar(desc="Validating").attach(trainer)

        # run training
        state = trainer.run(loader, max_epochs=args.epochs)
        print(state.metrics)

        if i == 5:
            break
