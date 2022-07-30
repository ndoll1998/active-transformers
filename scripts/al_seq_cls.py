import torch
import numpy as np
# datasets and transformers
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
# import active learning components
from src.active.loop import ActiveLoop
from src.active.strategies import *
# import optimization utilities
from src.active.utils.schedulers import LinearWithWarmup
from src.active.utils.params import TransformerParameterGroups
# import ignite progress bar and script utils
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from scripts.common import build_argument_parser, run_active_learning

def prepare_seq_cls_datasets(ds, tokenizer, max_length, label_column):
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
            .rename_column(label_column, 'labels')
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

def build_strategy(args, model):
    if args.strategy == 'random': return Random()
    elif args.strategy == 'least-confidence': return LeastConfidence(model)
    elif args.strategy == 'prediction-entropy': return PredictionEntropy(model)
    elif args.strategy == 'badge': return BadgeForSequenceClassification(model.bert, model.classifier)
    elif args.strategy == 'alps': return Alps(model, mlm_prob=0.15)
    elif args.strategy == 'egl': return EglByTopK(model, k=3)
    elif args.strategy == 'egl-fast': return EglFastByTopK(model, k=3)
    elif args.strategy == 'egl-sampling': return EglBySampling(model, k=5)
    elif args.strategy == 'egl-fast-sampling': return EglFastBySampling(model, k=5)

if __name__ == '__main__':
    
    # parse arguments
    parser = build_argument_parser()
    args = parser.parse_args()
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    # load and prepare dataset
    ds = datasets.load_dataset(args.dataset, split={'train': 'train', 'test': 'test'})
    ds = prepare_seq_cls_datasets(ds, tokenizer, args.max_length, args.label_column)
    num_labels = len(ds['train'].info.features['labels'].names)

    # load model and create optimizer
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_ckpt, num_labels=num_labels)
    optim = torch.optim.AdamW(TransformerParameterGroups(model, lr=args.lr, lr_decay=args.lr_decay, weight_decay=args.weight_decay))
    scheduler = LinearWithWarmup(optim, warmup_proportion=0.1)

    # create strategy and attach progress bar to strategy
    strategy = build_strategy(args, model)
    ProgressBar(ascii=True, desc='Strategy').attach(strategy)
    
    # create active learning loop
    loop = ActiveLoop(
        pool=ds['train'],
        strategy=strategy,
        batch_size=16,
        query_size=args.query_size,
        init_strategy=strategy if isinstance(strategy, Alps) else Random()
    )

    # run active learning loop
    run_active_learning(args, loop, model, optim, scheduler, ds)
