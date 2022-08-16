import torch
import numpy as np
# datasets and transformers
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForTokenClassification
)
# import active learning components
from src.active.loop import ActiveLoop
from src.active.strategies import *
# import optimization utilities
from src.active.utils.schedulers import LinearWithWarmup
from src.active.utils.params import TransformerParameterGroups
# import data processor
from src.data.processors import SequenceTaggingProcessor
# import ignite progress bar and script utils
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from scripts.common import build_argument_parser, run_active_learning

def prepare_token_cls_datasets(ds, tokenizer, min_length, max_length, label_column):

    # build processor
    processor = SequenceTaggingProcessor(
        tokenizer=tokenizer,
        dataset_info=next(iter(ds.values())).info,
        tags_field=label_column,
        max_length=max_length
    )
    # prepare datasets
    ds = {
        key: dataset \
            .filter(
                lambda ex: len(ex['tokens']) > min_length,
                desc="Filtering"
            )
            .map(
                processor,
                batched=False,
                desc=key
            )
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

def get_encoder_from_model(model):
    # get encoder model class   
    model_class = AutoModel._model_mapping.get(type(model.config), None)
    assert model_class is not None, "Model type not registered!"
    # find member of encoder class in model
    for module in model.children():
        if isinstance(module, model_class):
            return module
    # attribute error
    raise AttributeError("Encoder member of class %s not found" % model_class.__name__)

def build_strategy(args, model):
    if args.strategy == 'random': return Random()
    elif args.strategy == 'least-confidence': return LeastConfidence(model)
    elif args.strategy == 'prediction-entropy': return PredictionEntropy(model)
    elif args.strategy == 'badge': 
        return BadgeForTokenClassification(get_encoder_from_model(model), model.classifier)
    elif args.strategy == 'alps': return AlpsConstantEmbeddings(model, mlm_prob=0.15)
    elif args.strategy == 'egl': return EglByTopK(model, k=1)
    elif args.strategy == 'egl-sampling': return EglBySampling(model, k=8)
    elif args.strategy == 'entropy-over-max': return EntropyOverMax(model, ignore_labels=[0])

if __name__ == '__main__':
    
    # parse arguments
    parser = build_argument_parser()
    args = parser.parse_args()
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt, add_prefix_space=True)
    # load and prepare dataset
    ds = datasets.load_dataset(args.dataset, split={'train': 'train', 'test': 'test'})
    ds = prepare_token_cls_datasets(ds, tokenizer, args.min_length, args.max_length, args.label_column)
    num_labels = ds['train'].info.features[args.label_column].feature.num_classes

    # load model and create optimizer
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_ckpt, num_labels=num_labels)
    optim = torch.optim.AdamW(
        TransformerParameterGroups(
            model, 
            lr=args.lr, 
            lr_decay=args.lr_decay, 
            weight_decay=args.weight_decay
        )
    )

    # create strategy and attach progress bar to strategy
    strategy = build_strategy(args, model)
    ProgressBar(ascii=True, desc='Strategy').attach(strategy)
    
    # create active learning loop
    loop = ActiveLoop(
        pool=ds['train'],
        strategy=strategy,
        batch_size=64,
        query_size=args.query_size,
        init_strategy=strategy if isinstance(strategy, Alps) else Random()
    )

    # run active learning loop
    run_active_learning(args, loop, model, optim, None, ds)
