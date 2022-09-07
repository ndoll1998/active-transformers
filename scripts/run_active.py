import os
import torch
import numpy as np
from torch.utils.data import DataLoader
# datasets and transformers
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
# import active learning components
from src.active.loop import ActiveLoop
from src.active.strategies import *
from src.active.engine import ActiveLearningEvents, ActiveLearningEngine
from src.active.metrics import AreaUnderLearningCurve, WorkSavedOverSampling
from src.active.utils.engines import Trainer, Evaluator
# import data processor for token classification tasks
from src.data.processor import TokenClassificationProcessor
# import ignite
from ignite.engine import Events
from ignite.metrics import Recall, Precision, Average,Fbeta, Accuracy, ConfusionMatrix
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# dimensionality reduction
from sklearn.manifold import TSNE
# others
import wandb
from matplotlib import pyplot as plt

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
    return lambda example: example['input_ids'].count(tokenizer.pad_token_id) > args.min_length

def prepare_datasets(ds, processor, filter_):
    # process and filter datasets
    ds = {
        key: dataset \
            .map(processor, batched=False, desc=key, load_from_cache_file=False)
            .filter(filter_, batched=False, desc=key, load_from_cache_file=False)
        for key, dataset in ds.items()
    }
    # set data formats
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
    elif args.strategy == 'badge' and args.task == 'sequence':
        return BadgeForSequenceClassification(get_encoder_from_model(model), model.classifier)
    elif args.strategy == 'badge' and args.task == 'token':
        return BadgeForTokenClassification(get_encoder_from_model(model), model.classifier)
    elif args.strategy == 'alps': return AlpsConstantEmbeddings(model, mlm_prob=0.15)
    elif args.strategy == 'egl': return EglByTopK(model, k=3)
    elif args.strategy == 'egl-sampling': return EglBySampling(model, k=5)
    elif args.strategy == 'entropy-over-max': return EntropyOverMax(model)
    elif args.strategy == 'entropy-over-max-sample': return EntropyOverMax(model, random_sample=True)

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

def visualize_embeds(strategy):
    # get selected indices and processing output of unlabeled pool
    idx = strategy.selected_indices
    output = strategy.output
    output = output if output.ndim == 2 else \
        output.reshape(-1, 1) if output.ndim == 1 else \
        output.flatten(start_dim=1)
    # reduce dimension for visualization using t-sne    
    X = TSNE(
        n_components=2,
        perplexity=50,
        learning_rate='auto',
        init='random'
    ).fit_transform(
        X=output
    )
    # plot
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], s=1.0, color="blue", alpha=0.1)
    ax.scatter(X[idx, 0], X[idx, 1], s=2.0, color='red', alpha=1.0)
    ax.legend(['pool', 'query'])
    # return axes and figure
    return ax, fig

if __name__ == '__main__':

    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks using active learning.")
    # specify task and data
    parser.add_argument("--task", type=str, default=None, choices=['sequence', 'token'], help="The learning task to solve. Either sequence or token classification.")
    parser.add_argument("--dataset", type=str, default=None, help="The dataset to use. Sequence Classification datasets must have feature 'text'. Token Classification datasets must have feature 'tokens'.")
    parser.add_argument("--label-column", type=str, default="label", help="Dataset column containing target labels.")
    parser.add_argument('--min-length', type=int, default=0, help="Minimum sequence length an example must fulfill. Samples with less tokens will be filtered from the dataset.")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum length of input sequences")
    # specify strategy and active learning params
    parser.add_argument("--strategy", type=str, default="random", help="Active Learning Strategy to use")
    parser.add_argument("--query-size", type=int, default=25, help="Number of data points to query from pool at each AL step")
    parser.add_argument("--steps", type=int, default=-1, help="Number of Active Learning Steps. Defaults to -1 meaning the whole dataset will be processed.")
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

    ModelTypes = {
        'sequence': AutoModelForSequenceClassification,
        'token': AutoModelForTokenClassification
    }
    # load model and create optimizer
    model = ModelTypes[args.task].from_pretrained(args.pretrained_ckpt, num_labels=num_labels)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None

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

    config = vars(args)
    config['dataset'] = ds['train'].info.builder_name
    # initialize wandb, project and run name are set
    # as environment variables (see `experiments/run-active.sh`)
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
    # attach confusion matrix metric to tester
    # needed for some active learning metrics
    ConfusionMatrix(
        num_classes=model.config.num_labels,
        output_transform=tester.get_logits_and_labels
    ).attach(tester, "cm")
    
    # create active learning engine
    al_engine = ActiveLearningEngine(
        trainer=trainer,
        trainer_run_kwargs=dict(
            max_epochs=args.epochs,
            epoch_length=args.epoch_length
        ),
        train_batch_size=args.batch_size,
        eval_batch_size=64,
        max_convergence_retries=3,
        train_val_ratio=0.9
    )
    
    @al_engine.on(Events.ITERATION_STARTED)
    def on_started(engine):
        # log active learning step
        i = engine.state.iteration
        print("-" * 8, "AL Step %i" % i, "-" * 8)
        
    @al_engine.on(ActiveLearningEvents.DATA_SAMPLING_COMPLETED)
    def save_selected_samples(engine):
        # create path to save samples in
        path = os.path.join(
            "output",
            os.environ["WANDB_RUN_GROUP"],
            os.environ["WANDB_NAME"],
        )
        os.makedirs(path, exist_ok=True)
        # get selected samples and re-create input texts
        data = engine.state.batch
        texts = tokenizer.batch_decode(
            sequences=[sample['input_ids'] for sample in data],
            skip_special_tokens=True
        )
        # save selected samples to file
        with open(os.path.join(path, "step-%i.txt" % len(engine.train_dataset)), 'w+') as f:
            f.write('\n'.join(texts))

    @al_engine.on(ActiveLearningEvents.CONVERGED | ActiveLearningEvents.CONVERGENCE_RETRY_COMPLETED)
    def on_converged(engine):
        print(
            "Training Converged:", engine.trainer.converged, 
            "Train F-Score: %.03f" % trainer.state.metrics['train/F']
        )

    @al_engine.on(Events.ITERATION_COMPLETED)
    def evaluate_and_log(engine):
        # create validation and test dataloaders
        val_loader = DataLoader(engine.val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(ds['test'], batch_size=64, shuffle=False)
        # run on validation data
        val_metrics = validator.run(val_loader).metrics
        print("Validation Metrics:", val_metrics)

        # run on test data
        test_metrics = tester.run(test_loader).metrics
        print("Test Metrics:", test_metrics)
        # don't log test confusion matrix
        test_metrics = test_metrics.copy()
        test_metrics.pop('cm')

        # get total time spend in strategy
        strategy_time = strategy.state.times[Events.COMPLETED.name]
        strategy_time = {'times/strategy': strategy_time} if strategy_time is not None else {}
        # log all remaining metrics to weights and biases
        wandb.log(
            data=(trainer.state.metrics | val_metrics | test_metrics | strategy_time),
            step=len(engine.train_dataset)
        )
    
        # check if there is an output to visualize
        if strategy.output is not None:
            ax, fig = visualize_embeds(strategy)
            ax.set(title="%s embedding iteration %i (t-SNE)" % (args.strategy, engine.state.iteration))
            wandb.log({"Embedding": wandb.Image(fig)}, step=len(engine.train_dataset))
            # close figure
            plt.close(fig)
    
    # add metrics to active learning engine
    wss = WorkSavedOverSampling(output_transform=lambda _: tester.state.metrics['cm'])
    area = AreaUnderLearningCurve(
        output_transform=lambda _: (
            # point of learning curve given
            # by iteration and accuracy value
            al_engine.state.iteration,
            tester.state.metrics['test/F']
        )
    )
    wss.attach(al_engine, "test/wss")
    area.attach(al_engine, "test/Area(F)")

    # run active learning experiment
    state = al_engine.run(loop, steps=args.steps)
    print("Active Learning Metrics:", state.metrics)
    # log active learning metric scores
    wandb.run.summary.update(state.metrics)

    # run finished
    wandb.finish(quiet=True)
