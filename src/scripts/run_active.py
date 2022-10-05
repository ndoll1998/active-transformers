import os
import torch
import numpy as np
from torch.utils.data import DataLoader
# datasets and transformers
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
# import active learning components
from src.active.loop import ActiveLoop
from src.active.strategies import *
from src.active.engine import ActiveLearningEvents, ActiveLearningEngine
from src.active.metrics import AreaUnderLearningCurve, WorkSavedOverSampling
from src.active.helpers.engines import Trainer, Evaluator
from src.active.utils.model import get_encoder_from_model
# import ignite
from ignite.engine import Events
from ignite.metrics import ConfusionMatrix
from ignite.contrib.handlers.tqdm_logger import ProgressBar
# dimensionality reduction
from sklearn.manifold import TSNE
# others
import wandb
from matplotlib import pyplot as plt
# helper functions
from src.scripts.run_train import (
    add_data_args,
    add_model_and_training_args,
    load_and_preprocess_datasets,
    create_model_optim_scheduler,
    attach_metrics
)


#
# Argument Parsing
#

def add_active_learning_args(parser, group_name="Active Learning Arguments"):
    
    group = parser.add_argument_group(group_name)
    # specify strategy and active learning params
    group.add_argument("--strategy", type=str, default="random", help="Active Learning Strategy to use")
    group.add_argument("--query-size", type=int, default=25, help="Number of data points to query from pool at each AL step")
    group.add_argument("--steps", type=int, default=-1, help="Number of Active Learning Steps. Defaults to -1 meaning the whole dataset will be processed.")
    # return argument group
    return group


#
# Build Strategy and AL-Engine
#

def build_strategy(args, model):
    if args.strategy == 'random': return Random()
    elif args.strategy == 'least-confidence': return LeastConfidence(model)
    elif args.strategy == 'prediction-entropy': return PredictionEntropy(model)
    elif args.strategy == 'badge' and args.task == 'sequence':
        return BadgeForSequenceClassification(get_encoder_from_model(model), model.classifier)
    elif args.strategy == 'badge' and args.task == 'token':
        return BadgeForTokenClassification(get_encoder_from_model(model), model.classifier)
    elif args.strategy == 'alps': return AlpsConstantEmbeddings(model, mlm_prob=0.15)
    elif args.strategy == 'egl': return EglByTopK(model, k=5)
    elif args.strategy == 'egl-sampling': return EglBySampling(model, k=8)
    elif args.strategy == 'entropy-over-max': return EntropyOverMax(model)
    elif args.strategy == 'entropy-over-max-ignore': return EntropyOverMax(model, ignore_labels=[0])
    elif args.strategy == 'entropy-over-max-sample': return EntropyOverMax(model, random_sample=True)

def build_engine_and_loop(args, ds):

    # create model, optimizer and scheduler
    model, optim, scheduler = create_model_optim_scheduler(args, ds)

    # create strategy and attach progress bar to strategy
    strategy = build_strategy(args, model)

    # create active learning loop
    loop = ActiveLoop(
        pool=ds['train'],
        strategy=strategy,
        batch_size=64,
        query_size=args.query_size,
        init_strategy=strategy if isinstance(strategy, Alps) else Random()
    )

    # create active learning engine
    al_engine = ActiveLearningEngine(
        trainer=Trainer(
            model=model,
            optim=optim,
            scheduler=scheduler,
            acc_threshold=args.acc_threshold,
            patience=args.patience,
            incremental=True
        ),
        trainer_run_kwargs=dict(
            max_epochs=args.epochs,
            epoch_length=args.epoch_length
        ),
        train_batch_size=args.batch_size,
        eval_batch_size=64,
        max_convergence_retries=3,
        train_val_ratio=0.9
    )

    # return engine and loop
    return al_engine, loop

#
# Visualization
#

def visualize_embeds(strategy):
    # get selected indices and processing output of unlabeled pool
    idx = loop.strategy.selected_indices
    output = loop.strategy.output
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
    add_data_args(parser)
    add_model_and_training_args(parser)    
    add_active_learning_args(parser)
    # parse arguments
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # load and preprocess datasets
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_ckpt)
    ds = load_and_preprocess_datasets(args, tokenizer=tokenizer)

    config = vars(args)
    config['dataset'] = ds['train'].info.builder_name
    # initialize wandb, project and run name are set
    # as environment variables (see `experiments/run-active.sh`)
    wandb.init(config=config)
    
    # build active learning engine
    al_engine, loop = build_engine_and_loop(args, ds)
    
    # attach progress bar to strategy
    ProgressBar(ascii=True, desc='Strategy').attach(loop.strategy)
    
    # get trainer and create validater and tester
    trainer = al_engine.trainer
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
    # attach confusion matrix metric to tester
    # needed for some active learning metrics
    ConfusionMatrix(
        num_classes=trainer.unwrapped_model.config.num_labels,
        output_transform=tester.get_logits_and_labels
    ).attach(tester, "cm")

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
        strategy_time = loop.strategy.state.times[Events.COMPLETED.name]
        strategy_time = {'times/strategy': strategy_time} if strategy_time is not None else {}
        # log all remaining metrics to weights and biases
        wandb.log(
            data=(trainer.state.metrics | val_metrics | test_metrics | strategy_time),
            step=len(engine.train_dataset)
        )
    
        # check if there is an output to visualize
        if loop.strategy.output is not None:
            ax, fig = visualize_embeds(loop.strategy)
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
