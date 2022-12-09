import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
# import trainer, evaluator and metrics
from active.helpers.trainer import Trainer
from active.helpers.evaluator import Evaluator
from active.helpers.metrics import (
    SeqEvalMetrics, 
    NestedSeqEvalMetrics
)
# import data processors
from active.helpers.processor import (
    BioTaggingProcessor,
    NestedBioTaggingProcessor,
    SequenceClassificationProcessor
)
# import ignite
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Recall, Precision, Average, Fbeta, Accuracy
# hugginface
import datasets
import transformers
# others
from enum import Enum
from typing import Tuple, Dict, Union, Callable, Type
from pydantic import BaseModel, validator, root_validator

class ForwardForNestedTokenClassification(transformers.PreTrainedModel):
    """ Forward container class used by transformer models for nested token classification.
        Has to be combined with a transformer model for token classification.
    """

    def forward(self, *args, **kwargs):
        """ Forward function for nested token classification.
            Based on forward for standard token classification.
        
            Output logits are of shape (B, S, E, 3) where B
            is the batch size, S is the sequence length and
            E is the number of classifiers/entities to predict.
            The final dimension corresponds to BIO labeling scheme.    
        """
        # expects dictionary output
        kwargs['return_dict'] = True
        # pop labels from arguments if given
        # to avoid loss-computation in parent
        labels = kwargs.pop('labels') if 'labels' in kwargs else None
        
        # call forward transformer for token classification and reorganize logits
        out = super(ForwardForNestedTokenClassification, self).forward(*args, **kwargs)
        out['logits'] = out.logits.reshape(out.logits.size(0), out.logits.size(1), -1, 3)

        # compute loss
        if labels is not None:
            out['loss'] = torch.nn.functional.cross_entropy(out.logits.reshape(-1, 3), labels.reshape(-1))
        
        # return output
        return out 
        
class AutoModelForNestedTokenClassification(object):
    
    @staticmethod
    def from_pretrained(pretrained_ckpt, **kwargs):
        # re-interpret num labels as num entities
        # for each entity introduce 3 labels (i.e. B,I,O)
        kwargs['num_labels'] = 3 * kwargs.get('num_labels', 1)

        # get requested model type        
        config = transformers.AutoConfig.from_pretrained(pretrained_ckpt)
        model_type = transformers.AutoModelForTokenClassification._model_mapping.get(type(config), None)
        # create new model type overwriting the forward method
        # for nested bio tagging predictions and loss computation
        model_type_for_nested = type(
            "TransformerForNestedTokenClassification",
            (ForwardForNestedTokenClassification, model_type),
            {}
        )

        return model_type_for_nested.from_pretrained(pretrained_ckpt, **kwargs)

class Task(Enum):
    """ Enum defining supported task types:
            (1) sequence:    marks the experiment as a sequence classification task
            (2) bio-tagging: marks the experiment as a bio-tagging task
    """
    SEQUENCE = "sequence"
    BIO_TAGGING = "bio-tagging"
    NESTED_BIO_TAGGING = "nested-bio-tagging"

    @property
    def model_type(self) -> Type[transformers.PreTrainedModel]:
        if self is Task.SEQUENCE:
            return transformers.AutoModelForSequenceClassification
        elif (self is Task.BIO_TAGGING):
            return transformers.AutoModelForTokenClassification
        elif (self is Task.NESTED_BIO_TAGGING):
            return AutoModelForNestedTokenClassification

        raise ValueError("No model found for task <%s>" % self)
    
    @property
    def processor_type(self) -> Type[Callable]:
        if self is Task.SEQUENCE:
            return SequenceClassificationProcessor
        elif self is Task.BIO_TAGGING:
            return BioTaggingProcessor
        elif self is Task.NESTED_BIO_TAGGING:
            return NestedBioTaggingProcessor
 
        raise ValueError("No data processor found for task <%s>" % self)

class DataConfig(BaseModel):
    """ Data Configuration Model """
    task:Task =None
    # dataset config
    dataset:str
    text_column:str
    label_column:str
    # bio-tag prefixes
    begin_tag_prefix:str =None
    in_tag_prefix:str =None
    # sequence length
    min_sequence_length:int
    max_sequence_length:int

    @root_validator()
    def _check_config(cls, values):
        # get values
        task = values.get('task')
        dataset = values.get('dataset')
        text_col = values.get('text_column')        
        label_col = values.get('label_column')        
        min_seq_len = values.get('min_sequence_length')
        max_seq_len = values.get('max_sequence_length')
        begin_tag_prefix = values.get('begin_tag_prefix')
        in_tag_prefix = values.get('in_tag_prefix')

        # check if dataset is none
        if dataset is None:
            raise ValueError("Dataset set to None")

        try:
            # try to load dataset builder
            builder = datasets.load_dataset_builder(dataset)
        except FileNotFoundError as e:
            # handle invalid/unkown datasets
            raise ValueError("Unkown dataset: %s" % dataset) from e
        
        # get dataset info
        info = builder._info()

        # check if label column is valid
        if label_col not in info.features:
            raise ValueError("Specified label column `%s` is not present in dataset. Valid columns are: %s" % (label_col, ','.join(info.features.keys())))

        # check if text column is valid
        if text_col not in info.features:
            raise ValueError("Specified text column `%s` is not present in dataset. Valid columns are: %s" % (text_col, ','.join(info.features.keys())))

        # make sure sequence length range is set properly
        if min_seq_len > max_seq_len:
            raise ValueError("Minimum sequence length (%i) larger than maximum sequence length (%i)" % (min_seq_len, max_seq_len))

        # make sure the tag prefixes are set for bio tagging tasks
        if (task is not None) and (task is Task.BIO_TAGGING):
            if (begin_tag_prefix is None) or (in_tag_prefix is None):
                raise ValueError("Begin/In tag prefixes must be set for bio tagging tasks!")

        # all done
        return values

    #
    # Helper Functions
    #

    @property
    def dataset_info(self) -> datasets.DatasetInfo:
        builder = datasets.load_dataset_builder(self.dataset)
        return builder._info()

    @property
    def label_space(self) -> Tuple[str]:
        if self.task is Task.SEQUENCE:
            return tuple(self.dataset_info.features[self.label_column].names)
        elif self.task is Task.BIO_TAGGING:
            return tuple(self.dataset_info.features[self.label_column].feature.names)
        elif self.task is Task.NESTED_BIO_TAGGING:
            # for nested bio tagging tasks the label space
            # generated here is actually only the entity types
            # keep in mind that for each unique entity type
            # the Begin, In and Out-tags are introduces
            # the overall label space is three times larger
            # (see `AutoModelForNestedTokenClassification.from_pretrained`)
            return tuple(self.dataset_info.features[self.label_column].keys())

    def load_dataset(
        self, 
        tokenizer:transformers.PreTrainedTokenizer,
        *,
        split:Dict[str, str] ={'train': 'train', 'test': 'test'}, 
        use_cache:bool =False
    ) -> Dict[str, datasets.arrow_dataset.Dataset]:
        # load dataset 
        ds = datasets.load_dataset(self.dataset, split=split)
        
        # create processor
        processor = self.task.processor_type(
            tokenizer=tokenizer,
            max_length=self.max_sequence_length,
            text_column=self.text_column,
            label_column=self.label_column,
            label_space=self.label_space,
            begin_tag_prefix=self.begin_tag_prefix,
            in_tag_prefix=self.in_tag_prefix         
        )
        # create filter function
        pad_token_id = tokenizer.pad_token_id
        filter_ = lambda e: (np.asarray(e['input_ids']) != pad_token_id).sum() > self.min_sequence_length

        # preprocess datasets
        ds = {
            key: dataset \
                .map(processor, batched=False, desc=key, load_from_cache_file=use_cache) \
                .filter(filter_, batched=False, desc=key, load_from_cache_file=use_cache)
            for key, dataset in ds.items()
        }

        # set data formats
        for dataset in ds.values():
            dataset.set_format(
                type='torch',
                columns=['input_ids', 'attention_mask', 'labels']
            )
        
        # return preprocessed datasets
        return ds
    
    def attach_metrics(self, engine:Engine, tag:str) -> None:
        # create token-level metrics
        L = Average(output_transform=lambda out: out['loss'])
        A = Accuracy(output_transform=type(engine).get_logits_and_labels)
        R = Recall(output_transform=type(engine).get_logits_and_labels, average=True)
        P = Precision(output_transform=type(engine).get_logits_and_labels, average=True)
        F = Fbeta(beta=1.0, output_transform=type(engine).get_logits_and_labels, average=True)
        
        if self.task is Task.SEQUENCE:
            # attach metrics
            L.attach(engine, '%s/L' % tag)
            A.attach(engine, '%s/A' % tag)
            R.attach(engine, '%s/R' % tag)
            P.attach(engine, '%s/P' % tag)
            F.attach(engine, '%s/F' % tag)

        elif self.task in (Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING):
            # attach token-level metrics
            L.attach(engine, '%s/token/L' % tag)
            A.attach(engine, '%s/token/A' % tag)
            R.attach(engine, '%s/token/R' % tag)
            P.attach(engine, '%s/token/P' % tag)
            F.attach(engine, '%s/token/F' % tag)
            
            # create and attach sequence level metric
            if self.task is Task.BIO_TAGGING:
                SeqEvalMetrics(
                    label_space=self.label_space, 
                    output_transform=type(engine).get_logits_labels_mask
                ).attach(engine, '%s/entity' % tag)

            elif self.task is Task.NESTED_BIO_TAGGING:
                NestedSeqEvalMetrics(
                    # note that here the label space is the entity types
                    entity_types=self.label_space,
                    output_transform=type(engine).get_logits_labels_mask
                ).attach(engine, '%s/entity' % tag)


class ModelConfig(BaseModel):
    """ Model Configuration Model """
    task:Task = None
    # pretrained model checkpoint
    pretrained_ckpt:str
    
    @validator('pretrained_ckpt')
    def _check_pretrained_ckpt(cls, value):
        try:
            # check if model is valid by loading config
            transformers.AutoConfig.from_pretrained(value)
        except OSError as e:
            # handle model invalid
            raise ValueError("Unkown pretrained checkpoint: %s" % value) from e

        return value

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return transformers.AutoTokenizer.from_pretrained(self.pretrained_ckpt, use_fast=True, add_prefix_space=True)

    def load_model(self, label_space:Tuple[str]) -> transformers.PreTrainedModel:
        return self.task.model_type.from_pretrained(
            self.pretrained_ckpt, 
            num_labels=len(label_space)
        )

class TrainerConfig(BaseModel):
    """ Trainer Configuration Model """
    incremental:bool
    # optimizer setup
    lr:float
    weight_decay:float
    # training setup
    batch_size:int
    max_epochs:Union[int, None]
    epoch_length:Union[int, None]
    min_epoch_length:Union[int, None]
    # convergence criteria
    early_stopping_patience:int
    accuracy_threshold:float

    def build_trainer(self, model:transformers.PreTrainedModel):
        # create optimizer
        optim=torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # create trainer
        return Trainer(
            model=model,
            optim=optim,
            scheduler=None,
            acc_threshold=self.accuracy_threshold,
            patience=self.early_stopping_patience,
            incremental=self.incremental
        )

    @property
    def run_kwargs(self) -> Dict[str, int]:
        return self.dict(include={'max_epochs', 'epoch_length', 'min_epoch_length'})

class ExperimentConfig(BaseModel):
    """ Experiment Configuration Model
        Includes all of the above configurations as sub-models
    """
    # experiment name used for referencing
    # and task type
    name:str
    task:Task
    # data, model and trainer config
    data:DataConfig
    model:ModelConfig
    trainer:TrainerConfig

    @validator('data', 'model', pre=True)
    def _pass_task_to_sub_configs(cls, v, values):
        assert 'task' in values
        if isinstance(v, BaseModel):
            return v.copy(update={'task': values.get('task')})
        elif isinstance(v, dict):
            return v | {'task': values.get('task')}

    def load_dataset(self, **kwargs) -> Dict[str, datasets.arrow_dataset.Dataset]:
        return self.data.load_dataset(self.model.tokenizer, **kwargs)

    def load_model(self) -> transformers.PreTrainedModel:
        return self.model.load_model(self.data.label_space)

    def build_trainer(self) -> Trainer:
        return self.trainer.build_trainer(self.load_model())


def train(config:str, seed:int, use_cache:bool, disable_tqdm:bool =False):
    import wandb
   
    config = ExperimentConfig.parse_file(config)
    print("Config:", config.json(indent=2))

    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # load datasets and build trainer
    ds = config.load_dataset(use_cache=use_cache)
    trainer = config.build_trainer()
    
    # split validation dataset from train set
    train_data, test_data = ds['train'], ds['test']
    train_data, val_data = random_split(train_data, [
        int(0.8 * len(train_data)),
        len(train_data) - int(0.8 * len(train_data))
    ])

    # create dataloaders
    val_loader = DataLoader(val_data, batch_size=config.trainer.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.trainer.batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=config.trainer.batch_size, shuffle=True)
 
    # set up wandb 
    wandb_config = config.dict(exclude={"active"})
    wandb_config['data']['dataset'] = ds['train'].info.builder_name
    wandb_config['seed'] = seed
    wandb.init(
        config=wandb_config,
        project=os.environ.get("WANDB_PROJECT", "train-final"),
        group=config.name
    )

    # create the validater and tester
    validator = Evaluator(trainer.unwrapped_model)
    tester = Evaluator(trainer.unwrapped_model)
    # attach metrics
    config.data.attach_metrics(trainer, tag="train")
    config.data.attach_metrics(validator, tag="val")
    config.data.attach_metrics(tester, tag="test")
    # attach progress bar
    if not disable_tqdm:
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
        if config.task is Task.SEQUENCE:
            print("Train F-Score:", engine.state.metrics['train/F'])
            print("Val   F-Score:", val_state.metrics['val/F'])
            print("Test  F-Score:", test_state.metrics['test/F'])
        elif config.task in (Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING):
            print("Train Entity F-Score:", engine.state.metrics['train/entity/weighted avg/F'])
            print("Val   Entity F-Score:", val_state.metrics['val/entity/weighted avg/F'])
            print("Test  Entity F-Score:", test_state.metrics['test/entity/weighted avg/F'])
        # log all metrics to weights and biases 
        wandb.log(
            test_state.metrics | val_state.metrics | engine.state.metrics, 
            step=engine.state.iteration
        )

    # run training
    return trainer.run(train_loader, **config.trainer.run_kwargs)

def main():
    from argparse import ArgumentParser
    # build argument parser
    parser = ArgumentParser(description="Train transformer model on sequence or token classification tasks.")
    parser.add_argument("--config", type=str, required=True, help="Path to a valid experiment configuration")
    parser.add_argument("--use-cache", action='store_true', help="Load cached preprocessed datasets if available")
    parser.add_argument("--seed", type=int, default=1337, help="Random Seed")
    # parse arguments
    args = parser.parse_args()
    
    # train model
    train(**vars(args))

if __name__ == '__main__':
    main() 
