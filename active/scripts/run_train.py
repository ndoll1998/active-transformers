import os
import torch
import numpy as np
from torch.utils.data import random_split
# import data processors
from active.utils.data import NamedTensorDataset
from active.scripts.utils.processor import (
    MinSequenceLengthFilter,
    BioTaggingProcessor,
    NestedBioTaggingProcessor,
    SequenceClassificationProcessor
)
# import metrics
from active.scripts.utils.metrics import (
    PRFS,
    SeqEval,
    NestedSeqEval
)

from active.scripts.utils.modeling import AutoModelForNestedTokenClassification
from active.scripts.utils.trainer import CustomTrainer
# hugginface
import datasets
import transformers
# others
import pydantic
import dataclasses
from enum import Enum
from datetime import datetime
from typing import Tuple, List, Dict, Union, Callable, Type, Optional

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

class DataConfig(pydantic.BaseModel):
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

    @pydantic.root_validator()
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
        use_cache:bool =False,
        max_data_points:int =25_000
    ) -> Dict[str, datasets.arrow_dataset.Dataset]:
        # load dataset 
        ds = datasets.load_dataset(self.dataset, split=split)

        # cut datasets if they are too large
        for key, data in ds.items():
            if len(data) > max_data_points:
                rand_idx = np.random.choice(len(data), max_data_points, replace=False)
                ds[key] = data.select(rand_idx)
            assert len(ds[key]) <= max_data_points

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
        filter_ = MinSequenceLengthFilter(
            tokenizer=tokenizer,
            min_seq_length=self.min_sequence_length
        )

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

class ModelConfig(pydantic.BaseModel):
    """ Model Configuration Model """
    task:Task = None
    # pretrained model checkpoint
    pretrained_ckpt:str

    @pydantic.validator('pretrained_ckpt')
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

@pydantic.dataclasses.dataclass
@dataclasses.dataclass
class TrainerConfig(transformers.TrainingArguments):
    """ Trainer Configuration """
    # passed from experiment config and needed for output directory
    name:str =None
    # create default for output directory
    run_name:str ="{name}-{timestamp}"
    output_dir:str ="output/{name}-{timestamp}"
    overwrite_output_dir:bool =True
    # early stopping setup
    early_stopping_patience:Optional[int] =1
    early_stopping_threshold:Optional[float] =0.0
    # 
    load_best_model_at_end:bool =True
    metric_for_best_model:str ='eval_loss'
    greater_is_better:bool =False
    # minimum steps between evaluations
    # overwrites epoch-based evaluation behavior
    min_epoch_length:Optional[int] =15
    # overwrite some default values
    do_train:bool =True
    do_eval:bool =True
    evaluation_strategy:transformers.trainer_utils.IntervalStrategy ="epoch"
    save_strategy:transformers.trainer_utils.IntervalStrategy ="epoch"
    eval_accumulation_steps:Optional[int] =1
    save_total_limit:Optional[int] =3
    label_names:List[str] =dataclasses.field(default_factory=lambda: ['labels'])
    report_to:Optional[List[str]] =dataclasses.field(default_factory=list)
    log_level:Optional[str] ='warning'
    # fields with incomplete types in Training Arguments
    # set type to avoid error in pydantic validation
    debug:Union[str, List[transformers.debug_utils.DebugOption]]              =""
    sharded_ddp:Union[str, List[transformers.trainer_utils.ShardedDDPOption]] =""
    fsdp:Union[str, List[transformers.trainer_utils.FSDPOption]]              =""
    fsdp_config:Union[None, str, dict]                                        =None

    @pydantic.root_validator()
    def _format_output_directory(cls, values):
        # get timestamp
        timestamp=datetime.now().isoformat()
        # format all values depending on output directory
        return values | {
            'output_dir': values.get('output_dir').format(
                name=values.get('name'),
                timestamp=datetime.now().isoformat()
            ),
            'logging_dir': values.get('logging_dir').format(
                name=values.get('name'),
                timestamp=datetime.now().isoformat()
            ),
            'run_name': values.get('run_name').format(
                name=values.get('name'),
                timestamp=datetime.now().isoformat()
            ),
        }

class ExperimentConfig(pydantic.BaseModel):
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

    @pydantic.validator('data', 'model', pre=True)
    def _pass_task_to_sub_configs(cls, v, values):
        assert 'task' in values
        if isinstance(v, pydantic.BaseModel):
            return v.copy(update={'task': values.get('task')})
        elif isinstance(v, dict):
            return v | {'task': values.get('task')}

    @pydantic.validator('trainer', pre=True)
    def _pass_name_to_trainer_config(cls, v, values):
        assert 'name' in values
        if isinstance(v, pydantic.BaseModel):
            return v.copy(update={'name': values.get('name')})
        elif isinstance(v, dict):
            return v | {'name': values.get('name')}

    def load_dataset(self, **kwargs) -> Dict[str, datasets.arrow_dataset.Dataset]:
        return self.data.load_dataset(self.model.tokenizer, **kwargs)

    def load_model(self) -> transformers.PreTrainedModel:
        return self.model.load_model(self.data.label_space)

    def build_trainer(self) -> transformers.Trainer:

        # create the metric class for the task
        if self.task is Task.SEQUENCE:
            metrics = PRFS(label_space=self.data.label_space)
        elif self.task is Task.BIO_TAGGING:
            metrics = SeqEval(label_space=self.data.label_space)
        elif self.task is Task.NESTED_BIO_TAGGING:
            metrics = NestedSeqEval(entity_types=self.data.label_space)

        # create trainer instance
        #return transformers.Trainer(
        return CustomTrainer(
            model=self.load_model(),
            args=self.trainer,
            # datasets are set manually later on
            train_dataset=None,
            eval_dataset=None,
            # add early stopping callback
            callbacks=[
                transformers.EarlyStoppingCallback(
                    early_stopping_patience=self.trainer.early_stopping_patience,
                    early_stopping_threshold=self.trainer.early_stopping_threshold
                )
            ],
            # get predicted labels from logits as expected by metrics
            preprocess_logits_for_metrics=lambda logits, _: logits.argmax(dim=-1),
            compute_metrics=metrics
        )

def train(config:str, seed:int, use_cache:bool, disable_tqdm:bool =False):

    # parse config
    config = ExperimentConfig.parse_file(config)
    # overwrite a few values
    config.trainer.seed = seed
    config.trainer.data_seed = seed
    config.trainer.disable_tqdm |= disable_tqdm
    # report to weights and biases
    config.trainer.report_to = ["wandb"]
    # print configuration
    print("Config:", config.json(indent=2))

    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load datasets and build trainer
    ds = config.load_dataset(use_cache=use_cache)
    # split validation dataset from train set
    train_data, test_data = ds['train'], ds['test']
    train_data, val_data = random_split(train_data, [
        int(0.8 * len(train_data)),
        len(train_data) - int(0.8 * len(train_data))
    ])
    # convert all datasets to tensor datasets
    train_data = NamedTensorDataset.from_dataset(train_data)
    val_data = NamedTensorDataset.from_dataset(val_data)
    test_data = NamedTensorDataset.from_dataset(test_data)

    # count total number of tokens in train split
    n_tokens = sum((item['attention_mask'].sum().item() for item in train_data))
    print("#Train Tokens:", n_tokens)

    # create trainer and set datasets
    trainer = config.build_trainer()
    trainer.train_dataset = train_data
    trainer.eval_dataset = val_data

    # start training
    trainer.train()

    # test trained model
    # trainer loads best model at end
    metrics = trainer.evaluate(test_data, metric_key_prefix="test")
    print("-" * 15 + "Final Test Scores" + "-" * 15)
    print("Test Micro F1:    %.4f" % metrics['test_micro-avg_F'])
    print("Test Macro F1:    %.4f" % metrics['test_macro-avg_F'])
    print("Test Weighted F1: %.4f" % metrics['test_weighted-avg_F'])

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
