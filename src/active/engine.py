import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
# import ignite
from ignite.engine import Engine
from ignite.engine.events import State, Events, EventEnum
# others
from .loop import ActiveLoop
from .helpers.engines import Trainer
from typing import Optional

class ActiveLearningEvents(EventEnum):
    """ Costum Events fired by Active Learning Engine.

        This list of events also contains some convergence
        management events. I.e. a model might be trained
        for multiple tries until the trainers convergence
        criteria are met.

        Events:
            DATA_SAMPLING_COMPLETE: 
                called after data is sampled and split into training and validation.
                The `engine.training_data` and `engine.validation_data` are up to date.
            CONVERGENCE_RETRY_STARTED: called on start of model training
            CONVERGENCE_RETRY_COMPLETED: called on completion of a model training try
            CONVERGED: called when trainer convergence criteria is met
            DIVERGED: called when trainer convergence criteria is not met after all retries
    """

    DATA_SAMPLING_COMPLETED = "data-sampling-completed"
    # convergence events
    CONVERGENCE_RETRY_STARTED = "convergence_retry_started"
    CONVERGENCE_RETRY_COMPLETED = "convergence_retry_completed"
    CONVERGED = "converged"
    DIVERGED = "diverged"

class ActiveLearningEngine(Engine):
    """ Active Learning Engine implementing the basic active
        learning procedure of iterating the following steps:
            1. gathering the next samples from the active loop
               provided in the `run` method
            2. splitting samples into training and validation data
            3. running trainer on the full dataset sampled up to this point

        Args:
            trainer (Trainer): model trainer
            trainer_run_kwargs (Optional[dict]): 
                keyword arguments passed to the trainers run method.
                Sets values like `max_epochs` and `epoch_length`.
            train_batch_size (Optional[int]): 
                batch size used for model training.
                Defaults to 32.
            eval_batch_size (Optional[int]): 
                batch size used for model evaluation. 
                Defaults to `train_batch_size`.
            max_convergence_retries (Optional[int]):
                maximum number of model training retries to meet
                the trainers convergence criterion. Defaults to 3.
            train_val_ratio (Optional[float]):
                ratio between training and validation data sizes.
                Defaults to 0.9 meaning 90% of the sampled data is
                used for model training and the remaining 10% is
                used as validation data.
    """

    def __init__(
        self,
        trainer:Trainer,
        trainer_run_kwargs:Optional[dict] ={},
        train_batch_size:Optional[int] =32,
        eval_batch_size:Optional[int] =None,
        max_convergence_retries:Optional[int] =3,
        train_val_ratio:Optional[float] =0.9
    ) -> None:
        # initialize engine
        super(ActiveLearningEngine, self).__init__(type(self).step)

        # register costum events
        self.register_events(*ActiveLearningEvents)

        # save trainer and other arguments
        self.trainer = trainer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size or self.train_batch_size
        self.trainer_run_kwargs = trainer_run_kwargs
        self.max_convergence_retries = max_convergence_retries
        self.train_val_ratio = train_val_ratio

        # active datasets
        self.train_data = []
        self.val_data = []
        # reset datasets on start
        self.add_event_handler(Events.STARTED, type(self)._reset_datasets)

    @property
    def train_dataset(self) -> ConcatDataset:
        """ Training dataset """
        return ConcatDataset(self.train_data)
    
    @property
    def val_dataset(self) -> ConcatDataset:
        """ Validation dataset"""
        return ConcatDataset(self.val_data)

    def _reset_datasets(self):
        """ Event handler to reset sampled datasets on engine start. """
        self.train_data.clear()
        self.val_data.clear()

    def step(self, samples:Dataset):
        """ Engines step function implementing a single step of
            the default active learning procedure 

            Args:
                samples (Dataset): samples selected for the active learning step
        """
        # split into train and validation samples
        train_samples, val_samples = random_split(
            samples, [
                int(len(samples) * self.train_val_ratio),
                len(samples) - int(len(samples) * self.train_val_ratio)
            ]
        )
        # create datasets
        self.train_data.append(train_samples)
        self.val_data.append(val_samples)

        # fire data generated event
        self.fire_event(ActiveLearningEvents.DATA_SAMPLING_COMPLETED)
        
        # create dataloaders and update validation loader in trainer
        train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.trainer.val_loader = DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False)

        for _ in range(self.max_convergence_retries):
            self.fire_event(ActiveLearningEvents.CONVERGENCE_RETRY_STARTED)
            # train model on dataset
            self.trainer.run(train_loader, **self.trainer_run_kwargs)
            # check convergence
            if self.trainer.converged:
                self.fire_event(ActiveLearningEvents.CONVERGED)
                break
            # call event retry completed
            self.fire_event(ActiveLearningEvents.CONVERGENCE_RETRY_COMPLETED)

        else:
            # diverged
            self.fire_event(ActiveLearningEvents.DIVERGED)

    def run(
        self, 
        loop:ActiveLoop, 
        steps:Optional[int] =None,
        seed:Optional[int] =None
    ) -> State:
        return super(ActiveLearningEngine, self).run(
            data=loop,
            max_epochs=1,
            epoch_length=steps,
            seed=seed
        )
