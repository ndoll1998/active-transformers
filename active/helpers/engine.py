import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
# import ignite
from ignite.engine import Engine
from ignite.engine.events import State, Events, EventEnum
# others
from active.core.loop import ActiveLoop
from .trainer import Trainer
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
    """
    DATA_SAMPLING_COMPLETED = "data-sampling-completed"

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
        self.train_val_ratio = train_val_ratio

        # active datasets
        self.train_data = []
        self.val_data = []

        # reset engine state and datasets on start
        self.add_event_handler(Events.STARTED, type(self)._reset)

    @property
    def num_train_samples(self) -> int:
        return sum(map(len, self.train_data))

    @property
    def num_val_samples(self) -> int:
        return sum(map(len, self.val_data))

    @property
    def num_total_samples(self) -> int:
        return self.num_train_samples + self.num_val_samples

    @property
    def train_dataset(self) -> ConcatDataset:
        """ Training dataset """
        return ConcatDataset(self.train_data)
    
    @property
    def val_dataset(self) -> ConcatDataset:
        """ Validation dataset. Fallback to train dataset if no validation
            data is present yet.
        """
        if self.num_val_samples > 0:
            return ConcatDataset(self.val_data)
        else:
            return self.train_dataset

    def dataloader(self, data:Dataset, **kwargs) -> DataLoader:
        """ Create the dataloader for a given dataset with some specific configuration.            

            Args:
                data (Dataset): dataset to use
                **kwargs (Any): keyword arguments passed to the dataloader

            Returns:
                loader (DataLoader): dataloader from given dataset and configuration
        """
        return DataLoader(data, **kwargs)

    def _reset(self):
        """ Event handler to reset the engine, i.e. re-initialize the model,
            reset optimizer and scheduler states and clear the sampled datasets. 
            Called on `STARTED`.
        """
        # re-initialize model and reset optimizer and scheduler states
        self.trainer._load_init_ckpt(force=True)
        # clear datasets
        self.train_data.clear()
        self.val_data.clear()

    def _reset_state(self):
        """ Reset the internal state of the engine. Called at the very
            beginning of the `run` method.
        """
        self.state = State()

    def step(self, samples:Dataset):
        """ Engines step function implementing a single step of
            the default active learning procedure 

            Args:
                samples (Dataset): samples selected for the active learning step
        """
        # make sure samples are valid
        assert len(samples) > 0, "No samples provided!"
        # compute data split sizes satifying the train-validation ratio as closely as possible
        n_train = round((self.num_total_samples + len(samples)) * self.train_val_ratio) - self.num_train_samples
        n_val = len(samples) - n_train
        # split into train and validation samples
        train_samples, val_samples = random_split(samples, [n_train, n_val])
        
        # add to the respective datasets
        self.train_data.append(train_samples)
        self.val_data.append(val_samples)

        # fire data generated event
        self.fire_event(ActiveLearningEvents.DATA_SAMPLING_COMPLETED)
        
        # create dataloaders and update validation loader in trainer
        train_loader = self.dataloader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.trainer.val_loader = self.dataloader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False)
        # run training
        self.trainer.run(train_loader, **self.trainer_run_kwargs)

    def run(
        self, 
        loop:ActiveLoop, 
        steps:Optional[int] =None,
        seed:Optional[int] =None
    ) -> State:
        # fully reset state
        self._reset_state()
        # run
        return super(ActiveLearningEngine, self).run(
            data=loop,
            max_epochs=1,
            epoch_length=steps,
            seed=seed
        )
