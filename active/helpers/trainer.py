import io
# import torch
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.utils.data import DataLoader
# import ignite
import ignite.distributed as idist
from ignite.engine import Engine, State, Events
from ignite.metrics import Average, Accuracy
from ignite.handlers import EarlyStopping, Checkpoint
from ignite.handlers.checkpoint import BaseSaveHandler
# transformers
from transformers import PreTrainedModel
# others
from ..core.utils.model import get_encoder_from_model
from ..core.utils.data import map_tensors, move_to_device
from .evaluator import Evaluator
from .schedulers import _TrainingStepsDependentMixin
from copy import deepcopy
from collections import OrderedDict
from typing import Optional, Iterable

class MemorySaveHandler(BaseSaveHandler, OrderedDict):
    
    def __init__(self, max_length:int =1) -> None:
        super(MemorySaveHandler, self).__init__()
        self.max_length = max_length

    def __call__(self, checkpoint, filename, metadata=None) -> None:
        # store checkpoint in bytes buffer instead of copying
        # the checkpoint to avoid copies on gpu memory
        buf = io.BytesIO()
        torch.save(checkpoint, buf)
        # save copy of checkpoint in memory dict
        self[filename] = buf
        # remove very last element if memory exceeds size
        if len(self) > self.max_length:
            self.popitem(last=False)

    def remove(self, filename):
        # remove object from memory
        self.pop(filename)

    def load(self, filename):
        # get buffer
        buf = dict.__getitem__(self, filename)
        buf.seek(0)
        # load buffer
        return torch.load(buf)


class Trainer(Evaluator):
    """ Engine to train a transformer model. Can be used for both active and passive learning loops.

        Args:
            model (PreTrainedModel): pre-trained transformer model to train
            optim (Optimizer): optimizer to use for parameter updates
            scheduler (Optional[_LRScheduler]): 
                learning rate scheduler. Defaults to LambdaLR(optim, lambda _: 1.0).
            acc_theshold (Optional[float]): 
                accuracy threshold, training is stopped if threshold is 
                surpassed on training data (Default: 1.0)
            patience (Optional[int]):
                early stopping patience, training is stopped if no improvement
                in accuracy on the validation data is achived for `patient` epochs.
                If set to `None`, no early stopping is applied (Default: None).
            incremental (Optional[bool]):
                whether to reuse the model trained in previous calls to `run` or
                to reset it to it's inital state. This also resets the optimizer
                and learning rate scheduler states. If set to false only the scheduler
                will be resetted. (Default: False)
            val_loader (Optional[DataLoader]):
                validation data loader. (Default: Train data loader)
    """

    def __init__(
        self, 
        model:PreTrainedModel, 
        optim:Optimizer, 
        scheduler:Optional[_LRScheduler] =None,
        acc_threshold:Optional[float] =1.0, 
        patience:Optional[int] =None,
        incremental:Optional[bool] =False,
        val_loader:DataLoader =None
    ) -> None:
        # save arguments
        self.acc_threshold = acc_threshold
        self.incremental = incremental
        
        # get model encoder
        self.encoder = get_encoder_from_model(model)
        # save optimizer and initialize evaluator
        self.optim = idist.auto_optim(optim)
        self.scheduler = scheduler or LambdaLR(optim, lambda _: 1.0)
        super(Trainer, self).__init__(model)
        # save validation dataloader
        self.val_loader = val_loader

        # save initial checkpoint
        self.init_encoder_ckpt = deepcopy(self.encoder.state_dict())
        self.init_optim_ckpt = deepcopy(self.optim.state_dict())
        self.init_scheduler_ckpt = deepcopy(self.scheduler.state_dict())
        # add event handler        
        self.add_event_handler(Events.STARTED, type(self)._load_init_ckpt)
        self.add_event_handler(Events.STARTED, type(self)._prepare_scheduler)

        # create train accuracy metric
        self.train_acc = Accuracy(output_transform=type(self).get_logits_and_labels)
        # and manually attach handlers to avoid writing accuracy in state
        self.add_event_handler(Events.EPOCH_STARTED, self.train_acc.started)
        self.add_event_handler(Events.ITERATION_COMPLETED, self.train_acc.iteration_completed)
        # add convergence event handler
        self.add_event_handler(Events.EPOCH_COMPLETED, type(self)._check_convergence)

        # create validation evaluator
        self.val_evaluator = Evaluator(model)
        Average(output_transform=lambda out: out['loss']).attach(self.val_evaluator, 'L')
        # validate after each epoch
        self.add_event_handler(Events.EPOCH_COMPLETED, type(self)._validate)

        # attach checkpoint handler saving best model parameters
        # w.r.t. validation loss
        self.ckpt = Checkpoint(
            to_save={
                'model': self.model,
                # also save evaluator to capture metrics corresponding to model
                'val-evaluator': self.val_evaluator
            },
            save_handler=MemorySaveHandler(),
            n_saved=1,
            score_function=lambda e: -e.state.metrics['L'],
        )
        self.val_evaluator.add_event_handler(Events.COMPLETED, self.ckpt)
        # add checkpoint event handlers
        self.add_event_handler(Events.STARTED, type(self)._reset_ckpt)
        self.add_event_handler(Events.COMPLETED, type(self)._load_best_ckpt)

        # attach to evaluator
        if patience is not None:
            # early stopping
            self.stopper = EarlyStopping(
                patience=patience,
                score_function=lambda e: -e.state.metrics['L'],
                trainer=self
            )
            # add event handlers
            self.add_event_handler(Events.STARTED, type(self)._reset_stopper)
            self.val_evaluator.add_event_handler(Events.COMPLETED, self.stopper)

    @property
    def converged(self) -> bool:
        """ Whether the training process converged """
        # convergence is met if engine terminated before maximum number of epochs
        return self.state.epoch < self.state.max_epochs

    @property
    def train_accuracy(self) -> float:
        """ Final accuracy on training dataset """
        return self.train_acc.compute()
    
    def step(self, batch):
        """ Training step function executed by the engine. """
        # move to device
        batch = move_to_device(batch, device=idist.device())
        # forward through model and compute loss
        self.model.train()
        self.optim.zero_grad()
        out = self.model.forward(**batch)
        # compute average loss
        if out['loss'].ndim >= 0:
            out['loss'] = out['loss'].mean()
        # optimizer parameters
        out['loss'].backward()
        self.optim.step()
        self.scheduler.step()
        # pass labels through
        out['labels'] = batch['labels']
        # detach and move to cpu
        return map_tensors(out, fn=lambda t: t.detach().cpu())

    def run(
        self, 
        data: Optional[Iterable] =None,
        max_epochs: Optional[int] =None,
        epoch_length: Optional[int] =None,
        min_epoch_length: Optional[int] =None
    ) -> State:
        """ Runs the trainer over the passed data.

            Overwrites the typical `ignite.Engine` behavior of continuing a
            previous run by resetting the state before starting the engine run.

            Args:
                data (Optional[Iterable]):
                    data (usually a `DataLoader`) to train the model on
                max_epochs (Optional[int]): 
                    maximum number of training epochs
                epoch_length (Optional[int]): 
                    length of a single training epoch. If provided, the epoch length
                    induced by the length of the dataloader is overwritten.
                min_epoch_length(Optional[int]): 
                    minimum length of a single training epoch. Overwrites the induced
                    epoch length only if it is too small. This argument is especially
                    useful in active learning scenarios where the first few training 
                    steps have only limited data and thus less update steps. Setting
                    `min_epoch_length` ensures a constant minumum number of updates.
            Returns:
                state (State): engine state after the run
        """
        # usually trainer tries to resume run if state holds unfinished runs
        # due to early stopping or other convergence criteria
        # to avoid that the state's progress is reset at the start of the run
        self.state = State()
        # get epoch length to use
        if min_epoch_length is not None:
            epoch_length = epoch_length or len(data)
            epoch_length = epoch_length if epoch_length >= min_epoch_length else min_epoch_length
        # run trainer
        return super(Trainer, self).run(
            data=data,
            max_epochs=max_epochs,
            epoch_length=epoch_length
        )

    def _load_init_ckpt(self, force:bool =False):
        """ Event handler to load the initial checkpoints. Called on `STARTED`. 

            Args:
                force (Optional[bool]): 
                    whether to force reset the model and optimizer even if
                    the incremental attribute is set to True.
        """
        if (not self.incremental) or force:
            # reset model by first re-initializing all weights
            # and loading the pretrained encoder afterwards
            self.unwrapped_model.init_weights()
            self.encoder.load_state_dict(self.init_encoder_ckpt)
            # reset optimizer state
            self.optim.load_state_dict(self.init_optim_ckpt)
        # always reset scheduler state
        self.scheduler.load_state_dict(self.init_scheduler_ckpt)
        # just making sure ;)
        self.optim.zero_grad(set_to_none=True)

    def _prepare_scheduler(self):
        """ Event handler to prepare the scheduler. Called on `STARTED`. """
        # update scheduler train steps
        if isinstance(self.scheduler, _TrainingStepsDependentMixin):
            self.scheduler.set_num_training_steps(
                steps=len(self.state.dataloader) * self.state.max_epochs
            )

    def _check_convergence(self):
        """ Event handler stopping training when training accuracy 
            threshold is surpassed. Called on 'EPOCH_COMPLTED'. 
        """
        # evaluate model on train set and check accuracy
        if (self.train_accuracy >= self.acc_threshold):
            self.terminate()

    def _validate(self):
        """ Event handler validating the model on the validation data loader.
            Falls back to the training dataloader if validation loader is not set.
            Called on 'EPOCH_COMPLETED'.
        """
        # use validation loader if given and fallback to train data loader
        self.val_evaluator.run(self.val_loader or self.state.dataloader)

    def _load_best_ckpt(self):
        """ Event handler loading the best checkpoint after training finished.
            Called on 'COMPLETED'.
        """
        # check if there is a checkpoint to load
        if self.ckpt.last_checkpoint is not None:
            # load checkpoint
            self.ckpt.load_objects(
                to_load={
                    'model': self.model,
                    'val-evaluator': self.val_evaluator
                },
                checkpoint=self.ckpt.save_handler.load(self.ckpt.last_checkpoint)
            )

    def _reset_ckpt(self):
        """ Event handler to reset the checkpoint. Called on `STARTED`. """
        self.ckpt.reset()

    def _reset_stopper(self):
        """ Event handler to reset the early stopper state. Callend on `STARTED`. """
        self.stopper.best_score = None
        self.stopper.counter = 0
