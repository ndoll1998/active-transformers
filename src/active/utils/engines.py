# import torch
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
# import ignite
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics import Average, Accuracy
from ignite.handlers import EarlyStopping, Checkpoint, DiskSaver
# transformers
from transformers import PreTrainedModel
# others
from .schedulers import _TrainingStepsDependentMixin
from copy import deepcopy
from typing import Optional

class Evaluator(Engine):
    """ Engine evaluating a transformer model on a given dataset.

        Args:
            model (PreTrainedModel): pre-trained transformer model to evaluate
    """

    def __init__(
        self,
        model:PreTrainedModel,
    ) -> None:
        # save model and initialize engine
        self.model = idist.auto_model(model)
        super(Evaluator, self).__init__(type(self).step)

    @torch.no_grad()
    def step(self, batch):
        """ Evaluation step function """
        # move to device
        batch = {key: val.to(idist.device()) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
        # forward through model
        self.model.eval()
        out = self.model.forward(**batch)
        # detach and move to cpu
        out = {key: val.detach().cpu() if isinstance(val, torch.Tensor) else val for key, val in out.items()}
        out['labels'] = batch['labels'].cpu()
        # return all outputs for logging and metrics computation
        return out

    @staticmethod
    def get_logits_and_labels(out:dict):
        """ Function to extract the logits and labels from the model output.
            Can be used as `output_transform` in ignite metrics.
        
            Args:
                out (dict): output of a forward call to the `PreTrainedModel`

            Returns:
                logits (torch.Tensor): logits tensor only containing valid entries
                labels (torch.Tensor): labels tensor only containing valid entries
        """
        logits, labels = out['logits'], out['labels']
        mask = (labels >= 0)
        return logits[mask], labels[mask]

    @staticmethod
    def get_loss(out:dict):
        """ Function to extract the loss from the model output. 
            
            Args:
                out (dict): output of a forward call to the `PreTrainedModel`

            Returns:
                loss (torch.Tensor): singleton tensor containing the loss value
        """
        return out['loss'].mean()

class Trainer(Evaluator):
    """ Engine to train a transformer model. Can be used for both active and passive learning loops.

        Args:
            model (PreTrainedModel): pre-trained transformer model to train
            optim (Optimizer): optimizer to use for parameter updates
            scheduler (_LRScheduler): learning rate scheduler
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
            cache_dir (Optional[str]):
                cache directory to store current best model parameters.
                (Default: /tmp/model-cache)
    """

    def __init__(
        self, 
        model:PreTrainedModel, 
        optim:Optimizer, 
        scheduler:_LRScheduler,
        acc_threshold:Optional[float] =1.0, 
        patience:Optional[int] =None,
        incremental:Optional[bool] =False,
        val_loader:DataLoader =None,
        cache_dir:Optional[str] ="/tmp/model-cache"
    ) -> None:
        # save arguments
        self.acc_threshold = acc_threshold
        self.incremental = incremental

        # save optimizer and initialize evaluator
        self.optim = idist.auto_optim(optim)
        self.scheduler = scheduler
        super(Trainer, self).__init__(model)
        # save validation dataloader
        self.val_loader = val_loader

        # save initial checkpoint
        self.init_model_ckpt = deepcopy(self.model.state_dict())
        self.init_optim_ckpt = deepcopy(self.optim.state_dict())
        self.init_scheduler_ckpt = deepcopy(self.scheduler.state_dict())
        # add event handler        
        self.add_event_handler(Events.STARTED, type(self)._load_init_ckpt)
        self.add_event_handler(Events.STARTED, type(self)._prepare_scheduler)

        # create train evaluator
        self.train_evaluator = Evaluator(model)
        Accuracy(output_transform=Evaluator.get_logits_and_labels).attach(self.train_evaluator, 'A')
        # check if training accuracy threshold is reached
        self.add_event_handler(Events.EPOCH_COMPLETED, type(self)._check_convergence)

        # create validation evaluator
        self.val_evaluator = Evaluator(model)
        Average(output_transform=Evaluator.get_loss).attach(self.val_evaluator, 'L')
        # validate after each epoch
        self.add_event_handler(Events.EPOCH_COMPLETED, type(self)._validate)

        # attach checkpoint handler saving best model parameters
        # w.r.t. validation loss
        self.ckpt = Checkpoint(
            to_save={
                'model': self.model,
                # also save evaluators to capture metrics corresponding to model
                'train-evaluator': self.train_evaluator,
                'val-evaluator': self.val_evaluator
            },
            save_handler=DiskSaver(dirname=cache_dir, require_empty=False),
            score_function=lambda e: -e.state.metrics['L'],
            n_saved=1
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
        return self.train_evaluator.state.metrics['A']
    
    def step(self, batch):
        """ Training step function executed by the engine. """
        # move to device
        batch = {key: val.to(idist.device()) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
        # forward through model and compute loss
        self.model.train()
        self.optim.zero_grad()
        out = self.model.forward(**batch)
        # optimizer parameters
        type(self).get_loss(out).backward()
        self.optim.step()
        self.scheduler.step()
        # detach and move to cpu
        out = {key: val.detach().cpu() if isinstance(val, torch.Tensor) else val for key, val in out.items()}
        out['labels'] = batch['labels'].cpu()
        # return all outputs for logging and metrics computation
        return out

    def _load_init_ckpt(self):
        """ Event handler to load the initial checkpoints. Called on `STARTED`. """
        if not self.incremental:
            # reset model and optimizer states
            self.model.load_state_dict(self.init_model_ckpt)
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
        state = self.train_evaluator.run(self.state.dataloader)
        if (state.metrics['A'] >= self.acc_threshold):
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
        self.ckpt.load_objects(
            to_load={
                'model': self.model,
                'train-evaluator': self.train_evaluator,
                'val-evaluator': self.val_evaluator
            },
            checkpoint=self.ckpt.last_checkpoint
        )

    def _reset_ckpt(self):
        """ Event handler to reset the checkpoint. Called on `EPOCH_STARTED`. """
        self.ckpt.reset()

    def _reset_stopper(self):
        """ Event handler to reset the early stopper state.
            Callend on `EPOCH_STARTED`.
        """
        self.stopper.best_score = None
        self.stopper.counter = 0
