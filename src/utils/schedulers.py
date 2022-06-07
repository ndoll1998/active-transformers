import warnings
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

class _TrainingStepsDependentMixin(_LRScheduler):

    @property
    def num_training_steps(self) -> int:
        if not hasattr(self, '_num_training_steps'):
            warnings.warn("Number of training steps not set!", UserWarning)
            return -1
        return self._num_training_steps

    @num_training_steps.setter
    def num_training_steps(self, steps:int) -> None:
        self.set_num_training_steps(steps)

    def set_num_training_steps(self, steps:int) -> None:
        self._num_training_steps = steps

class LinearWithWarmup(LambdaLR, _TrainingStepsDependentMixin):

    def __init__(
        self,
        optim:Optimizer,
        warmup_proportion:float,
        last_epoch:int =-1,
        verbose:bool =False
    ) -> None:
        # save warmup proportion
        # number of training steps is handled by mixin
        self.warmup_proportion = warmup_proportion
        # initialize scheduler
        super(LinearWithWarmup, self).__init__(
            optimizer=optim,
            lr_lambda=self._lr_lambda,
            last_epoch=last_epoch,
            verbose=verbose
        )

    @property
    def num_warmup_steps(self) -> int:
        return int(self.num_training_steps * self.warmup_proportion)

    def _lr_lambda(self, current_step:int) -> float:
        # get total number of training steps and warmup steps
        total_steps = self.num_training_steps
        warmup_steps = self.num_warmup_steps
        # compute learning rate scaling factor
        return (
            # warmup
            (float(current_step) / max(1.0, float(warmup_steps))) if current_step < warmup_steps else \
            # linear decay
            max(0.0, float(total_steps - current_step) / max(1.0, float(total_steps - warmup_steps)))
        )
