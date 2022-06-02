import torch
import ignite.distributed as idist
from ignite.engine import Engine
from transformers import PreTrainedModel

class Evaluator(Engine):

    def __init__(
        self,
        model:PreTrainedModel,
    ) -> None:
        # save model and initialize engine
        self.model = idist.auto_model(model)
        super(Evaluator, self).__init__(self.step)

    @torch.no_grad()
    def step(self, engine, batch):
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
        logits, labels = out['logits'], out['labels']
        mask = (labels >= 0)
        return logits[mask], labels[mask]

    @staticmethod
    def get_loss(out:dict):
        return out['loss'].mean()

class Trainer(Evaluator):
    
    def __init__(
        self,
        model:PreTrainedModel,
        optim:torch.optim.Optimizer
    ) -> None:
        # save optimizer and initialize evaluator
        self.optim = idist.auto_optim(optim)
        super(Trainer, self).__init__(model)
    
    def step(self, engine, batch):
        # move to device
        batch = {key: val.to(idist.device()) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}
        # forward through model and compute loss
        self.model.train()
        self.optim.zero_grad()
        out = self.model.forward(**batch)
        # optimizer parameters
        type(self).get_loss(out).backward()
        self.optim.step()
        # detach and move to cpu
        out = {key: val.detach().cpu() if isinstance(val, torch.Tensor) else val for key, val in out.items()}
        out['labels'] = batch['labels'].cpu()
        # return all outputs for logging and metrics computation
        return out
