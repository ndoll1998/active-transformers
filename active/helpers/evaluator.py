import torch
import torch.nn as nn
# import ignite
import ignite.distributed as idist
from ignite.engine import Engine, Events
# transformers
from transformers import PreTrainedModel
# others
from ..core.utils.data import move_to_device

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

    @property
    def unwrapped_model(self) -> PreTrainedModel:
        """ The pure transformer model """
        if isinstance(self.model, PreTrainedModel):
            return self.model
        elif isinstance(self.model, nn.DataParallel):
            return self.model.module
        raise ValueError("Unexpected model type: %s" % type(self.model))

    @torch.no_grad()
    def step(self, batch):
        """ Evaluation step function """
        # move to device
        batch = move_to_device(batch, device=idist.device())
        # forward through model
        self.model.eval()
        out = self.model.forward(**batch)
        # pass labels through for metrics
        if 'labels' in batch:
            out['labels'] = batch['labels']
        # compute average loss over dataparallel model output
        if ('loss' in out) and (out['loss'] is not None) and (out['loss'].ndim >= 0):
            out['loss'] = out['loss'].mean()
        # back to cpu to clear device memory
        return move_to_device(out, device=torch.device('cpu'))

    @classmethod
    def get_logits_labels_mask(cls, out:dict):
        logits, labels = out['logits'], out['labels']
        return logits, labels, (labels >= 0)
    
    @classmethod
    def get_logits_and_labels(cls, out:dict):
        """ Function to extract the logits and labels from the model output.
            Can be used as `output_transform` in ignite metrics.
        
            Args:
                out (dict): 
                    output of a forward call to the `PreTrainedModel`.

            Returns:
                logits (torch.Tensor): logits tensor only containing valid entries
                labels (torch.Tensor): labels tensor only containing valid entries
        """
        logits, labels, mask = cls.get_logits_labels_mask(out)
        return logits[mask, :], labels[mask]
