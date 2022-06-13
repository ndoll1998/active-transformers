import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.distributed as idist
# base strategy and utils
from .strategy import AbstractStrategy
from .utils import move_to_device

from transformers import PreTrainedModel
from typing import Tuple, Sequence, Any

class GoodfellowGradientNormForLinear(object):
    """ Goodfellow Gradient Norm for single linear layer.
        Attaches hooks to the given linear module. Gradient
        Norm is ready for computation after backward call.

        Gradient norms are computed for a single batch only
        and resetted for each forward call to the linear module.
        However multiple backward calls are supported.
    
        Args:
            module (nn.Linear): linear layer
    """

    def __init__(self, module:nn.Linear) -> None:
        assert isinstance(module, nn.Linear)
        # register capture hooks
        module.register_forward_hook(self.forward_hook)
        module.register_full_backward_hook(self.backward_hook)
        # reset
        self.H:torch.FloatTensor = None
        self.Z_bar:torch.FloatTensor = None

    @torch.no_grad()
    def forward_hook(
        self,
        module:nn.Linear,
        x:Tuple[torch.FloatTensor], 
        y:Tuple[torch.FloatTensor]
    ) -> None:
        """ Forward hook capturing inputs to module 
        
            Args:
                x (Tuple[torch.FloatTensor]): input to linear module
                y (Tuple[torch.FloatTensor]): output of linear module
        """
        # compute squared norm of input
        self.H = (x[0] * x[0]).flatten(start_dim=1).sum(dim=1)
        self.Z_bar = []

    @torch.no_grad()
    def backward_hook(
        self, 
        module:nn.Linear,
        x_grad:Tuple[torch.FloatTensor], 
        y_grad:Tuple[torch.FloatTensor]
    ) -> None:
        """ Backward hook caturing gradients of module outputs 
        
            Args:
                x_grad (Tuple[torch.FloatTensor]): gradient of input
                y_grad (Tuple[torch.FloatTensor]): gradient of output
        """
        # sum of squared gradients
        z_bar = (y_grad[0] * y_grad[0]).flatten(start_dim=1).sum(dim=1)
        self.Z_bar.append(z_bar)

    def compute(self) -> torch.FloatTensor:
        """ Compute Squarred Gradient Norm for last processed batch 
        
            Returns:
                norm (torch.FloatTensor): 
                    squared gradient norm for weight of linear module
                    in shape (b, m) where b is the batch size and m is
                    the number of backward calls.
        """
        # stack gradients from multiple backward calls
        # +1 to account for bias
        H = self.H.unsqueeze(1) + 1.0
        Z_bar = torch.stack(self.Z_bar, dim=1)
        # check dimensions
        assert H.size(0) == Z_bar.size(0), "Mismatch between forward (%i) and backward (%i) calls" % (H.size(0), Z_bar.size(0))
        return H * Z_bar


class GoodfellowGradientNorm(object):
    """ Compute Example-wise gradient norm of all linear components in the
        given module following Goodfellow (2016).
        
        Gradient norms are computed for a single batch only
        and resetted for each forward call to the linear module.
        However multiple backward calls are supported.
    
        Args:
            module (nn.Module): module which's linear components to analyze
    """

    def __init__(self, model:nn.Module) -> None:
        # register capturing hooks
        self.grad_captures = [
            GoodfellowGradientNormForLinear(module)
            for module in model.modules()
            if isinstance(module, nn.Linear)
        ]

    def compute(self) -> torch.FloatTensor:
        """ Compute Squarred Gradient Norm for last processed batch 
        
            Returns:
                norm (torch.FloatTensor): 
                    squared gradient norm for weight of linear module
                    in shape (b, m) where b is the batch size and m is
                    the number of backward calls.
        """
        # compute all gradients and concatenate them
        return sum(grad.compute() for grad in self.grad_captures)


class EglForSequenceClassification(AbstractStrategy):
    """ Implements Maximum Expected Gradient Length Sampling Strategy
        for sequence classification tasks. Assumes cross-entropy loss.
    
        Args:
            model (PreTrainedModel): 
                transformer-based sequence classification model
            k (int):
                maximum number of classes to consider for
                expectation approximation. Defaults to 5.
    """

    def __init__(
        self,
        model:PreTrainedModel,
        k:int =5
    ) -> None:
        # initialize strategy
        super(EglForSequenceClassification, self).__init__()
        # create gradient norm computer
        self.grad_norm = GoodfellowGradientNorm(model)
        # move model to available device(s)
        self.model = idist.auto_model(model)
        # save k
        self.k = k

    def process(self, batch:Any) -> torch.FloatTensor:
        """ Compute expected gradient lengths for given batch 
        
            Args:
                batch (Any): input batch for which to compute grad norms
        
            Returns:
                norm (torch.FloatTensor): (squared) gradient norm length computed following Goodfellow (2016)
        """
        # move batch to device and pass through model
        batch = move_to_device(batch, device=idist.device())
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1) 
        # get top-k predictions
        assert logits.ndim == 2, "Expected simple sequence classification, i.e. logits of two dimensions but got logits of shape %s" % str(logits.shape)
        k = min(self.k, logits.size(1))
        probs, preds = torch.topk(probs, k=k, dim=-1)
        # compute loss and backpropagate
        for i in range(k):
            F.cross_entropy(logits, preds[:, i], reduction='sum') \
                .backward(retain_graph=True)
        # approximate expected gradient length
        with torch.no_grad():
            return torch.sum(probs * self.grad_norm.compute(), dim=1)

    def sample(self, output:torch.FloatTensor, query_size:int) -> Sequence[int]:
        """ Get samples with maximum expected gradient length 
        
            Args:
                output (torch.FloatTensor): expected gradient length of samples
                query_size (int): number of samples to draw

            Returns:
                indices (Sequence[int]): drawn examples given by their indices in `output`
        """
        # get the samples with top expected gradient length
        return output.topk(k=query_size).indices
