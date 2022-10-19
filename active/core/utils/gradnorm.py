import torch
import torch.nn as nn
from typing import Tuple, Any

class GoodfellowGradientNormForLinear(object):
    """ Goodfellow Gradient Norm for single linear layer.
        Attaches hooks to the given linear module. Gradient
        Norm is ready for computation after backward call.

        Gradient norms are computed for a single batch only
        and resetted for each forward call to the linear module.
        However multiple backward calls are supported.
    
        Can be used as a context manager.

        Args:
            module (nn.Linear): linear layer
    """
    
    def __init__(self, module:nn.Linear) -> None:
        assert isinstance(module, nn.Linear)
        # register capture hooks
        self.forward_handle = module.register_forward_hook(self.forward_hook)
        self.backward_handle = module.register_full_backward_hook(self.backward_hook)
        # reset
        self.H:torch.FloatTensor = None
        self.Z_bar:torch.FloatTensor = None

    def remove_hooks(self) -> None:
        """ Remove all hooks from module """
        # remove forward and backward hook
        self.forward_handle.remove()
        self.backward_handle.remove()

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

    def __enter__(self) -> "GoodfellowGradientNormForLinear":
        return self

    def __exit__(self, type:Any, value:Any, tb:Any) -> None:
        self.remove_hooks()


class GoodfellowGradientNorm(object):
    """ Compute Example-wise gradient norm of all linear components in the
        given module following Goodfellow (2016).
        
        Gradient norms are computed for a single batch only
        and resetted for each forward call to the linear module.
        However multiple backward calls are supported.
    
        Can be used as a context manager.
       
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

    def remove_hooks(self) -> None:
        """ Remove all hooks from module """
        for grad in self.grad_captures:
            grad.remove_hooks()

    def compute(self) -> torch.FloatTensor:
        """ Compute Squarred Gradient Norm for last processed batch 
        
            Returns:
                norm (torch.FloatTensor): 
                    squared gradient norm for weight of linear module
                    in shape (b, m) where b is the batch size and m is
                    the number of backward calls.
        """
        # compute all gradients and add them
        return sum(grad.compute() for grad in self.grad_captures)
    
    def __enter__(self) -> "GoodfellowGradientNorm":
        return self

    def __exit__(self, type:Any, value:Any, tb:Any) -> None:
        self.remove_hooks()

