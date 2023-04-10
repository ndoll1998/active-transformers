import torch
import torch.nn as nn
from torch._utils import _get_device_index
from active.utils.data import move_to_device
from collections import defaultdict
from typing import Tuple, List, Dict, Any


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
        # caches
        self.H:torch.FloatTensor = None
        self.Z_bar:List[torch.FloatTensor] = []

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
        self.H = (x[0] * x[0]).sum(dim=-1)
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
        z_bar = (y_grad[0] * y_grad[0]).sum(dim=-1)
        self.Z_bar.append(z_bar)

    def compute(self) -> torch.FloatTensor:
        """ Compute Squarred Gradient Norm for last processed batch

            Returns:
                norm (torch.FloatTensor):
                    squared gradient norm for weight of linear module
                    in shape (..., m) where m is the number of backward calls.
        """
        if self.H is None:
            return 0.0
        # stack gradients from multiple backward calls
        # +1 to account for bias
        h = self.H.unsqueeze(-1) + 1.0
        z_bar = torch.stack(self.Z_bar, dim=-1)
        # check dimensions
        assert h.size(0) == z_bar.size(0), "Mismatch between forward (%i) and backward (%i) calls" % (h.size(0), z_bar.size(0))
        return h * z_bar

    def __enter__(self) -> "GoodfellowGradientNormForLinear":
        return self

    def __exit__(self, type:Any, value:Any, tb:Any) -> None:
        self.remove_hooks()


class GoodfellowGradientNormForDataParallelLinear(GoodfellowGradientNormForLinear):

    def __init__(self, module:nn.Linear, device_ids:List[int], output_device_id:int) -> None:
        assert isinstance(module, nn.Linear)
        self.device_ids = device_ids
        self.output_device_id = output_device_id
        # initialize super class
        super(GoodfellowGradientNormForDataParallelLinear, self).__init__(module)
        # overwrite caches
        self.H:Dict[torch.device, torch.FloatTensor] = dict.fromkeys(self.device_ids, None)
        self.Z_bar:Dict[torch.device, torch.FloatTensor] = defaultdict(list)

    @torch.no_grad()
    def forward_hook(
        self,
        module:nn.Linear,
        x:Tuple[torch.FloatTensor],
        y:torch.FloatTensor
    ) -> None:
        """ Forward hook capturing inputs to module

            Args:
                x (Tuple[torch.FloatTensor]): input to linear module
                y (Tuple[torch.FloatTensor]): output of linear module
        """
        # get device id
        device_id = _get_device_index(x[0].device)
        assert device_id in self.device_ids, "Invalid device id %i (%s)" % (device_id, str(x[0].device))
        # compute squared norm of input
        self.H[device_id] = (x[0] * x[0]).sum(dim=-1)
        self.Z_bar[device_id] = []

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
        # get device id
        device_id = _get_device_index(y_grad[0].device)
        assert device_id in self.device_ids, "Invalid device id %i (%s)" % (device_id, str(y_grad[0].device))
        # sum of squared gradients
        z_bar = (y_grad[0] * y_grad[0]).sum(dim=-1)
        self.Z_bar[device_id].append(z_bar)

    def compute(self) -> torch.FloatTensor:
        """ Compute Squarred Gradient Norm for last processed batch

            Returns:
                norm (torch.FloatTensor):
                    squared gradient norm for weight of linear module
                    in shape (..., m) where m is the number of backward calls.
        """
        if all((h is None) for h in self.H.values()):
            return 0.0
        # concatenate over devices
        # +1 to account for gradient of bias
        h = torch.cat([
            self.H[i].to(self.output_device_id)
            for i in self.device_ids if self.H[i] is not None
        ], dim=0) + 1.0
        z_bar = torch.cat([
            # stack over multiple backward calls
            torch.stack(move_to_device(self.Z_bar[i], device=self.output_device_id), dim=-1)
            for i in self.device_ids if self.H[i] is not None
        ], dim=0)

        # clear caches
        self.H = dict.fromkeys(self.device_ids, None)
        self.Z_bar = defaultdict(list)
        # check dimensions
        assert h.size(0) == z_bar.size(0), "Mismatch between forward (%i) and backward (%i) calls" % (h.size(0), z_bar.size(0))
        return h.unsqueeze(-1) * z_bar


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

        self.names, modules = zip(*[
            (name, module) for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ])

        # check if model uses multiple gpus
        is_multi_gpu = isinstance(model, nn.DataParallel) and len(model.device_ids) > 1
        # register capturing hooks
        self.grad_captures = [
            GoodfellowGradientNormForDataParallelLinear(module, model.device_ids, model.output_device)
            for module in modules
        ] if is_multi_gpu else [
            GoodfellowGradientNormForLinear(module)
            for module in modules
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
                    in shape (..., m) where m is the number of backward calls.
        """
        # compute all gradients and add them
        return sum(grad.compute() for grad in self.grad_captures)

    def per_module(self) -> torch.FloatTensor:
        """ Compute Squarred Gradient Norm for each linear module

            Returns:
                norms (torch.FloatTensor):
                    squared gradient norms for weight of each linear module
                    in shape (n, ..., m) where n is the number of linear modules
                    m is the number of backward calls.
        """
        return torch.stack(tuple(grad.compute() for grad in self.grad_captures), dim=0)

    def __enter__(self) -> "GoodfellowGradientNorm":
        return self

    def __exit__(self, type:Any, value:Any, tb:Any) -> None:
        self.remove_hooks()

