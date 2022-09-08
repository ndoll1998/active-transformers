import re
import torch
from transformers import PreTrainedModel
from collections import defaultdict
from typing import Optional, Dict

class TransformerParameterGroups(list):
    """ Helper class for Parameter Groups of transformer-based models. Used to initialize
        optimizers for transformer models. Adds basic hyperparameter-features like:
            - apply weight decay only to non-bias parameters
            - layer-wise learning rate decay

        Args:
            model (PreTrainedModel): pre-trained model to build parameter groups for
            lr (float): base learning rate
            lr_decay (Optional[float]): learning rate decay. Defaults to 1.0, i.e. constant learning rate.
            weight_decay (Optional[float]): weight decay rate. Default to 0.01.
    """

    def __init__(
        self,
        model:PreTrainedModel,
        lr:float,
        lr_decay:Optional[float] =1.0,
        weight_decay:Optional[float] =0.01
    ) -> None:
        # initialize list
        super(TransformerParameterGroups, self).__init__()

        # pattern to extract layer number of a
        # parameter from it's name
        pattern = re.compile(r".layer.(?P<num>[0-9]*).")
        num_index = pattern.groupindex['num']
        # group parameters into layers
        groups = defaultdict(dict)
        for name, param in model.named_parameters():
            match = pattern.search(name)
            layer = 'default' if match is None else int(match.group(num_index))
            groups[layer][name] = param

        # build parameter groups with set hyperparameters
        # start with default group
        self.add_parameters(
            named_parameters=groups.pop('default'),
            lr=lr,
            weight_decay=weight_decay
        )

        # get the total number of layers found
        # actually not the total number of but the biggest index
        num_layers = max(groups.keys())
        # add all layer parameters
        for i, params in groups.items():
            self.add_parameters(
                named_parameters=params,
                lr=lr * lr_decay**(num_layers - i),
                weight_decay=weight_decay
            )

    def add_parameters(
        self,
        named_parameters:Dict[str, torch.Tensor],
        lr:float,
        weight_decay:float
    ) -> None:
        """ Add parameter group for the given named parameters. Actually splits
            the parameters into bias and non-bias parameters and only applies
            weight decay to non-bias parameters.
        
            Args:
                named_parameters (Dict[str, torch.Tensor]): 
                    parameters to add to the group
                lr (float): 
                    learning rate for the parameter group
                weight_decay (float): 
                    weight decay for the parmaeter group. Only applied to
                    non-bias parameters
        """
        # non-biases with weight decay
        self.append(dict(
            params=[param for name, param in named_parameters.items() if not name.endswith('.bias')],
            lr=lr, weight_decay=weight_decay
        ))
        # disable weight decay for bias parameters
        self.append(dict(
            params=[param for name, param in named_parameters.items() if name.endswith('.bias')],
            lr=lr, weight_decay=0.0
        ))

