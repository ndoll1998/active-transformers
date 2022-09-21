# torch
import torch
import torch.nn as nn
# import gym and rllib
import gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
# import feature extractor
from ..extractors.extractor import FeatureExtractor
# typing
from typing import Type, List
from ray.rllib.utils.typing import TensorType, ModelConfigDict

class StreamBasedModel(TorchModelV2, nn.Module):

    def __init__(
        self,
        # observation and action space
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        # model setup and name
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        # feature extractor
        feature_extractor_type:Type[FeatureExtractor],
        feature_extractor_config:dict ={},
        # prediction heads
        pi_hidden_layers:List[int] =[128],
        vf_hidden_layers:List[int] =[128],
    ) -> None:
        # initialize super classes
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # create feature extractor
        self.feature_extractor = feature_extractor_type(
            obs_space=obs_space,
            **feature_extractor_config
        )

        # get feature dimension
        feature_dim = self.feature_extractor.feature_dim
        # action head
        self.pi = nn.Sequential(*[
            nn.Sequential(nn.Linear(n, m), nn.ReLU())
            for n, m in zip(
                [feature_dim] + list(pi_hidden_layers),
                list(pi_hidden_layers) + [num_outputs]
            )
        ])
        # value-function head
        self.vf = nn.Sequential(*[
            nn.Sequential(nn.Linear(n, m), nn.ReLU())
            for n, m in zip(
                [feature_dim] + list(vf_hidden_layers),
                list(pi_hidden_layers) + [1]
            )
        ])

        # set in forward call and used by value function
        self._features:TensorType = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens) -> (TensorType, List[TensorType]):
        # extract features and apply action head
        self._features = self.feature_extractor(input_dict)
        return self.pi(self._features), state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        # check for features and apply value-function head
        assert self._features is not None, "No features found: make sure to call `forward` before `value_function`"
        return self.vf(self._features).squeeze(1)
