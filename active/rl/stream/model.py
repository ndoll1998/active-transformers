# torch
import torch
import torch.nn as nn
# import gym and rllib
import gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.annotations import override
from ray.rllib.policy.rnn_sequencing import add_time_dimension
# import feature extractor
from ..extractors.extractor import FeatureExtractor
# typing
from typing import Type, List
from ray.rllib.utils.typing import TensorType, ModelConfigDict

class DQNModel(DQNTorchModel):
    
    @override(DQNTorchModel)
    def __init__(
        self,
        # observation and action space
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        # model setup and name
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        *,
        # feature extractor
        feature_extractor_type:Type[FeatureExtractor],
        feature_extractor_config:dict ={},
        # dqn model keyword arguments
        **kwargs
    ) -> None:

        # create feature extractor
        feature_extractor = feature_extractor_type(
            obs_space=obs_space,
            **feature_extractor_config
        )

        # get feature dimension
        feature_size = feature_extractor.feature_size
        
        # initialize super class
        DQNTorchModel.__init__(
            self,
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=feature_size,
            model_config=model_config,
            name=name,
            **kwargs
        )

        # set feature extractor after module is initialized
        self.feature_extractor = feature_extractor
        
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens) -> (TensorType, List[TensorType]):
        # extract features
        return self.feature_extractor(input_dict), state
        
        
class ActorCriticModel(TorchModelV2, nn.Module):

    def __init__(
        self,
        # observation and action space
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        # model setup and name
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        *,
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
        feature_size = self.feature_extractor.feature_size
        # action head
        self.pi = nn.Sequential(*[
            nn.Sequential(nn.Linear(n, m), nn.ReLU())
            for n, m in zip(
                [feature_size] + list(pi_hidden_layers),
                list(pi_hidden_layers) + [num_outputs]
            )
        ])
        # value-function head
        self.vf = nn.Sequential(*[
            nn.Sequential(nn.Linear(n, m), nn.ReLU())
            for n, m in zip(
                [feature_size] + list(vf_hidden_layers),
                list(pi_hidden_layers) + [1]
            )
        ])

        # set in forward call and used by value function
        self._features:TensorType = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens) -> (TensorType, List[TensorType]):
        # extract features and apply action head
        self._features = self.feature_extractor(input_dict)
        return self.pi(self._features), state

    @override(ModelV2)
    def value_function(self) -> TensorType:
        # check for features and apply value-function head
        assert self._features is not None, "No features found: make sure to call `forward` before `value_function`"
        return self.vf(self._features).squeeze(1)


class RecurrentActorCriticModel(ActorCriticModel):

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
        # rnn
        lstm_num_layers:int =1,
    ) -> None:
    
        # initialize actor critic module
        super(RecurrentActorCriticModel, self).__init__(
            # observation and action space
            obs_space=obs_space,
            action_space=action_space,
            # model setup and name
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            # feature extractor
            feature_extractor_type=feature_extractor_type,
            feature_extractor_config=feature_extractor_config,
            # prediction heads
            pi_hidden_layers=pi_hidden_layers,
            vf_hidden_layers=vf_hidden_layers
        )

        # get feature dimension
        feature_size = self.feature_extractor.feature_size
        # create rnn
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=feature_size,
            num_layers=lstm_num_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            # bi-directional doesn't make sense
            bidirectional=False
        )
        # create learnable initial lstm state
        self.h_0 = nn.Parameter(torch.zeros(lstm_num_layers, feature_size))
        self.c_0 = nn.Parameter(torch.zeros(lstm_num_layers, feature_size))

    @override(ModelV2)
    def get_initial_state(self):
        return [self.h_0, self.c_0]

    @override(ActorCriticModel)
    def forward(self, input_dict, state, seq_lens) -> (TensorType, List[TensorType]):

        # unpack state and transpose to match
        # expectation of lstm module
        h_i, c_i = state
        h_i = h_i.transpose(0, 1)
        c_i = c_i.transpose(0, 1)

        # extract features
        features = self.feature_extractor(input_dict)
        # get batch size
        n = features.size(0)

        # add time dimension        
        features = add_time_dimension(
            padded_inputs=features,
            max_seq_len=seq_lens.max(),
            framework='torch',
            time_major=False
        )

        # apply recurrent network
        features, (h_j, c_j) = self.lstm(features, (h_i, c_i))
        features = features.reshape(n, -1)

        # save features for value branch and compute logits
        self._features = features
        logits = self.pi(features)

        # reverse transposition to stay consistent
        h_j = h_j.transpose(0, 1)
        c_j = c_j.transpose(0, 1)

        # return policy output and new state
        return logits, [h_j, c_j]
