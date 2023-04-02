import gym
import torch.nn as nn
from abc import ABC, abstractmethod
from ray.rllib.utils.typing import ModelConfigDict

class FeatureExtractor(ABC, nn.Module):
    """ Abstract Base Feature Extractor used by Models """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
    ) -> None:
        # initialize module
        Module.__init__(self)

    @property
    @abstractmethod
    def feature_size(self) -> int:
        """ Dimension of the extracted features """
        raise NotImplementedError()
