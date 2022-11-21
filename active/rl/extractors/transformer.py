import gym
import torch
import transformers

from .extractor import FeatureExtractor

class TransformerFeatureExtractor(FeatureExtractor):

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        pretrained_ckpt:str,
    ) -> None:
        # initialize feature extractor
        super(TransformerFeatureExtractor, self).__init__(obs_space)
        # load pretrained transformer encoder
        self.encoder = transformers.AutoModel.from_pretrained(pretrained_ckpt)

    @property
    def feature_size(self) -> int:
        return self.encoder.config.hidden_size

    def forward(self, input_dict) -> torch.Tensor:
        # get observations from input
        input_ids = input_dict['obs']['input_ids'].long()
        attention_mask = input_dict['obs']['attention_mask'].long()

        # avoid invalid attention mask of all zeros, this seems to
        # happend when the environment is checked before actual
        # execution, also this should make sense as the first token
        # should always be the initial [CLS] special token
        attention_mask[:, 0] = 1

        # pass through transformer
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # return hidden state of [CLS]
        return out.last_hidden_state[:, 0, :]
