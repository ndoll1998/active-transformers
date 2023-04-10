import torch
import torch.nn as nn
import ignite.distributed as idist
from transformers import PreTrainedModel
# import base classes
from .badge import Badge
from ..common.strategy import ActiveStrategy
# import utils
import warnings
from typing import Any, Tuple

class BadgeForTokenClassification(Badge):
    """ Implementation of `Deep Batch Active Learning By Diverse,
        Uncertain Gradient Lower Bounds` (Ash et al., 2019) for
        transformer-based token classification models.

        Uses the final hidden states produced by the encoder as
        embeddings which are passed to the linear classifier.

        Pipeline:
        [input] -> encoder -> [out] -> [out.pooler_output] -> classifier -> [logits]

        Args:
            encoder (PreTrainedModel): transformer-based encoder
            classifier (nn.Linear): linear classifier
    """

    def __init__(
        self,
        encoder:PreTrainedModel,
        classifier:nn.Linear
    ) -> None:
        # initialize strategy
        ActiveStrategy.__init__(self)
        # move encoder to device(s)
        self.encoder = idist.auto_model(encoder)
        # make sure the classifier is linear and move
        # it to available device(s)
        assert isinstance(classifier, nn.Linear)
        self.classifier = idist.auto_model(classifier)
        # save classification type
        self.is_tok_cls = True

    def _get_logits_and_embeds(self, batch:Any) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # set model to eval mode
        self.encoder.eval()
        self.classifier.eval()
        # apply encoder and extract embeddings from output
        out = self.encoder(**batch, output_hidden_states=True)
        embeds = out.hidden_states[-1]
        # apply classifier and return
        return self.classifier(embeds), embeds

