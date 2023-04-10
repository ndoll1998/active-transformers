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

class BadgeForSequenceClassification(Badge):
    """ Implementation of `Deep Batch Active Learning By Diverse,
        Uncertain Gradient Lower Bounds` (Ash et al., 2019) for
        transformer-based sequence classification models.

        Uses the `pooler_output` field of the encoder output as
        embeddings which are passed to the linear classifier. If
        the field is not specified the hidden state of the first
        token (usually [CLS]) is used as embedding.

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
        self.is_tok_cls = False

    def _get_logits_and_embeds(self, batch:Any) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # set model to eval mode
        self.encoder.eval()
        self.classifier.eval()
        # apply encoder and extract embeddings from output
        # fallback to encoding of [CLS] token is pooler output is not specified
        out = self.encoder(**batch, output_hidden_states=True)
        embeds = out.hidden_states[-1][:, 0, :] if not hasattr(out, 'pooler_output') else \
            out.hidden_states[-1][:, 0, :] if out.pooler_output is None else \
            out.pooler_output
        # warn about fallback
        if not hasattr(out, 'pooler_output'):
            warnings.warn("Model output does not specify pooler output. Using final hidden state of [CLS] token instead.", UserWarning)
        # apply classifier and return
        return self.classifier(embeds), embeds
