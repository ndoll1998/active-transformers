import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import transformers
from active.engine.strategies.badge.sequence_badge import BadgeForSequenceClassification
# utils
import warnings
from active.utils.model import get_encoder_from_model
from tests.utils.strategy import _test_run_strategy

class TestBadgeForSequenceClassification(object):

    def test_run_strategy(self):
        # create distil bert model
        config = transformers.DistilBertConfig()
        model = transformers.DistilBertForSequenceClassification(config)
        # create and test strategy
        with warnings.catch_warnings(record=True):
            strategy = BadgeForSequenceClassification(
                encoder=get_encoder_from_model(model),
                classifier=model.classifier
            )
            _test_run_strategy(strategy, vocab_size=config.vocab_size)
