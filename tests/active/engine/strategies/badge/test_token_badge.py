import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import transformers
from active.engine.strategies.badge.token_badge import BadgeForTokenClassification
# utils
import warnings
from active.utils.model import get_encoder_from_model
from tests.utils.strategy import RunTokenStrategyTests

class TestBadgeForTokenClassification(RunTokenStrategyTests):

    def create_strategy(self, model):
        return BadgeForTokenClassification(
            encoder=get_encoder_from_model(model),
            classifier=model.classifier
        )
