import pytest
import warnings
import transformers
from active.strategies.badge import Badge

class TestBadge:
    """ Tests for Badge Strategy """

    def test_initialization_warnings(self):
        """ Test Badge initialization warnings """
        # create bert config
        config = transformers.BertConfig()
        # warning message to test for
        message = "Pipeline output doesn't match model output"

        # create bert model for sequence classification
        model = transformers.BertForSequenceClassification(config)
        with warnings.catch_warnings(record=True) as record:
            # expected to throw warning
            # badge uses hidden states as embedding by default but model uses pooler_output
            # results in mismatch between pipeline used by badge and model
            badge = Badge(model)
            # check infered classification type
            assert not badge.is_tok_cls, "Expected Sequence Classification but inferred Token Classification"
            # check expectation
            if not any((message in w.message.args[0]) for w in record):
                pytest.fail("Expected Warning: %s" % message)

        model = transformers.BertForTokenClassification(config)
        with warnings.catch_warnings(record=True) as record:
            # expects no warning
            # token classification model uses hidden states as embedding and simple linear
            # classifier on top, exactly matches badge pipeline
            badge = Badge(model)
            # check infered classification type
            assert badge.is_tok_cls, "Expected Token Classification but inferred Sequence Classification"
            # check expectation
            if any((message in w.message.args[0]) for w in record):
                pytest.fail("Unexpected Warning: %s" % message)

