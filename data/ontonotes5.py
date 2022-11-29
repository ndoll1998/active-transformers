""" Mostly copied from `https://huggingface.co/datasets/tner/ontonotes5/blob/main/ontonotes5.py`
    but use actual label space instead of simple int32 value
"""

import json
from itertools import chain
import datasets

logger = datasets.logging.get_logger(__name__)
_DESCRIPTION = """[ontonotes5 NER dataset](https://aclanthology.org/N06-2015/)"""
_NAME = "ontonotes5"
_VERSION = "1.0.0"
_CITATION = """
@inproceedings{hovy-etal-2006-ontonotes,
    title = "{O}nto{N}otes: The 90{\%} Solution",
    author = "Hovy, Eduard  and
      Marcus, Mitchell  and
      Palmer, Martha  and
      Ramshaw, Lance  and
      Weischedel, Ralph",
    booktitle = "Proceedings of the Human Language Technology Conference of the {NAACL}, Companion Volume: Short Papers",
    month = jun,
    year = "2006",
    address = "New York City, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N06-2015",
    pages = "57--60",
}
"""

_HOME_PAGE = "https://github.com/asahi417/tner"
_URL = f'https://huggingface.co/datasets/tner/{_NAME}/raw/main/dataset'
_URLS = {
    str(datasets.Split.TEST): [f'{_URL}/test.json'],
    str(datasets.Split.TRAIN): [f'{_URL}/train{i:02d}.json' for i in range(4)],
    str(datasets.Split.VALIDATION): [f'{_URL}/valid.json'],
}

_NER_TAGS = [
    "O",
    "B-CARDINAL",
    "B-DATE",
    "I-DATE",
    "B-PERSON",
    "I-PERSON",
    "B-NORP",
    "B-GPE",
    "I-GPE",
    "B-LAW",
    "I-LAW",
    "B-ORG",
    "I-ORG", 
    "B-PERCENT",
    "I-PERCENT", 
    "B-ORDINAL", 
    "B-MONEY", 
    "I-MONEY", 
    "B-WORK_OF_ART", 
    "I-WORK_OF_ART", 
    "B-FAC", 
    "B-TIME", 
    "I-CARDINAL", 
    "B-LOC", 
    "B-QUANTITY", 
    "I-QUANTITY", 
    "I-NORP", 
    "I-LOC", 
    "B-PRODUCT", 
    "I-TIME", 
    "B-EVENT",
    "I-EVENT",
    "I-FAC",
    "B-LANGUAGE",
    "I-PRODUCT",
    "I-ORDINAL",
    "I-LANGUAGE"
]

class Ontonotes5Config(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """BuilderConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Ontonotes5Config, self).__init__(**kwargs)


class Ontonotes5(datasets.GeneratorBasedBuilder):
    """Dataset."""

    BUILDER_CONFIGS = [
        Ontonotes5Config(name=_NAME, version=datasets.Version(_VERSION), description=_DESCRIPTION),
    ]

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download_and_extract(_URLS)
        return [datasets.SplitGenerator(name=i, gen_kwargs={"filepaths": downloaded_file[str(i)]})
                for i in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]]

    def _generate_examples(self, filepaths):
        _key = 0
        for filepath in filepaths:
            logger.info(f"generating examples from = {filepath}")
            with open(filepath, encoding="utf-8") as f:
                _list = [i for i in f.read().split('\n') if len(i) > 0]
                for i in _list:
                    data = json.loads(i)
                    yield _key, data
                    _key += 1

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "tags": datasets.Sequence(datasets.ClassLabel(names=_NER_TAGS)),
                }
            ),
            supervised_keys=None,
            homepage=_HOME_PAGE,
            citation=_CITATION,
        )
