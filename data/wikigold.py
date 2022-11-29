import os
import random
import datasets
from data.conll_format_builder import ConllFormatBuilder

_RAW_DATA_URL = "https://figshare.com/ndownloader/files/9446377"

class WikigoldBuilder(ConllFormatBuilder):
    """ Dataset Generator for Wikigold Dataset """

    FORMAT=["tokens", "ner_tags"]
    FEATURE_TYPES={
        "tokens": datasets.Value("string"),
        "ner_tags": datasets.ClassLabel(names=[
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-MISC",
            "I-MISC"
        ])
    }

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="wikigold",
            version=datasets.Version("0.1.0"),
            description="Wikigold dataset"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="German Conll2003 dataset",
            features=self._features,
            supervised_keys=None,
            homepage="",
            citation=""
        )

    def _split_generators(self, dl_manager):

        # download file
        raw_data_fpath = dl_manager.download(_RAW_DATA_URL)
        raw_data_dir = os.path.dirname(raw_data_fpath)

        # read examples
        with open(raw_data_fpath, 'r') as f:
            examples = f.read().split('\n\n')

        n = len(examples)
        # typical 80/20 random split
        random.shuffle(examples)
        train_split = examples[:int(0.8*n)]
        test_split = examples[int(0.8*n):]

        train_fpath = os.path.join(raw_data_dir, "train_split.txt")
        test_fpath = os.path.join(raw_data_dir, "test_split.txt")
        # write to files
        with open(train_fpath, "w+") as f:
            f.write('\n\n'.join(train_split))
        with open(test_fpath, "w+") as f:
            f.write('\n\n'.join(test_split))

        # build generators
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'fpaths': [train_fpath]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'fpaths': [test_fpath]})
        ]

