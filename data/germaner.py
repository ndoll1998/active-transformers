import os
import random
import datasets
from data.conll_format_builder import ConllFormatBuilder

_RAW_DATA_URL = "https://raw.githubusercontent.com/tudarmstadt-lt/GermaNER/a206b554feca263d740302449fff0776c66d0040/data/v0.9.1/full_train.tsv"

class GermaNerBuilder(ConllFormatBuilder):
    """ Dataset Generator for GermaNER Dataset """

    FORMAT=["tokens", "ner_tags"]
    FEATURE_TYPES={
        "tokens": datasets.Value("string"),
        "ner_tags": datasets.ClassLabel(names=[
            "O",
            "B-LOC",
            "B-ORG",
            "B-OTH",
            "B-PER",
            "I-LOC",
            "I-ORG",
            "I-OTH",
            "I-PER"
        ])
    }

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="germaner",
            version=datasets.Version("0.1.0"),
            description="GermaNER dataset"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="GermaNER dataset",
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
            examples = f.read().replace('\t', ' ').split('\n\n')

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

