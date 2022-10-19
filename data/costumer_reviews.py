import os
import json
import random
import datasets

LABELS = ["negative", "positive"]
URL = "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/cr/all-0c9633c6.zip"

class CostumerReviews(datasets.GeneratorBasedBuilder):
    """ Dataset Generator for Costumer Reviews dataset """

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name='costumer_reviews',
            version=datasets.Version('0.1.0'),
            description="Costumer Reviews Dataset"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Costumer Reviews Dataset",
            features=datasets.Features({
                'text': datasets.Value("string"),
                'label': datasets.features.ClassLabel(names=LABELS)
            }),
            supervised_keys=None,
            homepage="",
            citation=""
        )

    def _split_generators(self, dl_manager):
        # download and extract
        fpath = dl_manager.download_and_extract(URL)
        # open file
        with open(os.path.join(fpath, "cr", "all-0c9633c6.json"), 'r') as f:
            samples = json.loads(f.read())
        # shuffle and split into train and test samples
        gen = random.Random(42)
        gen.shuffle(samples)
        n = int(len(samples) * 0.8)
        train_samples, test_samples = samples[:n], samples[n:]
        # build generators
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'samples': train_samples}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'samples': test_samples}),
        ]

    def _generate_examples(self, samples):
        for i, (text, label) in enumerate(samples):
            yield i, {'text': text, 'label': LABELS[int(label)]}
