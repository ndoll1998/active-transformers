import random
import datasets

_RAW_DATA_URL = "https://raw.githubusercontent.com/olofmogren/biomedical-ner-data-swedish/master/1177_annotated_sentences.txt"

class SwedishMedicalNerBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="swedish_medical_ner",
            version=datasets.Version("0.1.0"),
            description="Swedish medical NER corpus"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Swedish medical NER corpus",
            features=datasets.Features({
                'tokens': datasets.Sequence(datasets.Value("string")),
                'ner_tags': datasets.Sequence(datasets.ClassLabel(names=[
                    "O",
                    "B-DISORDER/FINDING",
                    "I-DISORDER/FINDING",
                    "B-DRUG",
                    "I-DRUG",
                    "B-BODY",
                    "I-BODY"
                ]))
            }),
            supervised_keys=None,
            homepage="",
            citation=""
        )

    def _split_generators(self, dl_manager):
        # download raw data
        fpath = dl_manager.download(_RAW_DATA_URL)
        # read contents
        with open(fpath, 'r') as f:
            samples = f.read().strip().split('\n')
        # split into train and test data
        n = len(samples)
        random.shuffle(samples)
        train_samples = samples[:int(0.8 * n)]
        test_samples = samples[int(0.8 * n):]
        # create data split generators
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'samples': train_samples}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'samples': test_samples})
        ]

    def _generate_examples(self, samples):
        for guid, sample in enumerate(samples):
            # split into tokens
            tokens_and_annotations = sample.split()
            tokens, ner_tags = [], []

            cur_tag = "O"
            # parse annotations
            for token_or_annotation in tokens_and_annotations:
                
                if token_or_annotation in '([{':
                    # check if is begin annotation -> set current tag
                    i = '([{'.index(token_or_annotation)
                    cur_tag = ["B-DISORDER/FINDING", "B-DRUG", "B-BODY"][i]
                elif token_or_annotation in ')]}':    
                    # check if is end annotation -> reset current tag
                    cur_tag = "O"
                else:
                    # its an actual token :)
                    tokens.append(token_or_annotation)
                    ner_tags.append(cur_tag)
                    # advance current tag
                    if cur_tag.startswith('B-'):
                        cur_tag = 'I-' + cur_tag[2:]

            yield guid, {
                'tokens': tokens,
                'ner_tags': ner_tags
            }
