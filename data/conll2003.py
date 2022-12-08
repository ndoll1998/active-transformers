import datasets
from data.conll_format_builder import ConllFormatBuilder

_POS_TAGS = ['$.', "$,", "$(", 'PRELS', 'PDS', 'ITJ', 'VVFIN', 'VVIMP', 'PWAV', 'ADV', 'PTKVZ', 'VAIMP', 'APPRART', 'TRUNC', 'CARD', 'KON', 'VMINF', 'KOUI', 'PWS', 'PTKANT', 'VVPP', 'APZR', 'XY', 'APPR', 'PIAT', 'ADJD', 'PTKZU', 'VAINF', 'PPER', 'PPOSAT', 'PIS', 'PIDAT', 'VAFIN', 'FM', 'PRF', 'KOKOM', 'PWAT', 'PDAT', 'APPO', 'ART', 'VVINF', 'ADJA', 'VMFIN', 'VVIZU', 'VAPP', 'PTKNEG', 'PPOSS', 'PRELAT', 'NE', 'PTKA', 'PAV', 'NN', 'KOUS', 'VMPP']
_CHUNK_TAGS = ['O', 'I-VC', 'B-NC', 'I-PC', 'B-VC', 'B-PC', 'I-NC']
_NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

_TRAIN_URL = "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.train"
_DEV_URL = "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testa"
_TEST_URL = "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testb"

class GermanConll2003(ConllFormatBuilder):
    """ Dataset Generator for German Conll2003 Dataset """

    FORMAT=['tokens', None, 'pos_tags', 'chunk_tags', 'ner_tags']
    FEATURE_TYPES={
        'tokens': datasets.Value("string"),
        'pos_tags': datasets.ClassLabel(names=_POS_TAGS),
        'chunk_tags': datasets.ClassLabel(names=_CHUNK_TAGS),
        'ner_tags': datasets.ClassLabel(names=_NER_TAGS)
    }

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="german_conll2003",
            version=datasets.Version("0.1.0"),
            description="German Conll2003 dataset"
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

        # download all files
        train_fpath, dev_fpath, test_fpath = dl_manager.download([_TRAIN_URL, _DEV_URL, _TEST_URL])
        # build generators
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'fpaths': [train_fpath]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={'fpaths': [dev_fpath]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'fpaths': [test_fpath]})
        ]
    
    def _generate_examples(self, **kwargs):

        for guid, item in super(GermanConll2003, self)._generate_examples(**kwargs):
            # replace initial entity tag with corresponding in tag
            prev_tag = "O"
            for i, tag in enumerate(item['ner_tags']):
                # check if tag marks beginning of a new entity
                if (tag != prev_tag) and (tag != 'O'):
                    item['ner_tags'][i] = "B-" + tag[2:]
                # update previous tag
                prev_tag = tag

            # yield updated item
            yield guid, item

