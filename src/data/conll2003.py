import datasets

logger = datasets.logging.get_logger(__name__)

_POS_TAGS = ['$.', "$,", "$(", 'PRELS', 'PDS', 'ITJ', 'VVFIN', 'VVIMP', 'PWAV', 'ADV', 'PTKVZ', 'VAIMP', 'APPRART', 'TRUNC', 'CARD', 'KON', 'VMINF', 'KOUI', 'PWS', 'PTKANT', 'VVPP', 'APZR', 'XY', 'APPR', 'PIAT', 'ADJD', 'PTKZU', 'VAINF', 'PPER', 'PPOSAT', 'PIS', 'PIDAT', 'VAFIN', 'FM', 'PRF', 'KOKOM', 'PWAT', 'PDAT', 'APPO', 'ART', 'VVINF', 'ADJA', 'VMFIN', 'VVIZU', 'VAPP', 'PTKNEG', 'PPOSS', 'PRELAT', 'NE', 'PTKA', 'PAV', 'NN', 'KOUS', 'VMPP']
_CHUNK_TAGS = ['O', 'I-VC', 'B-NC', 'I-PC', 'B-VC', 'B-PC', 'I-NC']
_NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

_TRAIN_URL = "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.train"
_DEV_URL = "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testa"
_TEST_URL = "https://raw.githubusercontent.com/MaviccPRP/ger_ner_evals/master/corpora/conll2003/deu.testb"

class GermanConll2003(datasets.GeneratorBasedBuilder):
    """ Dataset Generator for German Conll2003 Dataset """

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
            features=datasets.Features({
                "tokens": datasets.Sequence(datasets.Value("string")),
                "pos_tags": datasets.Sequence(datasets.features.ClassLabel(names=_POS_TAGS)),
                "chunk_tags": datasets.Sequence(datasets.features.ClassLabel(names=_CHUNK_TAGS)),
                "ner_tags": datasets.Sequence(datasets.features.ClassLabel(names=_NER_TAGS)),
            }),
            supervised_keys=None,
            homepage="",
            citation=""
        )

    def _split_generators(self, dl_manager):

        # download all files
        train_fpath, dev_fpath, test_fpath = dl_manager.download([_TRAIN_URL, _DEV_URL, _TEST_URL])
        # build generators
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'fpath': train_fpath}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={'fpath': dev_fpath}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'fpath': test_fpath})
        ]

    def _generate_examples(self, fpath):
        logger.info("Generating Examples from %s" % fpath)

        with open(fpath, 'r', encoding='latin-1') as f:

            guid = 0
            item = {'tokens': [], 'pos_tags': [], 'chunk_tags': [], 'ner_tags': []}

            for line in map(str.strip, f):

                if line.startswith('-DOCSTART-') or len(line) == 0:
                    # check if item is non-empty
                    if len(item['tokens']) > 0:
                        # yield item
                        yield guid, item
                        # reset item
                        guid += 1
                        item = {'tokens': [], 'pos_tags': [], 'chunk_tags': [], 'ner_tags': []}

                else:
                    # update current item
                    token, _, pos_tag, chunk_tag, ner_tag = line.split()
                    item['tokens'].append(token)
                    item['pos_tags'].append(pos_tag)
                    item['chunk_tags'].append(chunk_tag)
                    item['ner_tags'].append(ner_tag)
             
            # yield last item
            if len(item['tokens']) > 0:
                yield guid, item

