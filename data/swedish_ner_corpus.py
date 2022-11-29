import datasets
from data.conll_format_builder import ConllFormatBuilder

_DESCRIPTION = """\
Webbnyheter 2012 from Spraakbanken, semi-manually annotated and adapted for CoreNLP Swedish NER. Semi-manually defined in this case as: Bootstrapped from Swedish Gazetters then manually correcte/reviewed by two independent native speaking swedish annotators. No annotator agreement calculated.
"""
_HOMEPAGE_URL = "https://github.com/klintan/swedish-ner-corpus"
_TRAIN_URL = "https://raw.githubusercontent.com/klintan/swedish-ner-corpus/master/train_corpus.txt"
_TEST_URL = "https://raw.githubusercontent.com/klintan/swedish-ner-corpus/master/test_corpus.txt"
_CITATION = None

class SwedishNerCorpus(ConllFormatBuilder):
    VERSION = datasets.Version("1.0.0")

    FORMAT=["tokens", "entity_tags"]
    FEATURE_TYPES={
        'tokens': datasets.Value("string"),
        'entity_tags': datasets.ClassLabel(names=[
            "O", 
            "LOC", 
            "MISC", 
            "ORG", 
            "PER",
        ]),
        'ner_tags': datasets.ClassLabel(names=[
            "O", 
            "B-LOC", 
            "I-LOC", 
            "B-MISC", 
            "I-MISC", 
            "B-ORG", 
            "I-ORG", 
            "B-PER",
            "I-PER"
        ])
    }


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self._features,
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_URL)
        test_path = dl_manager.download_and_extract(_TEST_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"fpaths": [train_path]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"fpaths": [test_path]},
            ),
        ]

    def _generate_examples(self, **kwargs):

        for guid, item in super(SwedishNerCorpus, self)._generate_examples(**kwargs):
          
            # convert out tag '0' -> 'O'
            item['entity_tags'] = ["O" if tag == "0" else tag for tag in item['entity_tags']]
            # build bio tags by adding begin/in prefixes
            ner_tags = []
            prev_entity_tag = "O"
            for entity_tag in item['entity_tags']:
                # get bio tag
                bio_tag = "O" if entity_tag == "O" else \
                    ("B-" + entity_tag) if entity_tag != prev_entity_tag else \
                    ("I-" + entity_tag)
                # add to ner tags and update previous tag
                ner_tags.append(bio_tag)
                prev_entity_tag = entity_tag

            assert len(item['tokens']) == len(ner_tags)            
            # update and yield item
            item['ner_tags'] = ner_tags
            yield guid, item
