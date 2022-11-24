import os
import json
import datasets

logger = datasets.logging.get_logger(__name__)

# google drive download link
_ARCHIVE_URL = "https://drive.google.com/uc?export=download&id=1i37ZmJAofKXuOJbq1nG5kqPTM081WfXQ"
_ENTITY_TYPES = ["RNA", "DNA", "protein", "cell_type", "cell_line"]

class GeniaCorpus(datasets.GeneratorBasedBuilder):
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="genia_nested_ner",
            version=datasets.Version("0.1.0"),
            description="Genia Corpus for Nested Named-Entity-Recognition"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Genia Corpus for Nested Named-Entity-Recognition",
            features=datasets.Features({
                "tokens": datasets.Sequence(datasets.Value("string")),
                "ner_tags": {
                    entity_type: datasets.Sequence(datasets.ClassLabel(names=["O", "B", "I"]))
                    for entity_type in _ENTITY_TYPES
                }
            }),
            supervised_keys=None,
            homepage="",
            citation=""
        )

    def _split_generators(self, dl_manager):

        # download and unzip
        archive_path = dl_manager.download_and_extract(_ARCHIVE_URL)
        archive_path = os.path.join(archive_path, "genia91")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={'fpath': os.path.join(archive_path, "genia_train_dev_context.json")}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={'fpath': os.path.join(archive_path, "genia_test_context.json")}
            )
        ]

    def _generate_examples(self, fpath):
        logger.info("Generating Examples from %s" % fpath)
        # load raw data
        with open(fpath, 'r') as f:
            data = json.loads(f.read())
        # prepare and yield items
        for guid, item in enumerate(data):
            # build ner-tags for each entity type
            ner_tags = {entity_type: ["O"] * len(item['tokens']) for entity_type in _ENTITY_TYPES}
            for entity_type in _ENTITY_TYPES:
                # collect all entities of the current type
                # and annotate them in the corresponding ner tags
                entities = filter(lambda e: e['type'] == entity_type, item['entities'])
                for entity in entities:
                    i, j = entity['start'], entity['end']
                    ner_tags[entity_type][i] = "B"
                    ner_tags[entity_type][1+i:j] = ["I"] * (j-i-1)

            # yield prepared item
            yield guid, {
                'tokens': item['tokens'],
                'ner_tags': ner_tags
            }
