import json
import datasets
import numpy as np
import pandas as pd
from itertools import chain
from collections import defaultdict
from active.scripts.run_train import Task, ExperimentConfig

def analyse_dataset(config):
    print("Analysing %s..." % config.dataset)

    # first of all load the dataset
    train_data, test_data = datasets.load_dataset(
        config.dataset, 
        split=["train", "test"]
    )

    # statistics over input size, i.e. number of input tokens
    num_tokens = np.asarray(list(map(len, train_data[config.text_column])))
    avg_num_tokens = num_tokens.mean()
    std_num_tokens = num_tokens.std()

    num_entities = []
    entity_lengths_per_type = defaultdict(list)

    if config.task is Task.BIO_TAGGING:
     
        # statistics over entities
        label_space = config.label_space

        entity_types = [
            tag[len(config.begin_tag_prefix):] for tag in label_space
            if tag.startswith(config.begin_tag_prefix)
        ]
        begin_tag_ids = [label_space.index(config.begin_tag_prefix + etype) for etype in entity_types]
        in_tag_ids = [label_space.index(config.in_tag_prefix + etype) for etype in entity_types]

        for labels in train_data[config.label_column]:
           
            # count number of entities in current example by counting the
            # number of begin tags
            num_entities.append(sum(labels.count(tag_id) for tag_id in begin_tag_ids))

            while len(labels) > 0:
                i = labels.pop(0)
                
                # find entities
                if i in begin_tag_ids:
                    in_tag = in_tag_ids[begin_tag_ids.index(i)]
                    entity_type = entity_types[begin_tag_ids.index(i)]
                   
                    # compute the length of the entity, i.e.
                    # the number of tokens it spans over
                    entity_length = 1 
                    while len(labels) > 0 and (labels[0] == in_tag):
                        labels.pop(0)
                        entity_length += 1

                    entity_lengths_per_type[entity_type].append(entity_length)

    elif config.task is Task.NESTED_BIO_TAGGING:

        for labels_per_type in train_data[config.label_column]:
            
            # count the number of entities in the current example by counting the
            # number of begin tags
            num_entities.append(sum(labels.count(1) for labels in labels_per_type.values()))

            for entity_type, labels in labels_per_type.items():

                while len(labels) > 0:
                    i = labels.pop(0)

                    # find begin-tag
                    if i == 1:
                        entity_length = 1
                        # count number of following in-tags
                        while len(labels) > 0 and (labels[0] == 2):
                            labels.pop(0)
                            entity_length += 1
                        entity_lengths_per_type[entity_type].append(entity_length)
                        
    else:
        raise NotImplementedError()

    # number of entities per type
    num_entities_per_type = {etype: len(elengths) for etype, elengths in entity_lengths_per_type.items()}

    # average number of entities per sample
    avg_num_entities = np.mean(num_entities)
    std_num_entities = np.std(num_entities)
    
    # average size of entity
    avg_entity_length = np.mean(list(chain(*entity_lengths_per_type.values())))
    std_entity_length = np.std(list(chain(*entity_lengths_per_type.values())))

    return {
        "task": config.task.value,
        "#train": len(train_data),
        "#test": len(test_data),
        "#entities": sum(num_entities_per_type.values()),
        "avg #tokens": avg_num_tokens,
        "std #tokens": std_num_tokens,
        "avg #entities": avg_num_entities,
        "std #entities": std_num_entities,
        "avg len(entities)": avg_entity_length,
        "std len(entities)": std_entity_length
    }


if __name__ == '__main__':

    import os
    configs = [os.path.join("configs", fname) for fname in os.listdir("configs/") if fname.endswith('.json')]
    # load all configs and filter for bio-tasks
    configs = [ExperimentConfig.parse_file(fpath) for fpath in configs]
    configs = [config for config in configs if config.task is Task.BIO_TAGGING] + \
        [config for config in configs if config.task is Task.NESTED_BIO_TAGGING]

    # for now only bio-tagging datasets are supported
    assert all(config.task in (Task.BIO_TAGGING, Task.NESTED_BIO_TAGGING) for config in configs)

    # build statistics dataframe
    df = pd.DataFrame(
        data=[analyse_dataset(config.data) for config in configs],
        index=[config.name for config in configs]
    )

    print(df)
