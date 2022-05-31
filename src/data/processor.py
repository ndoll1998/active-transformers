import datasets
from transformers import PreTrainedTokenizer
from itertools import chain

class Conll2003Processor(object):

    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        dataset_info:datasets.DatasetInfo,
        tags_field:str,
        max_length:int,
        tag_pad_token_id:int =-100
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset_info = dataset_info
        self.max_length = max_length
        self.tags_field = tags_field
        self.tag_pad_token_id = tag_pad_token_id

    @property
    def num_labels(self) -> int:
        return len(self.dataset_info.features[self.tags_field].feature.names)

    @property
    def begin2in(self):
        # get bio labels and extract classes from it
        bio = self.dataset_info.features[self.tags_field].feature
        classes = set([label[2:] for label in bio.names if len(label) > 2])
        # map begin label-id to in label-id per class
        return {bio.str2int("B-%s" % c): bio.str2int("I-%s" % c) for c in classes}

    def __call__(self, item:dict) -> dict:

        # tokenize
        tokens = [self.tokenizer.tokenize(token) for token in item['tokens']]
        # build bio tags on sub-token level
        b2i = self.begin2in
        tags = [
            [tag] + [b2i.get(tag, tag)] * (len(ts) - 1) 
            for tag, ts in zip(item[self.tags_field], tokens)
        ]
        # concatenate sub-tokens and tags
        tokens = list(chain(*tokens))
        tags = list(chain(*tags))
        # check sizes
        assert len(tokens) == len(tags), "Mismatch between tokens (%i) and tags (%i)" % (len(tokens), len(tags))

        # build input ids and attention mask
        input_ids = self.tokenizer.encode(
            text=tokens, 
            add_special_tokens=True, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        attention_mask = [token_id != self.tokenizer.pad_token_id for token_id in input_ids]

        # pad tags to match max-sequence-length
        tags = tags[:self.max_length] + [self.tag_pad_token_id] * max(0, self.max_length - len(tags))

        # return all features
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': tags
        }
