import datasets
import numpy as np
from transformers import PreTrainedTokenizer
from functools import lru_cache
from itertools import chain

class TokenClassificationProcessor(object):

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
    @lru_cache()
    def begin2in(self) -> dict:
        # get bio labels and extract classes from it
        bio = self.dataset_info.features[self.tags_field].feature
        # check label scheme
        if all(label[:2] in ["O", "O-", "B-", "I-"] for label in bio.names):
            # begin-in marked by 'B-' and 'I-'
            classes = set([label[2:] for label in bio.names if len(label) > 2])
            # map begin label-id to in label-id per class
            return {bio.str2int("B-%s" % c): bio.str2int("I-%s" % c) for c in classes}
        
        # fallback use identity
        return {i: i for i, _ in enumerate(bio.names)}

    @property
    @lru_cache()
    def in2begin(self) -> dict:
        # reverse begin-to-in mapping
        return {i:b for b, i in self.begin2in.items()}

    @property
    def out_tag(self) -> int:
        # TODO: extract out tag from dataset info
        # currently assumes out tag to be zero
        return 0

    def __call__(self, item:dict) -> dict:

        # tokenize to find number of wordpieces per token
        n_subtokens = [len(self.tokenizer.tokenize(token)) for token in item['tokens']]
        
        # cleanup tags
        tags = [
            # continue a previous entity if
            self.begin2in.get(tag, tag) if prev_tag in [
                self.begin2in.get(tag, tag), # the previous tag must be the
                self.in2begin.get(tag, tag)  # corresponding being or in tag
            # or start a new entity
            ] else self.in2begin.get(tag, tag)
            for prev_tag, tag in zip(
                [self.out_tag] + item[self.tags_field][:-1], 
                item[self.tags_field]
            )
        ]

        # build bio tags on sub-token level
        tags = [
            [tag] + [self.begin2in.get(tag, tag)] * (n - 1) 
            for tag, n in zip(item[self.tags_field], n_subtokens)
        ]
        # concatenate sub-token tags
        tags = list(chain(*tags))
        
        # build input ids and attention mask
        encoding = self.tokenizer.encode_plus(
            text=item['tokens'], 
            add_special_tokens=True, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            is_split_into_words=True,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )

        valid_tokens_mask = ~np.asarray(encoding.special_tokens_mask, dtype=bool)
        # build padded tags
        padded_tags = np.full(self.max_length, self.tag_pad_token_id)
        padded_tags[valid_tokens_mask] = tags[:valid_tokens_mask.sum()]

        # return all features
        return {
            'input_ids': encoding.input_ids,
            'attention_mask': encoding.attention_mask,
            'labels': padded_tags.tolist()
        }
