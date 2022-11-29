import torch
from transformers import PreTrainedTokenizer
from typing import List

class SequenceClassificationProcessor(object):

    def __init__(
        self, 
        tokenizer:PreTrainedTokenizer,
        max_length:int,
        text_column:str,
        label_column:str,
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

    def __call__(self, example):
        return self.tokenizer(
            text=example[self.text_column],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
        ) | {'labels': example[self.label_column]}


class BioTaggingProcessor(object):

    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        max_length:int,
        text_column:str,
        label_column:str,
        label_space:List[str],
        begin_tag_prefix:str,
        in_tag_prefix:int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
       
        # map label to id
        label2id = dict(zip(label_space, range(len(label_space))))
        # map begin/in tags to corresponding entities
        begin_tags = {tag: tag[len(begin_tag_prefix):] for tag in label_space if tag.startswith(begin_tag_prefix)}
        in_tags = {tag: tag[len(in_tag_prefix):] for tag in label_space if tag.startswith(in_tag_prefix)}

        # make sure there is a in-tag for each begin-tag and vise versa
        assert set(begin_tags.values()) == set(in_tags.values())
        # make sure corresponding begin- and in-tags reference the same entity
        assert all(entity == in_tags[tag] for tag, entity in in_tags.items())

        # map begin- to corresponding in-tag
        begin2in = {tag: in_tag_prefix + entity for tag, entity in begin_tags.items()}
        begin2in = [begin2in.get(tag, tag) for tag in label_space]
        begin2in = [label2id[tag] for tag in begin2in]
        # convert to tensor
        # this tensor maps the label-id of begin-tags to the label-id of the
        # corresponding in-tags. Label-ids of non-begin-tags remain untouched.
        # Examples:
        #    - begin2in[label2id["B-ORG"]] = label2id["I-ORG"]
        #    - begin2in[label2id["I-ORG"]] = label2id["I-ORG"]
        self.begin2in = torch.LongTensor(begin2in)

        # check tokenizer is fast
        if not tokenizer.is_fast:
            raise ValueError("Bio tagging processor needs fast pretrained tokenizer but got %s" % type(tokenizer).__name__)

    def __call__(self, example):
        
        # tokenize
        enc = self.tokenizer(
            text=example[self.text_column],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=True
        )

        # extract special tokens mask from encoding
        special_tokens_mask = torch.BoolTensor(enc.special_tokens_mask)

        # create word-ids tensor
        word_ids = torch.LongTensor([i for i in enc.word_ids() if i is not None])
        # create label-ids tensor and apply word-ids
        label_ids = torch.LongTensor(example[self.label_column])
        label_ids = label_ids[word_ids]

        # find tokens marked that are begin-tag but split into
        # word-pieces by tokenizer, only the first word-piece
        # should be annotated with the begin-tag however
        # the above assigns each wordpiece of the token
        # the same tag
        in_mask = (word_ids[:-1] == word_ids[1:])
        # note: this actually marks all wordpieces of
        # the same token except the first one no matter
        # the tag, however only the begin tags are altered
        # by the mapping below
        label_ids[1:][in_mask] = self.begin2in[label_ids[1:][in_mask]]

        # create the bio-tags which bascially are the
        # label-ids above with padding for special tokens
        bio_tags = torch.where(special_tokens_mask, -100, 0)
        bio_tags[~special_tokens_mask] = label_ids
 
        return {
            'input_ids': torch.LongTensor(enc.input_ids),
            'attention_mask': torch.LongTensor(enc.attention_mask),
            'labels': bio_tags
        }

class NestedBioTaggingProcessor(object):
    def __init__(
        self, 
        tokenizer:PreTrainedTokenizer, 
        max_length:int, 
        text_column:str, 
        label_column:str, 
        label_space:List[str],
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        self.entity_types = label_space
        # 
        self.begin2in = torch.LongTensor([0, 2, 2])

    def __call__(self, example):
        
        # tokenize
        enc = self.tokenizer(
            text=example[self.text_column],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=True
        )
        
        # extract special tokens mask from encoding
        special_tokens_mask = torch.BoolTensor(enc.special_tokens_mask)

        # create word-ids tensor
        word_ids = torch.LongTensor([i for i in enc.word_ids() if i is not None])

        # build label ids tensor
        label_ids = torch.LongTensor([example[self.label_column][et] for et in self.entity_types])
        label_ids = label_ids[:, word_ids]
       
        # handle multiple word-pieces annotated as begin 
        in_mask = (word_ids[:-1] == word_ids[1:])
        label_ids[:, 1:][:, in_mask] = self.begin2in[label_ids[:, 1:][:, in_mask]]

        # create bio tags
        bio_tags = torch.where(special_tokens_mask, -100, 0)
        bio_tags = bio_tags.unsqueeze(1).repeat(1, label_ids.size(0))
        bio_tags[~special_tokens_mask, :] = label_ids.t()

        return {
            'input_ids': torch.LongTensor(enc.input_ids),
            'attention_mask': torch.LongTensor(enc.attention_mask),
            'labels': bio_tags
        }
