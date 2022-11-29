import torch
from ignite.metrics import Metric
from seqeval.metrics import classification_report
from itertools import chain
from typing import Tuple, List

class SeqEvalMetrics(Metric):

    def __init__(self, label_space:List[str], **kwargs) -> None:
        # initialize metric
        super(SeqEvalMetrics, self).__init__(**kwargs)
        # save label space
        self.label_space = label_space

    def build_y_pred_and_true(
        self, 
        logits:torch.FloatTensor, 
        labels:torch.LongTensor,
        mask:torch.BoolTensor
    ) -> Tuple[List[List[str]], List[List[str]]]:
        # compute predictions
        y_pred = logits.argmax(dim=-1)
        y_true = labels
        # check shapes (note this doesn't apply to nested setup)
        assert y_true.ndim == y_pred.ndim == 2
        assert y_true.size(0) == y_pred.size(0)
        assert y_true.size(1) == y_pred.size(1)
        # convert all to lists
        mask = mask.cpu().tolist()
        y_pred = y_pred.cpu().tolist()
        y_true = y_true.cpu().tolist()

        # apply mask and label space
        y_pred = [
            [self.label_space[i] for i, v in zip(y_pred_seq, mask_seq) if v]
            for y_pred_seq, mask_seq in zip(y_pred, mask)
        ]
        y_true = [
            [self.label_space[i] for i, v in zip(y_true_seq, mask_seq) if v]
            for y_true_seq, mask_seq in zip(y_true, mask)
        ]
        # make sure targets and predictions align
        return y_pred, y_true

    def reset(self) -> None:
        self.y_true = []
        self.y_pred = []

    def update(self, out) -> None:
        y_pred, y_true = self.build_y_pred_and_true(*out)
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
    
    def compute(self):
        # compute metrics
        metrics = classification_report(
            y_true=list(chain(*self.y_true)),
            y_pred=list(chain(*self.y_pred)),
            output_dict=True
        )
        # post-process metrics dict
        return {
            "%s/%s" % (entity_type, metric_type[0].upper()): metric_value
            for entity_type, metrics_per_entity in metrics.items()
            for metric_type, metric_value in metrics_per_entity.items()
        }

    def completed(self, engine, name):
        # overwrite default behaviour of writing metrics to
        # engine state
        metrics = self.compute()
        for metric_name, metric_value in metrics.items():
            engine.state.metrics["%s/%s" % (name, metric_name)] = metric_value


class NestedSeqEvalMetrics(SeqEvalMetrics):

    def __init__(self, entity_types:List[str], **kwargs):

        self.entity_types = entity_types
        # build label space
        # make sure that the order of the BIO tags matches
        # the order specified in the dataset 
        label_space = (["O", "B-%s" % n, "I-%s" % n] for n in entity_types)
        label_space = list(chain(*label_space))

        # initialize metric
        super(NestedSeqEvalMetrics, self).__init__(label_space=label_space, **kwargs)
    
        # build entity dimension offsets
        self.offsets = 3 * torch.arange(self.num_entities)

    @property
    def num_entities(self) -> int:
        return len(self.entity_types)

    def build_y_pred_and_true(
        self, 
        logits:torch.FloatTensor, 
        labels:torch.LongTensor,
        mask:torch.BoolTensor
    ) -> Tuple[List[List[str]], List[List[str]]]:
        # compute predictions
        y_pred = logits.argmax(dim=-1)
        y_true = labels 
        # check shapes
        assert y_true.ndim == y_pred.ndim == 3
        assert y_true.size(1) == y_pred.size(1) # sequence length
        assert y_true.size(2) == y_pred.size(2) == self.num_entities
        # mask is also three dimensional but the last dimensions
        # corresponding to the entity types should be redundant
        # as only the sequences are masked
        assert (mask[:, :, 0] == mask.all(dim=-1)).all()
        mask = mask[:, :, 0]
        # apply entity offsets as preparation for
        # lookup in label space
        y_true[mask, :] += self.offsets.reshape(1, -1)
        y_pred[mask, :] += self.offsets.reshape(1, -1)
        
        # re-organize to nested lists of sequences
        y_true = y_true.transpose(1, 2).cpu().tolist()
        y_pred = y_pred.transpose(1, 2).cpu().tolist()
        mask = mask.cpu().tolist() # shape (B, S)
        # flatten and apply label space
        flat_y_true = [
            [self.label_space[i] for i, v in zip(y_true_one_entity, mask_for_seq) if v]
            for y_true_all_entities, mask_for_seq in zip(y_true, mask)
            for y_true_one_entity in y_true_all_entities
        ]
        flat_y_pred = [
            [self.label_space[i] for i, v in zip(y_pred_one_entity, mask_for_seq) if v]
            for y_pred_all_entities, mask_for_seq in zip(y_pred, mask)
            for y_pred_one_entity in y_pred_all_entities
        ]

        return flat_y_true, flat_y_pred
