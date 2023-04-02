import torch
import numpy as np
# metrics backbones
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report
# helpers
from itertools import chain
from transformers import EvalPrediction
from typing import List, Dict, Optional, Union

class PRFS(object):
    """ Compute Precision-Recall-FScore-Support Metrics """

    def __init__(self,
        label_space:List[str],
        beta:float =1.0,
        prefix:Optional[str] =None,
        zero_division:Union[str, float] =0
    ) -> None:
        self.label_space = label_space
        self.beta = beta
        self.prefix = ("%s_" % prefix) if prefix is not None else ""
        self.zero_division = zero_division

    @torch.no_grad()
    def __call__(self, eval_pred:EvalPrediction) -> Dict[str, float]:
        assert isinstance(eval_pred.predictions, np.ndarray)
        assert isinstance(eval_pred.label_ids, np.ndarray)
        # compute metrics
        mask = (eval_pred.label_ids >= 0)
        P, R, F, S = precision_recall_fscore_support(
            eval_pred.predictions[mask],
            eval_pred.label_ids[mask],
            beta=self.beta,
            zero_division=self.zero_division
        )
        # build metrics dict
        return {
            "%s%s_%s" % (self.prefix, label, metric): v[i]
            for v, metric in zip([P, R, F, S], "PRFS")
            for i, label in enumerate(self.label_space)
        } | {
            # macro averages
            "%smacro-avg_P" % self.prefix: P.mean(),
            "%smacro-avg_R" % self.prefix: R.mean(),
            "%smacro-avg_F" % self.prefix: F.mean(),
            "%smacro-avg_S" % self.prefix: S.sum(),
            # weighted averages
            "%sweighted-avg_P" % self.prefix: np.dot(P, S) / S.sum(),
            "%sweighted-avg_R" % self.prefix: np.dot(R, S) / S.sum(),
            "%sweighted-avg_F" % self.prefix: np.dot(F, S) / S.sum(),
            "%sweighted-avg_S" % self.prefix: S.sum(),
        } | dict(zip(
            # micro averages
            [
                "%smicro-avg_P" % self.prefix,
                "%smicro-avg_R" % self.prefix,
                "%smicro-avg_F" % self.prefix,
                "%smicro-avg_S" % self.prefix
            ],
            precision_recall_fscore_support(
                eval_pred.predictions[mask],
                eval_pred.label_ids[mask],
                beta=self.beta,
                average='micro',
                zero_division=self.zero_division
            )
        ))


class SeqEval(object):

    def __init__(self,
        label_space:List[str],
        prefix:Optional[str] =None,
        zero_division:Union[str, float] =0
    ) -> None:
        self.label_space = label_space
        self.prefix = ("%s_" % prefix) if prefix is not None else ""
        self.zero_division = zero_division

    def build_pred_and_true(self, eval_pred:EvalPrediction):
        # get predicitons and labels
        y_pred, y_true = eval_pred.predictions, eval_pred.label_ids
        # check shapes (note this doesn't apply to nested setup)
        assert y_true.ndim == y_pred.ndim == 2
        assert y_true.shape[0] == y_pred.shape[0]
        assert y_true.shape[1] == y_pred.shape[1]
        # convert all to lists
        mask = (y_true >= 0).tolist()
        y_pred = y_pred.tolist()
        y_true = y_true.tolist()
        # apply mask and label space
        y_pred = [
            [self.label_space[i] for i, v in zip(y_pred_seq, mask_seq) if v]
            for y_pred_seq, mask_seq in zip(y_pred, mask)
        ]
        y_true = [
            [self.label_space[i] for i, v in zip(y_true_seq, mask_seq) if v]
            for y_true_seq, mask_seq in zip(y_true, mask)
        ]
        # return predicitons and targets
        return y_pred, y_true

    @torch.no_grad()
    def __call__(self, eval_pred:EvalPrediction) -> Dict[str, float]:
        assert isinstance(eval_pred.predictions, np.ndarray)
        assert isinstance(eval_pred.label_ids, np.ndarray)
        # build predictions and targets
        y_pred, y_true = self.build_pred_and_true(eval_pred)
        # compute metrics
        metrics = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            zero_division=self.zero_division
        )
        # post-process metrics dict
        return {
            "%s%s_%s" % (
                self.prefix,
                entity_type.replace(' ', '-'),
                metric_type[0].upper()
            ): metric_value
            for entity_type, metrics_per_entity in metrics.items()
            for metric_type, metric_value in metrics_per_entity.items()
        }

class NestedSeqEval(SeqEval):

    def __init__(self,
        entity_types:List[str],
        prefix:Optional[str] =None,
        zero_division:Union[str, float] =0
    ) -> None:

        self.entity_types = entity_types
        # build label space
        # make sure that the order of the BIO tags matches
        # the order specified in the dataset 
        label_space = (["O", "B-%s" % n, "I-%s" % n] for n in entity_types)
        label_space = list(chain(*label_space))
        # initialize metric
        super(NestedSeqEval, self).__init__(
            label_space=label_space,
            prefix=prefix,
            zero_division=zero_division
        )

        # build entity dimension offsets
        self.offsets = 3 * np.arange(self.num_entities)

    @property
    def num_entities(self) -> int:
        return len(self.entity_types)

    def build_pred_and_true(self, eval_pred:EvalPrediction):
        # get predicitons and labels
        y_pred, y_true = eval_pred.predictions, eval_pred.label_ids
        mask = (y_true >= 0)
        # check shapes
        assert y_true.ndim == y_pred.ndim == 3
        assert y_true.shape[1] == y_pred.shape[1] # sequence length
        assert y_true.shape[2] == y_pred.shape[2] == self.num_entities
        # mask is also three dimensional but the last dimensions
        # corresponding to the entity types should be redundant
        # as only the sequences are masked
        assert (mask[:, :, 0] == mask.all(axis=-1)).all()
        mask = mask[:, :, 0]
        # apply entity offsets as preparation for
        # lookup in label space
        y_true[mask, :] += self.offsets.reshape(1, -1)
        y_pred[mask, :] += self.offsets.reshape(1, -1)

        # re-organize to nested lists of sequences
        y_true = y_true.swapaxes(1, 2).tolist()
        y_pred = y_pred.swapaxes(1, 2).tolist()
        mask = mask.tolist() # shape (B, S)
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

        return flat_y_pred, flat_y_true
