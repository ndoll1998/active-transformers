import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.distributed as idist
# base strategy and utils
from .strategy import ScoreBasedStrategy
from ..utils.data import move_to_device
from ..utils.gradnorm import GoodfellowGradientNorm
from ..utils.sequence import topk_sequences
# import transformers and others
from transformers import PreTrainedModel
from abc import abstractmethod
from typing import Tuple, Sequence, Any, Union, Optional

__all__ = [
    "EglByTopK",
    "EglBySampling"
]

class _EglBase(ScoreBasedStrategy):
    """ Abstract Base for Maximum Expected Gradient Length Sampling 
        Strategy. Assumes cross-entropy loss. Child classes must
        overwrite the abstract method `_get_hallucinated_labels`
        which generates the labels for gradient computations.
    
        Args:
            model (PreTrainedModel): 
                transformer-based sequence classification model
            random_sample (Optional[bool]): 
                whether to random sample query according to distribution given by expected gradient length.
                Defaults to False, meaning the elements with maximum expected gradient langth are selected.
    """

    def __init__(
        self,
        model:PreTrainedModel,
        random_sample:Optional[bool] =False
    ) -> None:
        # initialize strategy
        super(_EglBase, self).__init__(
            random_sample=random_sample
        )
        # move model to available device(s)
        self.model = idist.auto_model(model)

    @abstractmethod
    def _get_hallucinated_labels(
        self, 
        logits:torch.FloatTensor,
        mask:Union[torch.BoolTensor, None]
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """ Get hallucinated labels for which to compute the loss and gradients.
        
            Args:
                logits (torch.FloatTensor): 
                    logits of shape (n, m) where n is the number of examples 
                    and m is the number of labels. For Token Classification tasks
                    the shape is (n, s, m) where s is the sequence length.
                mask (Union[torch.BoolTensor, None]):
                    attention mask of shape (n, s, m). None if no attention mask
                    is provided by the batch.

            Returns:
                probs (torch.FloatTensor):
                    probabilities of the specific label infered from the logits.
                    Must be of shape (n, k) where k is the number of hallucinated
                    labels to choose.
                labels (torch.LongTensor):
                    hallucinated labels selected for loss computation. Must be of the
                    shape (n, k). Again k refers to the number of hallucinated labels.
                    For Token Classification tasks the shape must be (n, s, k).
        """
        raise NotImplementedError()
    
    def process(self, batch:Any) -> torch.FloatTensor:
        """ Compute expected gradient lengths for given batch 
        
            Args:
                batch (Any): input batch for which to compute grad norms
        
            Returns:
                norm (torch.FloatTensor): (squared) gradient norm length computed following Goodfellow (2016)
        """
        self.model.eval()
        # create gradient norm tracking context
        with GoodfellowGradientNorm(self.model) as gradnorm:
            # move batch to device and pass through model
            batch = move_to_device(batch, device=idist.device())
            logits = self.model(**batch).logits
            # get top-k predictions
            mask = batch.get('attention_mask', None)
            probs, labels = self._get_hallucinated_labels(logits, mask)
            probs, labels = probs.to(idist.device()), labels.to(idist.device())
            # check dimensions
            assert probs.size(0) == labels.size(0) == logits.size(0), \
                "Mismatch between batchsizes of probs (%i), labels (%i) and logits (%i)" % \
                    (probs.size(0), labels.size(0), logits.size(0))
            assert probs.size(-1) == labels.size(-1), \
                 "Mismatch between selected labels of probs (%i) and labels (%i)" % \
                 (probs.size(-1), labels.size(-1))
            assert probs.ndim == 2, \
                "Expected probabilities two have two dimensions but got shape $s" % \
                str(tuple(probs.size()))
            # compute loss and backpropagate
            for i in range(labels.size(-1)):
                F.cross_entropy(
                    logits.flatten(end_dim=-2), 
                    labels[..., i].flatten(), 
                    reduction='sum'
                ).backward(retain_graph=i < labels.size(-1)-1)
            # approximate expected gradient length
            with torch.no_grad():
                return torch.sum(probs * gradnorm.compute(), dim=1)


class EglByTopK(_EglBase):
    """ Implements Maximum Expected Gradient Length Sampling Strategy
        by selecting the top-k predictions as labels for gradient 
        computation. Assumes cross-entropy loss.
    
        Args:
            model (PreTrainedModel): 
                transformer-based sequence classification model
            k (int):
                maximum number of targets to consider for
                expectation approximation. Defaults to 5.
    """
    
    def __init__(
        self,
        model:PreTrainedModel,
        k:int =5
    ) -> None:
        # initialize strategy
        super(EglByTopK, self).__init__(model)
        self.k = k   
 
    @torch.no_grad()
    def _get_hallucinated_labels(
        self, 
        logits:torch.FloatTensor, 
        mask:Union[torch.BoolTensor, None]
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """ Get hallucinated labels for which to compute the loss and gradients.
            Selects the top-k predictions as hallucinated labels.
        
            Args:
                logits (torch.FloatTensor): 
                    logits of shape (n, m) where n is the number of examples 
                    and m is the number of labels. For Token Classification tasks
                    the shape is (n, s, m) where s is the sequence length.
                mask (Union[torch.BoolTensor, None]):
                    attention mask of shape (n, s, m). None if no attention mask
                    is provided by the batch.

            Returns:
                probs (torch.FloatTensor):
                    probabilities of the specific label infered from the logits.
                    Must be of shape (n, k) where k is the number of hallucinated
                    labels to choose.
                labels (torch.LongTensor):
                    hallucinated labels selected for loss computation. Must be of the
                    shape (n, k). Again k refers to the number of hallucinated labels.
                    For Token Classification tasks the shape must be (n, s, k).
        """
        probs = torch.softmax(logits, dim=-1) 
        
        if logits.ndim == 2:
            # sequence classification
            k = min(self.k, logits.size(-1))
            return torch.topk(probs, k=k, dim=-1)
        
        if logits.ndim == 3:
            # token classification
            n, s, _ = logits.size()
            lengths = torch.full(n, fill_value=s) if mask is None else mask.sum(dim=1)
            # put probs to cpu
            probs = probs.cpu()
            # get top-k sequences
            posteriors = torch.zeros((n, self.k), dtype=float)
            labels = torch.zeros((n, s, self.k), dtype=int)
            # build top-k sequences for batch one by one
            # this is very slow but not easily vectorizable
            for i, j in enumerate(lengths):
                # get valid sub-matrix for faster top-k compuations
                p, l = topk_sequences(probs[i, :j, :], k=self.k)
                # handle less than k sequence generated due
                # note that posteriors are initialized to zeros
                # meaning invalid sequences are irrelevant
                k_ = p.size(0)                
                posteriors[i, :k_] = p
                labels[i, :j, :k_] = l.t()
            # return probabilities and labels
            return posteriors, labels
        
        raise NotImplementedError()

class EglBySampling(_EglBase):
    """ Implements Maximum Expected Gradient Length Sampling Strategy
        by random sampling labels from the predicted label distribution.
        Assumes cross-entropy loss.
    
        Args:
            model (PreTrainedModel): 
                transformer-based sequence classification model
            k (int):
                maximum number of targets to consider for
                expectation approximation. Defaults to 5.
    """
    
    def __init__(
        self,
        model:PreTrainedModel,
        k:int =5
    ) -> None:
        # initialize strategy
        super(EglBySampling, self).__init__(model)
        self.k = k   

    @torch.no_grad()
    def _get_hallucinated_labels(
        self, 
        logits:torch.FloatTensor, 
        mask:Union[torch.BoolTensor, None]
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """ Get hallucinated labels for which to compute the loss and gradients.
            Samples labels from distribution defined by logits.
            
            Args:
                logits (torch.FloatTensor): 
                    logits of shape (n, m) where n is the number of examples 
                    and m is the number of labels. For Token Classification tasks
                    the shape is (n, s, m) where s is the sequence length.
                mask (Union[torch.BoolTensor, None]):
                    attention mask of shape (n, s, m). None if no attention mask
                    is provided by the batch.

            Returns:
                probs (torch.FloatTensor):
                    probabilities of the specific label infered from the logits.
                    Must be of shape (n, k) where k is the number of hallucinated
                    labels to choose.
                labels (torch.LongTensor):
                    hallucinated labels selected for loss computation. Must be of the
                    shape (n, k). Again k refers to the number of hallucinated labels.
                    For Token Classification tasks the shape must be (n, s, k).
        """
        # compute probabilities
        probs = torch.softmax(logits, dim=-1) 
        # sample labels from logits
        labels = torch.multinomial(
            probs.flatten(end_dim=-2),
            num_samples=self.k,
            replacement=True
        ).reshape(*logits.size()[:-1], self.k)
        # token classification, apply mask to disable loss for invalid tokens
        if (mask is not None) and (logits.ndim == 3):
            labels.masked_fill_(~mask.bool().unsqueeze(-1), -100)
        # weight each label the same, weighting is done by sampling
        return torch.ones((logits.size(0), self.k), device=labels.device), labels
