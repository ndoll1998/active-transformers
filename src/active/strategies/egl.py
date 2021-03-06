import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.distributed as idist
# base strategy and utils
from .strategy import AbstractStrategy
from .utils import move_to_device
# import transformers and others
from transformers import PreTrainedModel
from abc import abstractmethod
from typing import Tuple, Sequence, Any, Union

class GoodfellowGradientNormForLinear(object):
    """ Goodfellow Gradient Norm for single linear layer.
        Attaches hooks to the given linear module. Gradient
        Norm is ready for computation after backward call.

        Gradient norms are computed for a single batch only
        and resetted for each forward call to the linear module.
        However multiple backward calls are supported.
    
        Args:
            module (nn.Linear): linear layer
    """

    def __init__(self, module:nn.Linear) -> None:
        assert isinstance(module, nn.Linear)
        # register capture hooks
        module.register_forward_hook(self.forward_hook)
        module.register_full_backward_hook(self.backward_hook)
        # reset
        self.H:torch.FloatTensor = None
        self.Z_bar:torch.FloatTensor = None

    @torch.no_grad()
    def forward_hook(
        self,
        module:nn.Linear,
        x:Tuple[torch.FloatTensor], 
        y:Tuple[torch.FloatTensor]
    ) -> None:
        """ Forward hook capturing inputs to module 
        
            Args:
                x (Tuple[torch.FloatTensor]): input to linear module
                y (Tuple[torch.FloatTensor]): output of linear module
        """
        # compute squared norm of input
        self.H = (x[0] * x[0]).flatten(start_dim=1).sum(dim=1)
        self.Z_bar = []

    @torch.no_grad()
    def backward_hook(
        self, 
        module:nn.Linear,
        x_grad:Tuple[torch.FloatTensor], 
        y_grad:Tuple[torch.FloatTensor]
    ) -> None:
        """ Backward hook caturing gradients of module outputs 
        
            Args:
                x_grad (Tuple[torch.FloatTensor]): gradient of input
                y_grad (Tuple[torch.FloatTensor]): gradient of output
        """
        # sum of squared gradients
        z_bar = (y_grad[0] * y_grad[0]).flatten(start_dim=1).sum(dim=1)
        self.Z_bar.append(z_bar)

    def compute(self) -> torch.FloatTensor:
        """ Compute Squarred Gradient Norm for last processed batch 
        
            Returns:
                norm (torch.FloatTensor): 
                    squared gradient norm for weight of linear module
                    in shape (b, m) where b is the batch size and m is
                    the number of backward calls.
        """
        # stack gradients from multiple backward calls
        # +1 to account for bias
        H = self.H.unsqueeze(1) + 1.0
        Z_bar = torch.stack(self.Z_bar, dim=1)
        # check dimensions
        assert H.size(0) == Z_bar.size(0), "Mismatch between forward (%i) and backward (%i) calls" % (H.size(0), Z_bar.size(0))
        return H * Z_bar


class GoodfellowGradientNorm(object):
    """ Compute Example-wise gradient norm of all linear components in the
        given module following Goodfellow (2016).
        
        Gradient norms are computed for a single batch only
        and resetted for each forward call to the linear module.
        However multiple backward calls are supported.
    
        Args:
            module (nn.Module): module which's linear components to analyze
    """

    def __init__(self, model:nn.Module) -> None:
        # register capturing hooks
        self.grad_captures = [
            GoodfellowGradientNormForLinear(module)
            for module in model.modules()
            if isinstance(module, nn.Linear)
        ]

    def compute(self) -> torch.FloatTensor:
        """ Compute Squarred Gradient Norm for last processed batch 
        
            Returns:
                norm (torch.FloatTensor): 
                    squared gradient norm for weight of linear module
                    in shape (b, m) where b is the batch size and m is
                    the number of backward calls.
        """
        # compute all gradients and add them
        return sum(grad.compute() for grad in self.grad_captures)


class _EglBase(AbstractStrategy):
    """ Abstract Base for Maximum Expected Gradient Length Sampling 
        Strategy. Assumes cross-entropy loss. Child classes must
        overwrite the abstract method `_get_hallucinated_labels`
        which generates the labels for gradient computations.
    
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
        super(_EglBase, self).__init__()
        # create gradient norm computer
        self.grad_norm = GoodfellowGradientNorm(model)
        # move model to available device(s)
        self.model = idist.auto_model(model)
        # save k
        self.k = k

    @abstractmethod
    def _get_hallucinated_labels(
        self, 
        logits:torch.FloatTensor,
        mask:Union[torch.BoolTensor, None],
        k:int
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
                k (int): number of labels to generate per example

            Returns:
                probs (torch.FloatTensor):
                    probabilities of the specific label infered from the logits.
                    Must be of shape (n, k). For Token Classification tasks the
                    shape must be (n, s, k).
                labels (torch.LongTensor):
                    hallucinated labels selected for loss computation. Must be of the
                    shape (n, k).
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
        # move batch to device and pass through model
        batch = move_to_device(batch, device=idist.device())
        logits = self.model(**batch).logits
        # get top-k predictions
        mask = batch.get('attention_mask', None)
        probs, labels = self._get_hallucinated_labels(logits, mask, k=self.k)
        assert probs.size(0) == labels.size(0) == logits.size(0), "Mismatch between batchsizes of probs (%i), labels (%i) and logits (%i)" % (probs.size(0), labels.size(0), logits.size(0))
        assert probs.size(-1) == labels.size(-1), "Mismatch between selected labels of probs (%i) and labels (%i)" % (probs.size(-1), labels.size(-1))
        assert probs.ndim == 2, "Expected probabilities two have two dimensions but got shape $s" % str(tuple(probs.size()))
        # compute loss and backpropagate
        for i in range(labels.size(-1)):
            F.cross_entropy(
                logits.flatten(end_dim=-2), 
                labels[..., i].flatten(), 
                reduction='sum'
            ).backward(retain_graph=i < labels.size(-1)-1)
        # approximate expected gradient length
        with torch.no_grad():
            return torch.sum(probs * self.grad_norm.compute(), dim=1)

    def sample(self, output:torch.FloatTensor, query_size:int) -> Sequence[int]:
        """ Get samples with maximum expected gradient length 
        
            Args:
                output (torch.FloatTensor): expected gradient length of samples
                query_size (int): number of samples to draw

            Returns:
                indices (Sequence[int]): drawn examples given by their indices in `output`
        """
        # get the samples with top expected gradient length
        return output.topk(k=query_size).indices


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
    
    @torch.no_grad()
    def _get_hallucinated_labels(
        self, 
        logits:torch.FloatTensor, 
        mask:Union[torch.BoolTensor, None],
        k:int
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """ Get hallucinated labels for which to compute the loss and gradients.
            Selects the top-k predictions as hallucinated labels.
        
            Args:
                logits (torch.FloatTensor): 
                    logits of shape (n, m) where n is the number of examples 
                    and m is the number of labels. For Token Classification tasks
                    the shape is (n, s, m) where s is the sequence length.
                k (int): number of labels to generate per example

            Returns:
                probs (torch.FloatTensor):
                    probabilities of the specific label infered from the logits.
                    Must be of shape (n, k). For Token Classification tasks the
                    shape must be (n, s, k).
                labels (torch.LongTensor):
                    hallucinated labels selected for loss computation. Must be of the
                    shape (n, k).
        """
        # assert logits.ndim == 2, "Expected simple sequence classification, i.e. logits of two dimensions but got logits of shape %s" % str(logits.shape)
        k = min(self.k, logits.size(-1))
        probs = torch.softmax(logits, dim=-1) 
        probs, labels = torch.topk(probs, k=k, dim=-1)
        # handle token classification
        if logits.ndim == 3:
            # token classification, compute probability for label sequence
            probs = probs if mask is None else probs.masked_fill_(~mask.unsqueeze(-1), 1.0)
            probs = torch.prod(probs, dim=1)
        # return probabilities and labels
        return probs, labels


class EglBySampling(_EglBase):
    """ Implements Maximum Expected Gradient Length Sampling Strategy
        by random sampling labels from the predicted logits. Assumes 
        cross-entropy loss.
    
        Args:
            model (PreTrainedModel): 
                transformer-based sequence classification model
            k (int):
                maximum number of targets to consider for
                expectation approximation. Defaults to 5.
    """

    @torch.no_grad()
    def _get_hallucinated_labels(
        self, 
        logits:torch.FloatTensor, 
        mask:Union[torch.BoolTensor, None],
        k:int
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """ Get hallucinated labels for which to compute the loss and gradients.
            Samples labels from distribution defined by logits.
        
            Args:
                logits (torch.FloatTensor): 
                    logits of shape (n, m) where n is the number of examples 
                    and m is the number of labels. For Token Classification tasks
                    the shape is (n, s, m) where s is the sequence length.
                k (int): number of labels to generate per example

            Returns:
                probs (torch.FloatTensor):
                    probabilities of the specific label infered from the logits.
                    Must be of shape (n, k). For Token Classification tasks the
                    shape must be (n, s, k).
                labels (torch.LongTensor):
                    hallucinated labels selected for loss computation. Must be of the
                    shape (n, k).
        """
        # compute probabilities
        probs = torch.softmax(logits, dim=-1) 
        # sample labels from logits
        labels = torch.multinomial(
            probs.flatten(end_dim=-2),
            num_samples=k,
            replacement=True
        ).reshape(*logits.size()[:-1], k)
        # token classification, apply mask to disable loss for invalid tokens
        if (mask is not None) and (logits.ndim == 3):
            labels.masked_fill_(~mask.unsqueeze(-1), -100)
        # weight each label the same, weighting is done by sampling
        return torch.ones((logits.size(0), k), device=labels.device), labels


class _EglFastBase(_EglBase):
    """ Abstract Base for Fast Maximum Expected Gradient Length Sampling 
        Strategy. Assumes cross-entropy loss. Child classes must
        overwrite the abstract method `_get_hallucinated_labels`
        which generates the labels for gradient computations.
    
        Differs from `_EglBase` by computing a weighted sum over the
        losses resulting from the k hallucinated labels, thus only
        requiring a single backward pass through the model. However
        this leads to the following gradient norm approximation

        ||g||^2 = (p1*g1*g1 + ... + pk*gk*gk) ~ (p1*g1+p2*g2+...+pk*gk)^2

        where gi is the gradient induced by the i-th hallucinated label.

        Args:
            model (PreTrainedModel): 
                transformer-based sequence classification model
            k (int):
                maximum number of targets to consider for
                expectation approximation. Defaults to 5.
    """

    def process(self, batch:Any) -> torch.FloatTensor:
        """ Compute expected gradient lengths for given batch 
        
            Args:
                batch (Any): input batch for which to compute grad norms
        
            Returns:
                norm (torch.FloatTensor): (squared) gradient norm length computed following Goodfellow (2016)
        """
        self.model.eval()
        # move batch to device and pass through model
        batch = move_to_device(batch, device=idist.device())
        logits = self.model(**batch).logits
        # get top-k predictions
        probs, labels = self._get_hallucinated_labels(logits, k=self.k)
        assert probs.size(0) == labels.size(0) == logits.size(0), "Mismatch between batchsizes of probs (%i), labels (%i) and logits (%i)" % (probs.size(0), labels.size(0), logits.size(0))
        assert probs.size(-1) == labels.size(-1), "Mismatch between selected labels of probs (%i) and labels (%i)" % (probs.size(-1), labels.size(-1))
        # compute all losses and stack them
        losses = torch.stack([
            F.cross_entropy(
                logits.flatten(start_dim=1), 
                labels[..., i].flatten(), 
                reduction='none'
            )
            for i in range(labels.size(-1))
        ], dim=-1)
        # weighted sum and backpropagate
        (probs.sqrt() * losses).sum().backward()
        # approximate expected gradient length
        with torch.no_grad():
            return self.grad_norm.compute().flatten()


class EglFastByTopK(_EglFastBase, EglByTopK):
    """ Implements Maximum Expected Gradient Length Sampling Strategy
        by random sampling labels from the predicted logits. Assumes 
        cross-entropy loss.
        
        Differs from `EglByTopK` by computing a weighted sum over the
        losses resulting from the k hallucinated labels, thus only
        requiring a single backward pass through the model. However
        this leads to the following gradient norm approximation

        ||g||^2 = (p1*g1*g1 + ... + pk*gk*gk) ~ (p1*g1+p2*g2+...+pk*gk)^2

        where gi is the gradient induced by the i-th hallucinated label.
    
        Args:
            model (PreTrainedModel): 
                transformer-based sequence classification model
            k (int):
                maximum number of targets to consider for
                expectation approximation. Defaults to 5.
    """


class EglFastBySampling(_EglFastBase, EglBySampling):
    """ Implements Maximum Expected Gradient Length Sampling Strategy
        by random sampling labels from the predicted logits. Assumes 
        cross-entropy loss.
        
        Differs from `EglBySampling` by computing a weighted sum over the
        losses resulting from the k hallucinated labels, thus only
        requiring a single backward pass through the model. However
        this leads to the following gradient norm approximation

        ||g||^2 = (p1*g1*g1 + ... + pk*gk*gk) ~ (p1*g1+p2*g2+...+pk*gk)^2

        where gi is the gradient induced by the i-th hallucinated label.
    
        Args:
            model (PreTrainedModel): 
                transformer-based sequence classification model
            k (int):
                maximum number of targets to consider for
                expectation approximation. Defaults to 5.
    """
