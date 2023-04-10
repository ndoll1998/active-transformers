import torch
from .common import UncertaintyStrategy

class EntropyOverMax(UncertaintyStrategy):
    """ Uncertainty Sampling Strategy for token classification tasks. Computes the entropy over the maximum
        predicted probability of each token. See Equation (4.1) in Joey Oehman (2021).

        Args:
            model (PreTrainedModel): transformer-based model used for prediction
            ignore_labels (Optional[Sequence[int]]): list of labels to ignore for computation of uncertainty scores

        References:
            - Joey Oehman, Active Learning for Named Entity Recognition with Swedish Language Models, 2021
    """

    def uncertainty_measure(self, probs:torch.FloatTensor) -> torch.FloatTensor:
        # only makes sense for (nested) token-classification tasks
        assert probs.ndim >= 3, "Entropy over Maxima Sampling only makes sense for token classification tasks!"
        # only take the maximum over label space here to match expected shapes
        # in process and reduce score functions
        return probs.max(dim=-1).values

    def reduce_scores(self, scores:torch.FloatTensor, mask:torch.BoolTensor) -> torch.FloatTensor:

        # compute valid sequence lengths and avoid division by zero
        lengths = mask.sum(dim=1, keepdims=True)
        lengths = torch.maximum(lengths, torch.ones_like(lengths))
        # note that scores at ~mask are already set to zero
        # and entropy is defined to be zero at x=0 
        # (see https://pytorch.org/docs/stable/special.html)
        return (torch.special.entr(scores) / lengths).flatten(start_dim=1).sum(dim=1)


class BinaryEntropyOverMax(EntropyOverMax):

    def reduce_scores(self, scores:torch.FloatTensor, mask:torch.BoolTensor) -> torch.FloatTensor:
        # compute the number of valid scores from mask
        # also avoid division by zero errors
        n_valids = mask.sum(dim=-1)
        n_valids = torch.maximum(n_valids, torch.ones_like(n_valids))
        # flatten out dimension of classifier/entities in case of nested
        # token classification tasks, leading to summation over entities
        scores = scores.flatten(start_dim=1)
        # note that scores at ~mask are already set to zero
        # and entropy is defined to be zero at x=0 
        # (see https://pytorch.org/docs/stable/special.html)
        return (
            torch.special.entr(scores) + \
            torch.special.entr(1.0 - scores)
        ).sum(dim=-1) / n_valids
