import torch
from itertools import product
from src.active.utils.sequence import topk_sequences

class TestTopKSequences:
    """ Test cases for top-k sequences utility function """

    def test_correct_posteriors(self):

        # hardcoded example
        probs = torch.FloatTensor(
            [
                [0.1, 0.3, 0.6],
                [0.0, 0.6, 0.4],
                [0.7, 0.3, 0.0],
                [0.3, 0.3, 0.4]
            ]
        )
        S, C = probs.size()
        
        # build sorted sequences using top-k function
        p, _ = topk_sequences(probs, C**S)
        
        # check if posteriors are ordered and sum up to 1
        assert (p == p.sort(descending=True).values).all()
        assert p.sum() == 1.0

        # compute all posteriors
        posteriors = torch.FloatTensor([
            probs[range(S), idx].prod().item()
            for idx in product(range(C), repeat=S)
        ]).sort(descending=True).values
        
        # check if posteriors match
        assert (p == posteriors).all()
 
    def test_posteriors_with_random(self):        
        # create random probabilities
        probs = torch.rand(32, 8)
        probs /= probs.sum(dim=1, keepdims=True)
        # build sorted sequences using top-k function
        p, _ = topk_sequences(probs, 16)
        # test order and sum must be <= 1.0
        assert (p == p.sort(descending=True).values).all()
        assert p.sum() <= 1.0
