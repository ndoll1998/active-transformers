import torch
import heapq

from dataclasses import dataclass, field

__all__ = ["topk_sequences"]

@dataclass
class Node:
    """ Helper class """
    probs:torch.FloatTensor
    seq:tuple
    _swapped_idx:set =field(default_factory=set)

    def __post_init__(self):
        self.seq = tuple(self.seq)
        assert len(self.seq) == self.probs.size(0)

    @property
    def class_probs(self) -> torch.FloatTensor:
        return self.probs[range(len(self.seq)), self.seq]

    @property
    def posterior(self) -> float:
       return self.class_probs.prod() 

    def __lt__(self, other):
        "Inverted to turn min-heap into max-heap"
        return self.posterior > other.posterior

    def __ge__(self, other):
        return self.posterior >= other.posterior
    def __le__(self, other):
        return self.posterior <= other.posterior

    """ needed for set operations """
    def __eq__(self, other):
        return self.seq == other.seq
    def __ne__(self, other):
        return not (self == other)
    def __hash__(self):
        return hash(self.seq)

    @property
    def children(self):
        
        idx = set(tuple(range(len(self.seq))))
        for i in (idx - self._swapped_idx):
            for j in range(self.probs.size(1)):
                if j != self.seq[i]:
                    yield Node(
                        probs=self.probs,
                        seq=self.seq[:i] + (j,) + self.seq[i+1:],
                        _swapped_idx=self._swapped_idx.union({i})
                    )

    def __str__(self) -> str:
        return "(%s)" % ', '.join([str(i) for i in self.seq])

    def __repr__(self) -> str:
        return "%s with P=%.02f" % (str(self), self.posterior)


def topk_sequences(probs, k):
    """ Find the top-k sequences with highest a-posteriori probabilities.

        Args:
            probs (torch.FloatTensor):
                input tensor of sequence with specific item probabilities.
                Shape must be (S, C) where S is the sequence length and C is
                the number of classes.
            k (int): number of sequences to generate. k is upper bounded by C^S.
        
        Returns:
            probs (torch.FloatTensor): posterior probabilities of the generated sequences.
            args (torch.LongTensor):
                 argument tensor holding the top-k sequences. Shape matches (k, S).
    """
    # check input
    k = min(k, probs.size(1) ** probs.size(0))

    # build root node
    root = Node(
        probs=probs,
        seq=torch.argmax(probs, dim=1).tolist()
    )

    # list of all leave nodes
    leaves = [root]
    nodes = []
    
    # only do k-1 to avoid building children in last iteration
    for _ in range(k - 1):
        # get leave with maximum posterior
        heapq.heapify(leaves)
        node = heapq.heappop(leaves)
       
        # 
        nodes.append(node)
        # add new leaves and uniquify
        leaves = list(set(leaves + list(node.children)) - set(nodes))

    # add last node
    heapq.heapify(leaves)
    nodes.append(heapq.heappop(leaves))

    # build output
    seq = torch.LongTensor([node.seq for node in nodes])
    p = torch.FloatTensor([node.posterior for node in nodes])

    return p, seq
