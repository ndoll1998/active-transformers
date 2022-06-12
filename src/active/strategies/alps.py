import numpy as np
# import torch and ignite
import torch
import torch.nn.functional as F
import ignite.distributed as idist
# import transformers
from transformers import (
    PreTrainedModel, 
    AutoTokenizer, 
    AutoModelForMaskedLM
)
# import sklearn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
# import base class and more
from .strategy import AbstractStrategy
from .utils import move_to_device
from typing import Sequence, Any

class Alps(AbstractStrategy):
    """ Implementation of ALPS presented in `Cold-Start Active Learning through 
        Self-supervised Language Modeling` (Yuan et al., 2020).
    
        Args:
            model (PreTrainedModel): pretrained transformer model
            mlm_prob (float): 
                masked-lanugage modeling probability used to generate 
                targets for loss that builds the surprisal embeddings
                Defaults to 15%.
    """

    def __init__(
        self,
        model:PreTrainedModel,
        mlm_prob:float =0.15
    ) -> None:
        # initialize strategy
        super(Alps, self).__init__()
        # save values
        self.mlm_prob = mlm_prob
        # get pretrained name
        pretrained_name_or_path = model.config.name_or_path
        assert pretrained_name_or_path is not None, "Expected pretrained model"
        # get special token ids
        tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)
        self.special_token_ids = torch.LongTensor(tokenizer.all_special_ids)
        # load the corresponding model with language model head and move to device(s)
        model = AutoModelForMaskedLM.from_pretrained(pretrained_name_or_path)
        self.model = idist.auto_model(model)

    @torch.no_grad()
    def process(self, batch:Any) -> torch.FloatTensor:
        """ Compute surpisal embeddings of given batch
            Args:
                batch (Any): input batch
            Returns:
                embed (torch.Tensor): surprisal embedding of samples in batch
        """
        # move batch to device and get input ids
        batch = move_to_device(batch, device=idist.device())
        input_ids = batch['input_ids']

        # build label mask, i.e. find all candidate tokens for masking
        mask = input_ids.unsqueeze(-1) == self.special_token_ids.to(idist.device())
        mask = torch.any(mask, dim=-1)
        # compute probability of each token being masked
        mask_probs = (1.0 - mask.float()) * self.mlm_prob
        label_mask = torch.bernoulli(mask_probs).bool()
        # build labels from mask
        labels = input_ids.masked_fill(~label_mask, -100)

        # predict and compute mlm-loss
        # as described in the paper the input to the model is not masked
        # but the loss is only computed for a percentage of the tokens
        out = self.model(**batch)
        loss = F.cross_entropy(
            out.logits.flatten(end_dim=1), 
            labels.flatten(),
            reduction='none'
        ).reshape(labels.size())
        # l2-normalization
        return F.normalize(loss, dim=-1)

    def sample(self, output:torch.FloatTensor, query_size:int) -> Sequence[int]:
        """ Select samples using kmeans clustering on surprisal embeddings 
        
            Args:
                output (torch.FloatTensor): surprisal embeddings
                query_size (int): number of samples to select

            Returns:
                indices (Sequence[int]): selected samples given by indices
        """
        assert output.ndim == 2
        # find cluster centers using kmeans
        kmeans = KMeans(n_clusters=query_size)
        kmeans = kmeans.fit(output.numpy())
        centers = kmeans.cluster_centers_
        # find unique nearst neighbors of cluster centers
        nearest = NearestNeighbors(n_neighbors=1)
        nearest = nearest.fit(output.numpy())
        centroids = nearest.kneighbors(centers, return_distance=False)
        centroids = set(centroids[:, 0])
        # handle doubles by filling up with random samples
        while len(centroids) < query_size:
            n = query_size - len(centroids)
            idx = np.random.randint(0, output.size(0), size=n)
            centroids |= set(idx)
        # return indices
        return centroids
