# import torch and ignite
import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.distributed as idist
# import transformers and kmeans++
from transformers import PreTrainedModel
from sklearn.cluster._kmeans import kmeans_plusplus
# import base and utils
from .strategy import AbstractStrategy
from ..utils.data import move_to_device
# import others
import warnings
from typing import Sequence, Tuple, Any

class Badge(AbstractStrategy):
    """ Implementation of `Deep Batch Active Learning By Diverse, 
        Uncertain Gradient Lower Bounds` (Ash et al., 2019) for 
        transformer models.

        Uses the final hidden state computed by the transformer model as 
        embedding, i.e. assumes a linear classifier on top of the hidden
        states. For Sequence Classification tasks the hidden state of the
        first token (usually [CLS]) is used. Note that this architecture is
        not always provided. As an example `BertForSequenceClassification`
        uses pooled outputs instead of the hidden states. To specify
        specific encoder see `BadgeForSequenceClassification` and 
        `BadgeForTokenClassification`.
        
        Pipeline:
        [input] -> encoder -> [out] -> [out.hidden_states[-1]], [out.logits]

        Args:
            model (PreTrainedModel): 
                pre-trained transformer-based classification model
            is_token_classification (bool):
                specifies the classification task, i.e. assumes
                token classification is set to `True` and sequence
                classification if set to `False`. By default tries
                to infer classification type by the model name.
    """

    def __init__(
        self,
        model:PreTrainedModel,
        is_token_classification:bool =None
    ) -> None:
        # initialize strategy
        super(Badge, self).__init__()
        # move model to available device(s)
        self.model = idist.auto_model(model)

        # save classification type
        self.is_tok_cls = is_token_classification
        # only if classification type is not given
        if self.is_tok_cls is None:
            # get the model task type
            name = type(model).__name__
            if name.endswith("ForSequenceClassification"):
                self.is_tok_cls = False
            elif name.endswith("ForTokenClassification"):
                self.is_tok_cls = True
            else:
                warnings.warn("Could not infer task type from model name. Assuming Sequence Classification Task!", UserWarning)
                self.is_tok_cls = False
   
        # test whether applying the classifier on the extracted embeddings
        # matches the logits predicted by the model
        if hasattr(model, 'classifier'):
            self._test_pipeline(model.classifier)

    def _test_pipeline(self, classifier:nn.Linear) -> None:
        # get model logits and embeddings
        target_logits, embeds = self._get_logits_and_embeds({
            'input_ids': torch.zeros((1, 1), dtype=int, device=idist.device())
        })
        # use embeddings to compute classifier logits
        classifier.eval()
        classifier = classifier.to(idist.device())
        classifier_logits = classifier(embeds)
        # compare
        if not torch.allclose(target_logits, classifier_logits):
            warnings.warn("Pipeline output doesn't match model output.", UserWarning)
        if not isinstance(classifier, nn.Linear):
            warnings.warn("Classifier must be linear.", UserWarning)

    def _get_logits_and_embeds(self, batch) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Get the logits predicted by the model and the embeddings which 
            are used for classification.
            
            Args:
                batch (Any): input batch to apply the model to
                
            Returns:
                logits (torch.Tensor): predited logits
                emebds (torch.Tensor): embeddings used for classification
        """
        # apply model
        self.model.eval()
        out = self.model(**batch, output_hidden_states=True)
        # get embeds, use encoding of [CLS] token for sequence classification
        hidden_states = out.hidden_states[-1]
        embeds = hidden_states if self.is_tok_cls else hidden_states[:, 0, :]
        # return logits and embeddings
        return out.logits, embeds

    @torch.no_grad()
    def process(self, batch:Any) -> torch.FloatTensor:
        """ Compute the hallucinated gradient embeddings of a given batch
        
            Args:
                batch (Any): input batch
            Returns:
                g (torch.Tensor): hallucinated gradient embedding of samples in batch
        """
        # move batch to device and apply model
        batch = move_to_device(batch, device=idist.device())
        logits, embeds = self._get_logits_and_embeds(batch)
        # compute predictions
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        # compute gradient embedding
        w = -(probs - F.one_hot(preds, num_classes=probs.size(-1)))
        if self.is_tok_cls and ('attention_mask' in batch):
            # apply attention mask in case of token classification
            mask = batch['attention_mask'].bool()
            w[~mask] = 0.0
        g = w.unsqueeze(-1) * embeds.unsqueeze(-2)
        # return gradient embedding
        return g.flatten(start_dim=1)
    
    def sample(self, output:torch.FloatTensor, query_size:int) -> Sequence[int]:
        """ Select samples using kmeans++ on the hallucinated gradient embeddings.

            Args:
                output (torch.Tensor): hallucinated gradient emebddings
                query_size (int): number of samples to select

            Returns:
                indices (Sequence[int]): selected samples given by their indices
        """
        # use kmeans++ algorithm to find spread samples
        # w.r.t. the computed gradient embedding
        _, indices = kmeans_plusplus(
            X=output.numpy(),
            n_clusters=query_size
        )
        # return the indices of the selected samples
        return indices

class BadgeForSequenceClassification(Badge):
    """ Implementation of `Deep Batch Active Learning By Diverse, 
        Uncertain Gradient Lower Bounds` (Ash et al., 2019) for 
        transformer-based sequence classification models.

        Uses the `pooler_output` field of the encoder output as
        embeddings which are passed to the linear classifier. If
        the field is not specified the hidden state of the first
        token (usually [CLS]) is used as embedding.

        Pipeline:
        [input] -> encoder -> [out] -> [out.pooler_output] -> classifier -> [logits]

        Args:
            encoder (PreTrainedModel): transformer-based encoder
            classifier (nn.Linear): linear classifier
    """

    def __init__(
        self,
        encoder:PreTrainedModel,
        classifier:nn.Linear
    ) -> None:
        # initialize strategy
        AbstractStrategy.__init__(self)
        # move encoder to device(s)
        self.encoder = idist.auto_model(encoder)
        # make sure the classifier is linear and move
        # it to available device(s)
        assert isinstance(classifier, nn.Linear)
        self.classifier = idist.auto_model(classifier)
        # save classification type
        self.is_tok_cls = False

    def _get_logits_and_embeds(self, batch:Any) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # set model to eval mode
        self.encoder.eval()
        self.classifier.eval()
        # apply encoder and extract embeddings from output
        # fallback to encoding of [CLS] token is pooler output is not specified
        out = self.encoder(**batch, output_hidden_states=True)
        embeds = out.hidden_states[-1][:, 0, :] if not hasattr(out, 'pooler_output') else \
            out.hidden_states[-1][:, 0, :] if out.pooler_output is None else \
            out.pooler_output
        # warn about fallback
        if not hasattr(out, 'pooler_output'):
            warnings.warn("Model output does not specify pooler output. Using final hidden state of [CLS] token instead.", UserWarning)
        # apply classifier and return
        return self.classifier(embeds), embeds

class BadgeForTokenClassification(Badge):
    """ Implementation of `Deep Batch Active Learning By Diverse, 
        Uncertain Gradient Lower Bounds` (Ash et al., 2019) for 
        transformer-based token classification models.

        Uses the final hidden states produced by the encoder as
        embeddings which are passed to the linear classifier. 

        Pipeline:
        [input] -> encoder -> [out] -> [out.pooler_output] -> classifier -> [logits]

        Args:
            encoder (PreTrainedModel): transformer-based encoder
            classifier (nn.Linear): linear classifier
    """
    
    def __init__(
        self,
        encoder:PreTrainedModel,
        classifier:nn.Linear
    ) -> None:
        # initialize strategy
        AbstractStrategy.__init__(self)
        # move encoder to device(s)
        self.encoder = idist.auto_model(encoder)
        # make sure the classifier is linear and move
        # it to available device(s)
        assert isinstance(classifier, nn.Linear)
        self.classifier = idist.auto_model(classifier)
        # save classification type
        self.is_tok_cls = True
    
    def _get_logits_and_embeds(self, batch:Any) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # set model to eval mode
        self.encoder.eval()
        self.classifier.eval()
        # apply encoder and extract embeddings from output
        out = self.encoder(**batch, output_hidden_states=True)
        embeds = out.hidden_states[-1]
        # apply classifier and return
        return self.classifier(embeds), embeds

