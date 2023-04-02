import torch
import transformers
from typing import Optional

class ForwardForNestedTokenClassification(transformers.PreTrainedModel):
    """ Forward container class used by transformer models for nested token classification.
        Has to be combined with a transformer model for token classification.
    """

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        """ Forward function for nested token classification.
            Based on forward for standard token classification.

            Output logits are of shape (B, S, E, 3) where B
            is the batch size, S is the sequence length and
            E is the number of classifiers/entities to predict.
            The final dimension corresponds to BIO labeling scheme.
        """
        # call forward transformer for token classification and reorganize logits
        out = super(ForwardForNestedTokenClassification, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # don't compute loss in token classification model
            labels=None,
            # expects dictionary output
            return_dict=True
        )
        out['logits'] = out.logits.reshape(out.logits.size(0), out.logits.size(1), -1, 3)

        # compute loss
        if labels is not None:
            out['loss'] = torch.nn.functional.cross_entropy(out.logits.reshape(-1, 3), labels.reshape(-1))

        # return output
        return out

class AutoModelForNestedTokenClassification(object):

    @staticmethod
    def from_pretrained(pretrained_ckpt, **kwargs):
        # re-interpret num labels as num entities
        # for each entity introduce 3 labels (i.e. B,I,O)
        kwargs['num_labels'] = 3 * kwargs.get('num_labels', 1)

        # get requested model type        
        config = transformers.AutoConfig.from_pretrained(pretrained_ckpt)
        model_type = transformers.AutoModelForTokenClassification._model_mapping.get(type(config), None)
        # create new model type overwriting the forward method
        # for nested bio tagging predictions and loss computation
        model_type_for_nested = type(
            "TransformerForNestedTokenClassification",
            (ForwardForNestedTokenClassification, model_type),
            {}
        )

        return model_type_for_nested.from_pretrained(pretrained_ckpt, **kwargs)
