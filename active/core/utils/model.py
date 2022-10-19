from transformers import PreTrainedModel, AutoModel

def get_encoder_from_model(model:PreTrainedModel):
    assert isinstance(model, PreTrainedModel), "Expected a PreTrainedModel but got %s" % type(model)
    # get encoder model class   
    model_class = AutoModel._model_mapping.get(type(model.config), None)
    assert model_class is not None, "Model type not registered!"
    # find member of encoder class in model
    for module in model.children():
        if isinstance(module, model_class):
            return module
    # attribute error
    raise AttributeError("Encoder member of class %s not found" % model_class.__name__)
