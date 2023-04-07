import os
import tempfile
import transformers

from torch.utils.data import Dataset

def accuracy(eval_pred):
    mask = (eval_pred.label_ids >= 0)
    acc = (eval_pred.predictions[mask] == eval_pred.label_ids[mask]).sum() / mask.sum()
    return {'A': acc}

class SimpleTrainer(transformers.Trainer):

    def __init__(
        self,
        model:transformers.PreTrainedModel,
        batch_size:int =4,
        lr:float       =0.01,
        num_epochs:int =1
    ) -> None:

        # create temporary directory for output
        self.tmp_dir = tempfile.TemporaryDirectory()
        # initialize trainer
        super(SimpleTrainer, self).__init__(
            model=model,
            args=transformers.TrainingArguments(
                output_dir=self.tmp_dir.name,
                num_train_epochs=num_epochs,
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                evaluation_strategy="no",
                save_strategy="no",
                no_cuda=True,
                report_to='none',
                # avoid deprecaton warning
                optim='adamw_torch'
            ),
            compute_metrics=accuracy,
            preprocess_logits_for_metrics=lambda logits, _: logits.argmax(dim=-1),
        )

