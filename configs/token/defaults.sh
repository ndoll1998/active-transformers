# Default arguments for sequence classification tasks

# script for sequence classification
SCRIPT=scripts/al_token_cls.py
# specify experiment name and dataset
EXPERIMENT=default
DATASET=;
LABEL=label
# pretrained model to use
PRETRAINED_CKPT=bert-base-uncased
# optimization hyperparameters
LR=2e-5
LR_DECAY=1.0
WEIGHT_DECAY=0.01
BATCH_SIZE=12
MAX_EPOCHS=15
MAX_LENGTH=64
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=0.98
# active learning hyperparameters
AL_STEPS=20
QUERY_SIZE=25
