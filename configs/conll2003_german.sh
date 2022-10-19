# specify task
TASK=token

# specify experiment name and dataset
EXPERIMENT=conll2003-german
DATASET=./data/conll2003.py
LABEL=ner_tags
MIN_LENGTH=16
MAX_LENGTH=64

# pretrained model
PRETRAINED_CKPT=distilbert-base-german-cased

# optimization hyperparameters
LR=2e-5
WEIGHT_DECAY=0.0
BATCH_SIZE=64
MAX_EPOCHS=100
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=1.0

