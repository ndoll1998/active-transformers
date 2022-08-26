# specify task
TASK=token

# specify experiment name and dataset
EXPERIMENT=conll2003-german
DATASET=src/data/conll2003.py
LABEL=ner_tags
MIN_LENGTH=0
MAX_LENGTH=64

# active learning hyperparameters
AL_STEPS=10
QUERY_SIZE=100

# pretrained model
PRETRAINED_CKPT=bert-base-german-cased

# optimization hyperparameters
LR=2e-5
WEIGHT_DECAY=0.01
BATCH_SIZE=16
MAX_EPOCHS=40
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=1.0

