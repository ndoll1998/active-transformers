# experiment setup for english Conll2003 Dataset

source configs/token/defaults.sh
# specify experiment name and dataset
EXPERIMENT=conll2003-german
DATASET=src/data/conll2003.py
LABEL=ner_tags
# pretrained model to use
PRETRAINED_CKPT=bert-base-german-cased
# optimization hyperparameters
LR=2e-5
LR_DECAY=1.0
WEIGHT_DECAY=0.01
BATCH_SIZE=12
MAX_EPOCHS=15
MIN_LENGTH=0
MAX_LENGTH=64
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=0.98
# active learning hyperparameters
AL_STEPS=10
QUERY_SIZE=100
