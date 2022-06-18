# experiment setup for english Conll2003 Dataset

source configs/token/defaults.sh
# specify experiment name and dataset
EXPERIMENT=conll2003
DATASET=conll2003
LABEL=ner_tags
# pretrained model to use
PRETRAINED_CKPT=distilroberta-base
# optimization hyperparameters
LR=2e-5
LR_DECAY=0.975
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
