# experiment setup for english Conll2003 Dataset

source configs/token/defaults.sh
# specify experiment name and dataset
EXPERIMENT=ncbi
DATASET=ncbi_disease
LABEL=ner_tags
# pretrained model to use
PRETRAINED_CKPT=dmis-lab/biobert-base-cased-v1.2
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
AL_STEPS=50
QUERY_SIZE=25
