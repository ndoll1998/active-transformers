# experiment setup for swedish ner corpus
# following the hyperparameters in Joey Oehman (2021)
# (see chapters 4.1.2-4.1.8)

source configs/token/defaults.sh
# specify experiment name and dataset
EXPERIMENT=swedish-ner
DATASET=swedish_ner_corpus
LABEL=ner_tags
# pretrained model to use
PRETRAINED_CKPT=af-ai-center/bert-base-swedish-uncased
# optimization hyperparameters
LR=2e-5
LR_DECAY=1.0
WEIGHT_DECAY=0.01
BATCH_SIZE=32
MAX_EPOCHS=50
MIN_LENGTH=0
MAX_LENGTH=128
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=0.98
# active learning hyperparameters
AL_STEPS=20
QUERY_SIZE=25
