# specify task
TASK=sequence

# specify experiment name and dataset
EXPERIMENT=trec-6
DATASET=trec
LABEL=label-coarse
MIN_LENGTH=0
MAX_LENGTH=64

# pretrained model
PRETRAINED_CKPT=bert-large-uncased

# optimization hyperparameters
LR=2e-5
WEIGHT_DECAY=0.01
BATCH_SIZE=12
MAX_EPOCHS=15
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=0.98

