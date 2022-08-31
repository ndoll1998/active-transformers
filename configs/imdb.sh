# specify task
TASK=sequence

# specify experiment name and dataset
EXPERIMENT=imdb
DATASET=imdb
LABEL=label
MIN_LENGTH=0
MAX_LENGTH=128

# pretrained model
PRETRAINED_CKPT=distilbert-base-uncased

# optimization hyperparameters
LR=2e-5
WEIGHT_DECAY=0.01
BATCH_SIZE=64
MAX_EPOCHS=15
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=0.98

