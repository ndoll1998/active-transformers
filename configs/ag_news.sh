# specify task
TASK=sequence

# specify experiment name and dataset
EXPERIMENT=ag-news
DATASET=ag_news
LABEL=label
MIN_LENGTH=0
MAX_LENGTH=64

# active learning hyperparameters
AL_STEPS=20
QUERY_SIZE=25

# pretrained model
PRETRAINED_CKPT=distilroberta-base

# optimization hyperparameters
LR=2e-5
WEIGHT_DECAY=0.01
BATCH_SIZE=12
MAX_EPOCHS=50
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=0.98
