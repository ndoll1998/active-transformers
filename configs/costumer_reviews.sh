# specify task
TASK=sequence

# specify experiment name and dataset
EXPERIMENT=costumer-reviews
DATASET=src/data/costumer_reviews.py
LABEL=label
MIN_LENGTH=0
MAX_LENGTH=40

# pretrained model
PRETRAINED_CKPT=distilroberta-uncased

# optimization hyperparameters
LR=2e-5
WEIGHT_DECAY=0.01
BATCH_SIZE=12
MAX_EPOCHS=15
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=0.98
