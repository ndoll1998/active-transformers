# specify task
TASK=token

# specify experiment name and dataset
EXPERIMENT=ncbi
DATASET=ncbi_disease
LABEL=ner_tags
MIN_LENGTH=0
MAX_LENGTH=64


# active learning hyperparameters
AL_STEPS=50
QUERY_SIZE=25

# pretrained model
PRETRAINED_CKPT=distilbert-base-uncased

# optimization hyperparameters
LR=2e-5
WEIGHT_DECAY=0.01
BATCH_SIZE=12
MAX_EPOCHS=15
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=1.0

