# specify task
TASK=token

# specify experiment name and dataset
EXPERIMENT=swedish-ner
DATASET=swedish_ner_corpus
LABEL=ner_tags
MIN_LENGTH=0
MAX_LENGTH=128


# active learning hyperparameters
AL_STEPS=20
QUERY_SIZE=25

# pretrained model
PRETRAINED_CKPT=af-ai-center/bert-base-swedish-uncased

# optimization hyperparameters
LR=2e-5
WEIGHT_DECAY=0.01
BATCH_SIZE=32
MAX_EPOCHS=50
# stopping criteria
PATIENCE=15
ACCURACY_THRESHOLD=1.0

