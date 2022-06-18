# Match exact setup from the paper `Revisiting Uncertainty-based Query 
# Strategies for Active Learning with Transformers` by Christoph Schroeder 
# et al. (https://arxiv.org/pdf/2107.05687.pdf).
  
# specify experiment name and dataset
DATASET=./src/data/costumer_reviews.py
EXPERIMENT=costumer-reviews
# pretrained model to use
PRETRAINED_CKPT=bert-base-uncased
# optimization hyperparameters
LR=2e-5
LR_DECAY=1.0 #0.975
WEIGHT_DECAY=0.01
BATCH_SIZE=12
MAX_EPOCHS=15
MAX_LENGTH=50
# stopping criteria
PATIENCE=5
ACCURACY_THRESHOLD=0.98
# active learning hyperparameters
AL_STEPS=20
QUERY_SIZE=25
