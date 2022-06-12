export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES='0'
export WANDB_MODE='online'

MAX_EPOCHS=15
MAX_LENGTH=50

STRATEGIES="random least-confidence prediction-entropy badge alps egl"

for SEED in 2567556381 20884829 1872321349 3003095696 72456076; do

    for STRATEGY in $STRATEGIES; do

        python scripts/al_seq_cls_bert.py \
            --dataset rotten_tomatoes \
            --pretrained-ckpt bert-base-uncased \
            --strategy $STRATEGY \
            --epochs $MAX_EPOCHS \
            --max-length $MAX_LENGTH \
            --seed $SEED
    
    done
done
