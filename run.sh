export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES='0'

# load config
source $1

# create random group id
GROUP_ID=$(cat /proc/sys/kernel/random/uuid | head -c 8)
# set up weights and biases
export WANDB_MODE='online'
export WANDB_PROJECT="active-transformers"
export WANDB_NAME="al-seq-cls"
export WANDB_RUN_GROUP="$EXPERIMENT-$GROUP_ID"
export WANDB_TAGS="AL,bert,seq-cls"

# model cache directory
MODEL_CACHE=/tmp/model-cache

# strategies to apply
STRATEGIES="random least-confidence prediction-entropy badge alps egl"

# run experiment
for SEED in 2567556381 20884829 1872321349 3003095696 72456076; do

    for STRATEGY in $STRATEGIES; do

        # run training
        python scripts/al_seq_cls_bert.py \
            --dataset $DATASET \
            --label-column $LABEL \
            --pretrained-ckpt $PRETRAINED_CKPT \
            --strategy $STRATEGY \
            --lr $LR \
            --lr-decay $LR_DECAY \
            --weight-decay $WEIGHT_DECAY \
            --steps $AL_STEPS \
            --epochs $MAX_EPOCHS \
            --batch-size $BATCH_SIZE \
            --query-size $QUERY_SIZE \
            --patience $PATIENCE \
            --acc-threshold $ACCURACY_THRESHOLD \
            --max-length $MAX_LENGTH \
            --model-cache $MODEL_CACHE \
            --seed $SEED
        
        # clear model cache
        rm -r $MODEL_CACHE

    done
done
