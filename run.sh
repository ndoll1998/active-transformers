export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'0'}

# make sure config is provided
if [ -z $1 ]; then
    echo No Configuration Provided!
    exit
fi

if [ ! -f $1 ]; then
    echo Configuration file not found!
    exit
fi

# load config
source $1

# set default random group id if not set
GROUP_ID="${GROUP_ID:-$(cat /proc/sys/kernel/random/uuid | head -c 8)}"
# set up weights and biases
export WANDB_MODE="online"
export WANDB_PROJECT="active-transformers-all-data"
export WANDB_RUN_GROUP="$EXPERIMENT-$GROUP_ID"

# model cache directory
CACHE_ID=$(cat /proc/sys/kernel/random/uuid | head -c 8)
MODEL_CACHE=/tmp/model-cache-$CACHE_ID

# strategies to apply
DEFAULT_STRATEGIES="badge" #random least-confidence prediction-entropy entropy-over-max badge alps egl-sampling"
STRATEGIES=${STRATEGIES:-$DEFAULT_STRATEGIES}

# use full dataset
AL_STEPS=300
QUERY_SIZE=32
BATCH_SIZE=64

# run experiment
for SEED in 2567556381 20884829 1872321349 3003095696 72456076; do

    for STRATEGY in $STRATEGIES; do

        # set wandb run name
        export WANDB_NAME="$EXPERIMENT-$STRATEGY-$SEED"

        # run training
        python $SCRIPT \
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
            --min-length $MIN_LENGTH \
            --seed $SEED
        
    done
done
