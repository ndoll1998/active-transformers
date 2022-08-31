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

# get active learning hyperparameters or
# use defaults if not given
AL_STEPS=${AL_STEPS:-30}
QUERY_SIZE=${QUERY_SIZE:-64}

# set default random group id if not set
GROUP_ID=${GROUP_ID:-$(cat /proc/sys/kernel/random/uuid | head -c 8)}
# set up weights and biases
export WANDB_MODE="online"
export WANDB_PROJECT="active-transformers-all-data"
export WANDB_RUN_GROUP="$EXPERIMENT-$GROUP_ID"

# strategies to apply
DEFAULT_STRATEGIES="prediction-entropy" #random least-confidence prediction-entropy entropy-over-max badge alps egl-sampling"
STRATEGIES=${STRATEGIES:-$DEFAULT_STRATEGIES}

# run experiment
for SEED in 2567556381 20884829 1872321349 3003095696 72456076; do

    for STRATEGY in $STRATEGIES; do

        # set wandb run name
        export WANDB_NAME="$EXPERIMENT-$STRATEGY-$SEED"

        # run training
        python scripts/run_active.py \
            --task $TASK \
            --dataset $DATASET \
            --label-column $LABEL \
            --min-length $MIN_LENGTH \
            --max-length $MAX_LENGTH \
            --pretrained-ckpt $PRETRAINED_CKPT \
            --strategy $STRATEGY \
            --query-size $QUERY_SIZE \
            --steps $AL_STEPS \
            --lr $LR \
            --weight-decay $WEIGHT_DECAY \
            --epochs $MAX_EPOCHS \
            --batch-size $BATCH_SIZE \
            --patience $PATIENCE \
            --acc-threshold $ACCURACY_THRESHOLD \
            --seed $SEED
        
    done
done
