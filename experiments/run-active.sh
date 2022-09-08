""" Execute the scripts/run_active.py script for a specifies setup
    and multiple random seeds. Also sets up the wandb environment in
    which the runs are logged.

    Expects a path to a valid configuration file as argument, i.e.
    >> bash run-active.sh configs/conll2003.sh

    Configuration File:
        TASK: task to solve, i.e. token or sequence.
        EXPERIMENT: experiment name. Only used for logging.
        DATASET: dataset to load for experiment.
        LABEL: which column of the dataset stores the target label(s).
        MIN_LENGTH: minimum sequence length
        MAX_LENGTH: maximum sequence length
        PRETRAINED_CKPT: pretrained transformer checkpoint
        LR: learning rate
        WEIGHT_DECAY: weight decay
        MAX_EPOCHS: maximum number of training epochs to do in an AL step
        PATIENCE: early stopping patience
        ACCURACY_THRESHOLD: accuracy threshold used as convergence criteria

    Environment Variables:
        CUDA_VISIBLE_DEVICES: 
            Specify which cuda device to use. Defaults to 0.
        GROUP_ID: 
            Specify the group-id used to log into weights and biases.
            The group name is build from the EXPERIMENT and the GROUP_ID
        AL_STEPS: 
            Specify the number of active learning steps to do.
            Defaults to 50.
        QUERY_SIZE: 
            Specify the number of samples to query in each AL step.
            Defaults to 32.
        STRATEGIES:
            Specify the Active Learning Strategies to apply.
        OVERWRITE_MIN_LENGTH: overwrite minimum sequence length set in configuration file.
        OVERWRITE_MAX_LENGTH: overwrite maximum sequence length set in confuguration file.
        OVERWRITE_LR: overwrite learning rate set in configuration file
        OVERWRITE_WEIGHT_DECAY: overwrite weight decay set in configuration file.
        OVERWRITE_MAX_EPOCHS: overwrite maximum number of training epochs per AL step set in configuration file.
        OVERWRITE_BATCH_SIZE: overwrite batch size set in configuration file.
        OVERWRITE_PATIENCE: overwrite early stopping patience set in configuration file.
        OVERWRITE_ACCURACY_THRESHOLD: overwrite accuracy threshold set in configuration file.
""" &> /dev/null

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
AL_STEPS=${AL_STEPS:-50}
QUERY_SIZE=${QUERY_SIZE:-32}

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
            --min-length ${OVERWRITE_MIN_LENGTH:-$MIN_LENGTH} \
            --max-length ${OVERWRITE_MAX_LENGTH:-$MAX_LENGTH} \
            --pretrained-ckpt $PRETRAINED_CKPT \
            --strategy $STRATEGY \
            --query-size $QUERY_SIZE \
            --steps $AL_STEPS \
            --lr ${OVERWRITE_LR:-$LR} \
            --weight-decay ${OVERWRITE_WEIGHT_DECAY:-$WEIGHT_DECAY} \
            --epochs ${OVERWRITE_MAX_EPOCHS:-$MAX_EPOCHS} \
            --batch-size ${OVERWRITE_BATCH_SIZE:-$BATCH_SIZE} \
            --patience ${OVERWRITE_PATIENCE:-$PATIENCE} \
            --acc-threshold ${OVERWRITE_ACCURACY_THRESHOLD:-$ACCURACY_THRESHOLD} \
            --seed $SEED
    done
done
