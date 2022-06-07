export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES='0'
export WANDB_MODE='online'

DATASET=src/data/costumer_reviews.py
MAX_EPOCHS=15
MAX_LENGTH=50

# for SEED in 2567556381 20884829 1872321349 3003095696 72456076
for SEED in 3003095696 72456076
do

    python scripts/al_seq_cls_bert_v2.py \
        --dataset $DATASET \
        --pretrained-ckpt bert-base-uncased \
        --heuristic least-confidence \
        --epochs $MAX_EPOCHS \
        --max-length $MAX_LENGTH \
        --seed $SEED
done

exit

python scripts/al_seq_cls_bert_v2.py \
    --dataset $DATASET \
    --pretrained-ckpt bert-base-uncased \
    --heuristic random \
    --epochs $MAX_EPOCHS \
    --max-length $MAX_LENGTH \
    --seed 1337

python scripts/al_seq_cls_bert_v2.py \
    --dataset $DATASET \
    --pretrained-ckpt bert-base-uncased \
    --heuristic least-confidence \
    --epochs $MAX_EPOCHS \
    --max-length $MAX_LENGTH \
    --seed 42

python scripts/al_seq_cls_bert_v2.py \
    --dataset $DATASET \
    --pretrained-ckpt bert-base-uncased \
    --heuristic least-confidence \
    --epochs $MAX_EPOCHS \
    --max-length $MAX_LENGTH \
    --seed 1337

python scripts/al_seq_cls_bert_v2.py \
    --dataset $DATASET \
    --pretrained-ckpt bert-base-uncased \
    --heuristic prediction-entropy \
    --epochs $MAX_EPOCHS \
    --max-length $MAX_LENGTH \
    --seed 42

python scripts/al_seq_cls_bert_v2.py \
    --dataset $DATASET \
    --pretrained-ckpt bert-base-uncased \
    --heuristic prediction-entropy \
    --epochs $MAX_EPOCHS \
    --max-length $MAX_LENGTH \
    --seed 1337
