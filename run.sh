export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES='3'

if false; then
    # parameters from https://huggingface.co/gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner
    python scripts/train_naive.py \
        --lang en \
        --pretrained-ckpt distilbert-base-uncased \
        --lr 2e-5 \
        --epochs 45 \
        --epoch-length 64 \
        --batch-size 16 \
        --max-length 128 \
        --use-cache False
fi

if true; then
    python scripts/train_active.py \
        --lang en \
        --pretrained-ckpt distilbert-base-uncased \
        --heuristic least-confidence \
        --lr 2e-5 \
        --steps 100 \
        --epochs 45 \
        --epoch-length 64 \
        --batch-size 16 \
        --query-size 8 \
        --patience 8 \
        --max-length 128 \
        --use-cache False \
        --seed 1337
fi
