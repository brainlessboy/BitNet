#!/bin/bash
# BLIPDistill v2: Teacher-corpus pipeline on RunPod
#
# Phase 1: Teacher generates its own training corpus (text only, small)
# Phase 2: Full-KL distillation on that corpus (proven approach)
#
# Usage:
#   bash blipdistill/runpod_pipeline.sh Qwen/Qwen2.5-0.5B cuda    # validation
#   bash blipdistill/runpod_pipeline.sh Qwen/Qwen2.5-3B cuda       # production

MODEL=${1:-"Qwen/Qwen2.5-0.5B"}
DEVICE=${2:-"cuda"}
MODEL_SHORT=$(echo $MODEL | sed 's/.*\///')
CORPUS_DIR="blipdistill/corpus_${MODEL_SHORT}"

echo "============================================"
echo "  BLIPDistill v2: $MODEL_SHORT"
echo "============================================"
echo ""
echo "  Phase 1: Generate teacher corpus"
echo "  Phase 2: Full-KL distillation"
echo ""

cd /root/BitNet

# Phase 1: Generate corpus
echo ">>> Phase 1: Generating teacher corpus..."
python -u blipdistill/generate_corpus.py \
  --teacher $MODEL \
  --device $DEVICE \
  --batch_size 32 \
  --seq_len 256 \
  --n_sequences 0 \
  --output $CORPUS_DIR

if [ $? -ne 0 ]; then
    echo ">>> ERROR: Corpus generation failed!"
    exit 1
fi

# Phase 2: Train
echo ""
echo ">>> Phase 2: Full-KL distillation..."
python -u blipdistill/train.py \
  --teacher $MODEL \
  --corpus $CORPUS_DIR \
  --device $DEVICE \
  --max_steps 10000 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --lr 5e-4 \
  --tau 2.0 \
  --save_every 1000

if [ $? -ne 0 ]; then
    echo ">>> ERROR: Training failed!"
    exit 1
fi

echo ""
echo ">>> Pipeline complete!"
echo ""
echo "  Test on pod:"
echo "    python distill/test_checkpoint.py blipdistill/checkpoints/final.pt -n 50 -t 0.7 -r 1.3"
echo ""
echo "  Download:"
echo "    runpodctl send /root/BitNet/blipdistill/checkpoints/final.pt"
