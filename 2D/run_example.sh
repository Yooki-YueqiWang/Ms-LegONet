#!/usr/bin/env bash
# Runs the whole 2D pipeline at a small size: data generation -> training -> solve on a mosaic.
# 264 training / 66 validation / 33 test patches, 300 training updates, one 4 x 4 mosaic.
# The block is trained far too briefly to reach the accuracy of the paper; this only checks
# that every stage runs. See README.md for the full-size commands.
#
#   bash run_example.sh                 (OUT=runs/example, WORKERS=4 by default)
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-runs/example}
WORKERS=${WORKERS:-4}
PY=${PYTHON:-python}

# 1. data: coefficient patches and labels (CPU; one BLAS thread per worker)
OMP_NUM_THREADS=1 $PY generate_data.py --split train --per-group 8 --seed 1 --out $OUT/data/train.npz --workers $WORKERS
OMP_NUM_THREADS=1 $PY generate_data.py --split val --per-group 2 --seed 2 --out $OUT/data/val.npz \
    --exclude $OUT/data/train.npz --workers $WORKERS
OMP_NUM_THREADS=1 $PY generate_data.py --split test --per-group 1 --seed 3 --out $OUT/data/test.npz \
    --exclude $OUT/data/train.npz $OUT/data/val.npz --no-labels --workers $WORKERS

# 2. training: the paper's schedule shrunk to 300 updates (GPU if available)
$PY train.py --train $OUT/data/train.npz --val $OUT/data/val.npz --out $OUT/model \
    --batch 8 --schedule-scale 0.0005 --log-every 50 --eval-every 100

# 3. online solve on a 4 x 4 mosaic of test patches with a random source, compared with the fine-grid Q1 solution
$PY solve.py --checkpoint $OUT/model/best.pt --data $OUT/data/test.npz --side 4 --problem grf --exact \
    --save $OUT/solution_4x4.npz
