#!/usr/bin/env bash
#SBATCH --job-name=coat-dior-layer
#SBATCH --partition=partition1
#SBATCH --nodelist=hpc-stat1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH --chdir=/mnt/data1/dsl01/eval/outerwear_dior_20260925
#SBATCH --output=/mnt/data1/dsl01/eval/outerwear_dior_20260925/job_%j.log
set -euo pipefail

BASE=/mnt/data1/dsl01
TASK=$BASE/eval/outerwear_dior_20260925
DATA=$BASE/eval/deepfashion_mm
PYTHON=$BASE/opt/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
PERSON=${DIOR_PERSON:?Set DIOR_PERSON to an image basename without extension}
TOP=${DIOR_TOP:?Set DIOR_TOP to an image basename without extension}
PERSON_SEG=${DIOR_PERSON_SEGMENTATION:-$DATA/segm/${PERSON}_segm.png}
PERSON_TAXONOMY=${DIOR_PERSON_TAXONOMY:-official}
RUNNER=$TASK/eval_dior_layer_removal.py

export PYTHONPATH=$TASK/deps:$BASE/web-site-packages:$BASE/site-packages:$TASK/repo
export TORCH_HOME=$TASK/torch_cache
export PYTHONUNBUFFERED=1 MPLCONFIGDIR=$TASK/matplotlib
mkdir -p "$MPLCONFIGDIR"

"$PYTHON" "$RUNNER" \
  --dior-repo "$TASK/repo" \
  --checkpoint-dir "$TASK/checkpoints" \
  --person "$DATA/images/$PERSON.jpg" \
  --person-segmentation "$PERSON_SEG" \
  --person-parsing-taxonomy "$PERSON_TAXONOMY" \
  --top-source "$DATA/images/$TOP.jpg" \
  --top-segmentation "$DATA/segm/${TOP}_segm.png" \
  --out "$TASK/output/$PERSON"
