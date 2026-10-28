#!/bin/bash
# Train BC-RNN(-GMM), one run per array task, from a manifest of
# "<task> <filter_key> <seed>" lines (cluster/gen_manifest.py).
#
#   sbatch --array=0-<N-1>%10 --export=ALL,MANIFEST=cluster/manifest_square.txt cluster/train_bc.slurm
#
# Paper settings: robomimic's BC-RNN low_dim recipe, 2000 epochs x 100 steps.
# The final checkpoint is evaluated later, in reeval.slurm.
#
# Add a --partition line for your cluster if it needs one.
#SBATCH --job-name=curaeval_bc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_bc_%A_%a.log
set -uo pipefail
ROOT=${CURAEVAL_ROOT:-$SLURM_SUBMIT_DIR}
source "$ROOT/cluster/env.sh"
cd "$ROOT"

MANIFEST=${MANIFEST:-cluster/manifest_bc.txt}
read -r T FK S <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
[ -z "${T:-}" ] && { echo "no manifest line for index $SLURM_ARRAY_TASK_ID"; exit 0; }
NAME="bc_${T}_${FK}_seed${S}"
case "$T" in square) HOR=500;; transport) HOR=700;; *) HOR=400;; esac
echo "[$(hostname)] idx=$SLURM_ARRAY_TASK_ID task=$T filter=$FK seed=$S name=$NAME"

rm -rf "$CURAEVAL_WORK/ckpt/$NAME"
$CURAEVAL_PY scripts/run_bc.py \
  --dataset "$CURAEVAL_WORK/data_masked/${T}_mh.hdf5" --name "$NAME" \
  --output_dir "$CURAEVAL_WORK/ckpt/$NAME" \
  --filter_key "$FK" --epochs 2000 --steps_per_epoch 100 \
  --rollout_n 50 --horizon "$HOR" --seed "$S" --fast
echo "TRAIN_DONE $NAME"
