#!/bin/bash
# Train Diffusion Policy, one run per array task, from a manifest of
# "<task> <filter_key> <seed>" lines (cluster/gen_manifest.py).
#
#   sbatch --array=0-<N-1>%10 --export=ALL,MANIFEST=cluster/manifest_square.txt cluster/train_dp.slurm
#
# Paper settings: 100 epochs x 500 gradient steps, batch 256, TF32 (--fast) on
# can and square; EPOCHS=300 for transport. All runs of one comparison used the
# same GPU type (NVIDIA A40). Evaluation of the final checkpoint happens later,
# in reeval.slurm.
#
# robomimic rewrites last.pth and last_bak.pth (~1 GB each) after every epoch.
# On a slow shared filesystem that dominates the wall clock, so training writes
# to node-local storage and the finished run is copied back once.
#
# Add a --partition line for your cluster if it needs one.
#SBATCH --job-name=curaeval_dp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/train_dp_%A_%a.log
set -uo pipefail
ROOT=${CURAEVAL_ROOT:-$SLURM_SUBMIT_DIR}
source "$ROOT/cluster/env.sh"
cd "$ROOT"

MANIFEST=${MANIFEST:-cluster/manifest.txt}
read -r T FK S <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
[ -z "${T:-}" ] && { echo "no manifest line for index $SLURM_ARRAY_TASK_ID"; exit 0; }
NAME="dp_${T}_${FK}_seed${S}"
case "$T" in square) HOR=500;; transport) HOR=700;; *) HOR=400;; esac
echo "[$(hostname)] idx=$SLURM_ARRAY_TASK_ID task=$T filter=$FK seed=$S name=$NAME"

LOCAL=${TMPDIR:-/dev/shm}
if [ "$(df -Pk "$LOCAL" 2>/dev/null | awk 'NR==2{print $4}')" -lt 8000000 ]; then
  echo "less than 8 GB free on $LOCAL; writing checkpoints to $CURAEVAL_WORK directly"
  LOCAL="$CURAEVAL_WORK/ckpt"
fi
WORK="$LOCAL/curaeval_$SLURM_JOB_ID/$NAME"
FINAL="$CURAEVAL_WORK/ckpt/$NAME"
mkdir -p "$WORK"; rm -rf "$FINAL"

copy_back() {
  mkdir -p "$FINAL"
  # keep the final model and its snapshots; drop the rolling 1 GB backup
  if rsync -a --exclude 'last_bak.pth' "$WORK/" "$FINAL/" || cp -r "$WORK/." "$FINAL/"; then
    rm -rf "$LOCAL/curaeval_$SLURM_JOB_ID"
  else
    echo "copy back failed; files left in $WORK"
  fi
}
trap 'echo "time limit approaching"; copy_back; exit 1' USR1

$CURAEVAL_PY scripts/run_dp.py \
  --dataset "$CURAEVAL_WORK/data_masked/${T}_mh.hdf5" --name "$NAME" --output_dir "$WORK" \
  --filter_key "$FK" --epochs "${EPOCHS:-100}" --steps_per_epoch 500 --batch_size 256 \
  --rollout_n 50 --horizon "$HOR" --seed "$S" --fast &
wait $!
STATUS=$?
copy_back
[ "$STATUS" -eq 0 ] && echo "TRAIN_DONE $NAME" || echo "TRAIN_FAILED $NAME status=$STATUS"
exit "$STATUS"
