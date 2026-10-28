#!/bin/bash
# CUPID influence scores for one square-MH scoring checkpoint (one GPU).
#
#   sbatch --export=ALL,S=1 cluster/cupid.slurm      # then S=2, S=3
#
# Uses the full-data Diffusion Policy checkpoint of training seed S
# (dp_square_none_seed$S, trained by train_dp.slurm). Stage 2 collects 100
# success-labelled rollouts of that checkpoint; stages 3.1-3.2 compute the
# per-demonstration influence scores with CUPID's defaults (resumable
# through --workdir). One seed takes about 12 hours on an NVIDIA A40.
# After all three seeds, merge and write the masks (cupid_port/README.md):
#
#   python cupid_port/rank_merge_cupid.py --scores $CURAEVAL_WORK/cupid/scores_square_seed{1,2,3}.json \
#       --out $CURAEVAL_WORK/cupid/scores_square_rankmerged.json
#   python cupid_port/make_cupid_masks.py --scores $CURAEVAL_WORK/cupid/scores_square_rankmerged.json \
#       --dataset $CURAEVAL_WORK/data_masked/square_mh.hdf5 --retentions 0.25 0.5 0.75
#
# Add a --partition line for your cluster if it needs one.
#SBATCH --job-name=curaeval_cupid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/cupid_%j.log
set -uo pipefail
ROOT=${CURAEVAL_ROOT:-$SLURM_SUBMIT_DIR}
source "$ROOT/cluster/env.sh"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S=${S:-1}
DS="$CURAEVAL_WORK/data_masked/square_mh.hdf5"
CKPT=$(find "$CURAEVAL_WORK/ckpt/dp_square_none_seed${S}" -name last.pth -print -quit 2>/dev/null)
[ -z "$CKPT" ] && { echo "no last.pth for seed ${S}"; exit 1; }
ROLL="$CURAEVAL_WORK/cupid/rollouts_square_seed${S}.hdf5"
OUT="$CURAEVAL_WORK/cupid/scores_square_seed${S}.json"
echo "[$(hostname)] seed=$S ckpt=$CKPT"

if [ ! -f "$ROLL" ]; then
  $CURAEVAL_PY cupid_port/collect_rollouts.py \
    --ckpt "$CKPT" --out "$ROLL" --n 100 --horizon 500 --seed 0 || exit 1
fi
$CURAEVAL_PY cupid_port/cupid_influence.py \
  --dataset "$DS" --ckpt "$CKPT" --rollouts "$ROLL" --out "$OUT" \
  --workdir "$CURAEVAL_WORK/cupid/scores_square_seed${S}_work" || exit 1
echo "CUPID_DONE seed=${S} -> $OUT"
