#!/bin/bash
# Evaluate the final checkpoint (last.pth) of each trained run, one run per
# array task, from a manifest of run names (e.g. dp_square_hlen_50_seed1).
#
#   awk '{print "dp_"$1"_"$2"_seed"$3}' cluster/manifest_square.txt > cluster/reeval_square_dp.txt
#   sbatch --array=0-<N-1>%10 --export=ALL,MANIFEST=cluster/reeval_square_dp.txt cluster/reeval.slurm
#   python scripts/parse_reeval.py 'logs/reeval_*.log' --tsv runs_square_dp.tsv
#
# Protocol of the paper: the final checkpoint, not the best of the training-time
# evaluations; 200 rollouts on square and transport and 100 on can (NROLL=auto);
# one evaluator seed (424242) for every condition, so all conditions start from
# the same initial states.
#
# Add a --partition line for your cluster if it needs one.
#SBATCH --job-name=curaeval_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/reeval_%A_%a.log
set -uo pipefail
ROOT=${CURAEVAL_ROOT:-$SLURM_SUBMIT_DIR}
source "$ROOT/cluster/env.sh"
cd "$ROOT"

MANIFEST=${MANIFEST:-cluster/reeval_manifest.txt}
NROLL=${NROLL:-auto}
EVALSEED=${EVALSEED:-424242}
NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
[ -z "$NAME" ] && { echo "no manifest line for index $SLURM_ARRAY_TASK_ID"; exit 0; }
CKPT=$(find "$CURAEVAL_WORK/ckpt/$NAME" -name last.pth -print -quit)
[ -z "$CKPT" ] && { echo "REEVAL_MISSING $NAME"; exit 0; }
case "$NAME" in
  *_square_*)    HOR=500; AUTO=200;;
  *_transport_*) HOR=700; AUTO=200;;
  *)             HOR=400; AUTO=100;;
esac
[ "$NROLL" = "auto" ] && NROLL=$AUTO
echo "[$(hostname)] idx=$SLURM_ARRAY_TASK_ID name=$NAME nroll=$NROLL horizon=$HOR evalseed=$EVALSEED"
echo "ckpt=$CKPT"
$CURAEVAL_PY -m robomimic.scripts.run_trained_agent \
    --agent "$CKPT" --n_rollouts "$NROLL" --horizon "$HOR" --seed "$EVALSEED"
echo "REEVAL_DONE $NAME"
