#!/usr/bin/env bash
# Copy the robomimic MH datasets into $CURAEVAL_WORK/data_masked/ and write the
# selection masks into the copies. CPU only (numpy/scipy/h5py), a few minutes
# per task; the greedy HG-WT selection on transport takes longest.
#
#   bash cluster/prepare_masks.sh
#
# CUPID masks are added later by cupid_port/make_cupid_masks.py (see cupid.slurm).
set -euo pipefail
ROOT=${CURAEVAL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
source "$ROOT/cluster/env.sh"
cd "$ROOT"
for t in can square transport; do
  dst="$CURAEVAL_WORK/data_masked/${t}_mh.hdf5"
  cp -f "$CURAEVAL_DATA/$t/mh/low_dim_v141.hdf5" "$dst"
  $CURAEVAL_PY scripts/make_masks.py --dataset "$dst" --retentions 0.25 0.5 0.75 --seeds 0 1 2
  $CURAEVAL_PY scripts/make_faktual_greedy.py --dataset "$dst" --retentions 0.25 0.5 0.75
  $CURAEVAL_PY scripts/make_geom_greedy.py --dataset "$dst" --retentions 0.25 0.5 0.75
done
echo "masks written to $CURAEVAL_WORK/data_masked/"
