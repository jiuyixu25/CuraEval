#!/usr/bin/env bash
# Download the robomimic multi-human (MH) low_dim datasets used in the paper:
# data/robomimic/{can,square,transport}/mh/low_dim_v141.hdf5 (about 0.9 GB).
# Needs robomimic installed (requirements/train.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m robomimic.scripts.download_datasets --download_dir data/robomimic \
    --tasks can square transport --dataset_types mh --hdf5_types low_dim
ls -la data/robomimic/*/mh/low_dim_v141.hdf5
