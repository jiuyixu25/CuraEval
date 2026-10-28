# Copy to cluster/env.sh and edit; every job script sources it.
#
#   cp cluster/env.example.sh cluster/env.sh
#
# Submit jobs from the repository root (the scripts fall back to
# $SLURM_SUBMIT_DIR), or export CURAEVAL_ROOT.

# Scratch space for dataset copies, checkpoints, CUPID artefacts and logs.
# One Diffusion Policy run keeps ~2 GB of checkpoints; the paper's matrix is
# about 270 runs.
export CURAEVAL_WORK=/path/to/scratch/curaeval

# The original robomimic datasets (<root>/<task>/mh/low_dim_v141.hdf5).
export CURAEVAL_DATA=${CURAEVAL_ROOT:-$PWD}/data/robomimic

# Python of the training environment (robomimic 0.5, see requirements/train.txt).
export CURAEVAL_PY=python

# Headless rendering for the evaluation rollouts (use osmesa without a GPU).
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# Single-threaded BLAS: the data are tiny, training is GPU-bound, and many
# threads per job can exhaust process limits on shared nodes.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$CURAEVAL_WORK"/ckpt "$CURAEVAL_WORK"/data_masked "$CURAEVAL_WORK"/cupid logs
