# CUPID for robomimic Diffusion Policy

This directory computes CUPID's per-demonstration performance influence
(Agia et al., *CUPID: Curating Data your Robot Loves with Influence Functions*,
CoRL 2025, arXiv:2506.19121) for Diffusion Policy checkpoints trained with
robomimic 0.5. The reference implementation is the CUPID release
(https://github.com/agiachris/cupid, commit `2941eba9`) with its TRAK fork
(https://github.com/agiachris/trak, commit `ff2fdc84`). Neither is imported;
the parts that are needed are re-implemented here.

| file | role |
|---|---|
| `collect_rollouts.py` | stage 2: success-labelled rollouts of one checkpoint (needs the simulator) |
| `cupid_influence.py` | stages 3.1 and 3.2: checkpoint, demonstrations and rollouts to per-demonstration scores (json) |
| `rank_merge_cupid.py` | averages per-demonstration ranks over several scoring checkpoints |
| `make_cupid_masks.py` | scores to `mask/cupid_<retention>` filter keys |
| `smoke_test.py` | end-to-end test on CPU with a synthetic dataset and an untrained checkpoint |

## Usage

For each scoring checkpoint (the paper used the all-data square checkpoints of
training seeds 1, 2 and 3; `cluster/cupid.slurm` runs both steps):

```bash
CKPT=$(find $CURAEVAL_WORK/ckpt/dp_square_none_seed$S -name last.pth | head -1)
# 1. 100 success-labelled rollouts of this checkpoint
python cupid_port/collect_rollouts.py --ckpt "$CKPT" \
    --out $CURAEVAL_WORK/cupid/rollouts_square_seed$S.hdf5 --n 100 --horizon 500 --seed 0
# 2. influence scores, resumable through --workdir
python cupid_port/cupid_influence.py \
    --dataset $CURAEVAL_WORK/data_masked/square_mh.hdf5 --ckpt "$CKPT" \
    --rollouts $CURAEVAL_WORK/cupid/rollouts_square_seed$S.hdf5 \
    --out $CURAEVAL_WORK/cupid/scores_square_seed$S.json \
    --workdir $CURAEVAL_WORK/cupid/scores_square_seed${S}_work
```

Then merge the checkpoints and write the masks:

```bash
python cupid_port/rank_merge_cupid.py \
    --scores $CURAEVAL_WORK/cupid/scores_square_seed{1,2,3}.json \
    --out $CURAEVAL_WORK/cupid/scores_square_rankmerged.json
python cupid_port/make_cupid_masks.py \
    --scores $CURAEVAL_WORK/cupid/scores_square_rankmerged.json \
    --dataset $CURAEVAL_WORK/data_masked/square_mh.hdf5 --retentions 0.25 0.5 0.75
```

The rollouts must come from the checkpoint that is being scored. On square-MH
(300 demonstrations, 80,731 training windows) one checkpoint takes about 12
hours on an NVIDIA A40.

`python cupid_port/smoke_test.py` checks the whole port on CPU in a few
minutes.

## What is computed

CUPID scores a training demonstration $\xi$ by the effect of upweighting it on
the expected return of the policy (paper Def. 1, Prop. 1 and Alg. 1). With $m$
rollouts $\tau_i$ of horizon $H$ and binary returns $R(\tau_i) \in \{-1, +1\}$,

```math
\hat\Psi(\xi) = \frac{1}{m} \sum_i \frac{R(\tau_i)}{H} \sum_{(s',a') \in \tau_i} \sum_{(s,a) \in \xi} \Psi_{a\text{-inf}}\big((s',a'), (s,a)\big),
```

the action influences summed over rollout steps and demonstration steps, added
for successful rollouts and subtracted for failed ones. The constants $1/m$
and $1/H$ do not change the ranking and are dropped, as in the reference code.

Action influence (stage 3.1, TRAK):

* model output $f(z;\theta) = \mathrm{mean}(\epsilon_\text{pred}^2)$ for a
  state-action window $z$, where $\epsilon_\text{pred}$ is the noise prediction
  at a uniformly sampled diffusion step with fresh Gaussian noise (CUPID's
  surrogate for $-\log\pi(a|s)$, `loss_fn="square"`), averaged over $K = 64$
  diffusion steps;
* gradients with respect to all parameters of the deployed (EMA) policy
  network. With low_dim observations the observation encoder has no
  parameters, so these are the 65,878,279 parameters of the U-Net, CUPID's
  `model.` choice;
* TRAK: $\phi(z) = P^\top \nabla_\theta f(z) / \sqrt{d}$ with a seeded
  Rademacher projection $P$ ($d$ the number of parameters, $k = 4000$
  columns), then
  $I(z_i, z_j) = [\Phi(\Phi^\top\Phi + \lambda I)^{-1}]_i\, \phi(z_j)$ with
  $\Phi$ the training features and $\lambda = 0$. As in the TRAK fork, the
  inverse is divided by its mean absolute value, which does not change the
  ranking. Features and scores are fp32; one checkpoint.

Per-demonstration score (stage 3.2): the influence matrix (rollout samples x
training samples) is reduced to rollouts x demonstrations by summing over the
samples of each block (`sum_of_sum`), and `net` adds the successful rollouts
and subtracts the failed ones, using all rollouts. The resulting key,
`sum_of_sum-net-all`, is the score CUPID's notebooks label "CUPID"
(`influence_sum_official`); the json also holds the other aggregations of the
reference code.

## Port to robomimic 0.5

The model output follows `DiffusionPolicyUNet.train_on_batch`
(`robomimic/algo/diffusion_policy.py`):

```
obs_feats = obs_encoder(obs[:, :To])                         # time-distributed
obs_cond  = obs_feats.flatten(1)                             # (B, To*D)
noisy     = DDPMScheduler.add_noise(actions[:, :Tp], eps, t)  # T = 100
eps_pred  = noise_pred_net(noisy, t, global_cond=obs_cond)
f         = mean(eps_pred ** 2)
```

`smoke_test.py` checks that this functional forward pass reproduces
robomimic's own.

* No condition mask: robomimic's Diffusion Policy is trained with a plain MSE
  loss, so CUPID's `loss_mask` is all ones.
* Windows: robomimic's `SequenceDataset` with `frame_stack=2`,
  `seq_length=16` and both paddings, one sample per time step (80,731 samples
  for square-MH). Rollout windows are built the same way.
* Weights: the EMA weights, which robomimic uses for rollouts.
* Normalization: the training runs use delta actions without action or
  observation normalization. Observation statistics stored in a checkpoint are
  applied if present; a checkpoint with action normalization is rejected.

## Differences from the reference implementation

1. Policy: robomimic 0.5 Diffusion Policy (To=2, Ta=8, Tp=16, DDPM with 100
   steps, no condition mask, EMA power 0.75) instead of the Diffusion Policy
   of the CUPID release, so windowing and the loss mask follow robomimic.
2. The K diffusion steps are averaged inside f, before the gradient; CUPID
   averages the K gradients. The two are equal by linearity.
3. Projector: a seeded, chunked Rademacher matrix generated on the fly (eight
   signs per random byte) instead of the `fast_jl` CUDA kernel. The entries
   have the same distribution and the same matrix projects training and test
   gradients; the random stream differs, which amounts to a different
   projection seed. The chunk size (2^16 rows) is part of the projection and
   is checked when a run resumes.
4. On CUDA, gradients are kept in fp16 between the gradient computation and
   the projection (relative error about 1e-3, well below the projection
   noise at k = 4000); on CPU everything is fp32.
5. Success labels: R = +1 if robosuite's `is_success()["task"]` held at any
   step of the rollout, otherwise -1.

Scores are bitwise reproducible for identical arguments. `--grad_batch`
changes the layout of the random draws (noise and diffusion steps), which is
statistically equivalent to another seed; `--proj_batch` does not change the
scores. `--grad_keys` restricts the gradients to some parameters; CUPID uses
all of them, and only `smoke_test.py` sets it.

## Defaults

| setting | value | in the reference code |
|---|---|---|
| projection dimension | 4000 | `train_trak.sh`, `LOAD_TRAK_KWARGS["default_diffusion"]` |
| ridge | 0 | same |
| model output | `square`, mean of the squared noise prediction | same |
| diffusion steps per sample | 64, uniform over the 100 training steps | `train_trak.sh` |
| gradients | all parameters of the deployed policy network | `MODEL_KEYS["low_dim"] = "model."` |
| checkpoints | 1 | `use_num_ckpts=1` |
| precision | fp32 features and scores | `use_half_precision=0` |
| seed | 0 (projection, diffusion steps, noise) | `exp_seed=0` |
| rollouts | 100 | paper, Sec. 7.2 |

## Environment

Python 3.10, PyTorch 2.x (a CPU build is enough for the smoke test) and
robomimic at commit `d309eaec` (`requirements/train.txt`), which pins
diffusers 0.11.1. The paper's training runs used robomimic installed from its
main branch on 2026-07-21 (version 0.5.0 with the PR #251 fix of the
Diffusion Policy train config, `seq_length=16` and `frame_stack=2`); its
`diffusion_policy.py` matches the pinned commit in every part used here.
`cupid_influence.py` stops with an error on a checkpoint trained without that
fix.

## Licenses

TRAK and CUPID are MIT-licensed. `cupid_influence.py` re-implements the TRAK
linear algebra and the stage-3.2 aggregation of CUPID; both copyright notices
are in `THIRD_PARTY_LICENSES.md`. robomimic (MIT) is used as a library.
