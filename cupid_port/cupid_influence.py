#!/usr/bin/env python
"""CUPID influence scores for robomimic-0.5 Diffusion Policy checkpoints.

Port of CUPID (Agia et al., arXiv:2506.19121, CoRL 2025) stage 3.1 (TRAK
action-influence) + stage 3.2 (per-demo performance influence) onto the
robomimic-0.5 `DiffusionPolicyUNet` stack.  Self-contained: pure torch +
robomimic; the TRAK linear algebra is re-implemented here 1:1 against the
reference implementation (https://github.com/agiachris/trak, MIT) instead of
importing it; README.md maps each step to the reference code.

Inputs
  --dataset   robomimic low_dim hdf5 the checkpoint was trained on
  --ckpt      robomimic `last.pth` (saved by TrainUtils.save_model)
  --rollouts  hdf5 of closed-loop rollouts of THIS checkpoint with per-episode
              success labels (write it with collect_rollouts.py)

Output: json with per-demo scores for every CUPID aggregation variant
(exp_key = "<aggr_fn>-<metric>-all"); the canonical CUPID score is
"sum_of_sum-net-all" (CUPID notebooks/data_curation.ipynb cell 14,
curation_key "influence_sum_official").  Higher = keep the demo.

Device-agnostic (CPU or CUDA).  Deterministic given --seed.  Resumable: the
projected-gradient memmaps live in --workdir; re-running skips finished rows.

Example (one GPU):
  python cupid_port/cupid_influence.py \
    --dataset  $CURAEVAL_WORK/data_masked/square_mh.hdf5 \
    --ckpt     $CURAEVAL_WORK/ckpt/dp_square_none_seed1/.../models/last.pth \
    --rollouts $CURAEVAL_WORK/cupid/rollouts_square_seed1.hdf5 \
    --out      $CURAEVAL_WORK/cupid/scores_square_seed1.json
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# CUPID defaults, from the CUPID release (github.com/agiachris/cupid):
# scripts/train/train_trak.sh and
# results_util.LOAD_TRAK_KWARGS["default_diffusion"].  The model output is
# CUPID's default loss_fn "square", f = mean(eps_pred^2).
# ---------------------------------------------------------------------------
CUPID_DEFAULTS = dict(
    proj_dim=4000,        # TRAK JL projection dim
    lambda_reg=0.0,       # ridge added to (PhiT Phi)
    num_timesteps=64,     # diffusion steps averaged per sample
    seed=0,               # projector + timestep/noise seed (trak exp_seed=0)
)
CANONICAL_EXP_KEY = "sum_of_sum-net-all"   # CUPID = influence_sum_official

# Fixed projector chunking (rows of the JL matrix generated per block).  Part
# of the projector's definition: changing it changes the projection.  Recorded
# in workdir meta and asserted on resume.
PROJ_CHUNK_ROWS = 1 << 16


# ===========================================================================
# Stage 3.2: per-demo performance influence (pure numpy).
# Re-implemented 1:1 from CUPID diffusion_policy/common/error_util.py.
# ===========================================================================
def ep_lens_to_idxs(ep_lens):
    ep_ends = np.asarray(ep_lens).cumsum()
    out = []
    for i, end in enumerate(ep_ends):
        start = 0 if i == 0 else ep_ends[i - 1]
        out.append(np.arange(start, end))
    return out


def _mean_of_mean(s, ok):
    return float(s.mean(axis=1).mean())


def _mean_of_mean_success(s, ok):
    return float(s.mean(axis=1).mean()) if ok else 0.0


def _sum_of_sum(s, ok):
    return float(s.sum(axis=1).sum())


def _sum_of_sum_success(s, ok):
    return float(s.sum(axis=1).sum()) if ok else 0.0


def _min_of_max(s, ok):
    v = s.max(axis=1).min()
    return float(-v if ok else v)


def _max_of_min(s, ok):
    v = s.min(axis=1).max()
    return float(v if ok else -v)


AGGR_FNS = {
    "mean_of_mean": _mean_of_mean,
    "mean_of_mean_success": _mean_of_mean_success,
    "sum_of_sum": _sum_of_sum,
    "sum_of_sum_success": _sum_of_sum_success,
    "min_of_max": _min_of_max,
    "max_of_min": _max_of_min,
}
METRICS = ("succ", "fail", "net")


def pairwise_sample_to_trajectory_scores(pairwise, test_idxs, train_idxs,
                                         success_mask, aggr_fn):
    """(N_test, N_train) sample influence -> (E_test, D_train) block reduce."""
    traj = np.zeros((len(test_idxs), len(train_idxs)), dtype=np.float32)
    for i, ti in enumerate(test_idxs):
        ok = bool(success_mask[i])
        rows = pairwise[ti]
        for j, tj in enumerate(train_idxs):
            traj[i, j] = aggr_fn(rows[:, tj], ok)
    return traj


def compute_demo_quality_scores(traj_scores, success_mask, metric="net"):
    success_mask = np.asarray(success_mask, dtype=bool)
    has_s, has_f = bool(success_mask.any()), bool((~success_mask).any())
    if metric == "succ":
        return traj_scores[success_mask].sum(axis=0) if has_s \
            else np.zeros(traj_scores.shape[1])
    if metric == "fail":
        return -traj_scores[~success_mask].sum(axis=0) if has_f \
            else np.zeros(traj_scores.shape[1])
    if metric == "net":
        return (compute_demo_quality_scores(traj_scores, success_mask, "succ")
                + compute_demo_quality_scores(traj_scores, success_mask, "fail"))
    raise ValueError(metric)


# ===========================================================================
# Chunked Rademacher projector.
# Mirrors the ROLE of trak's CudaProjector/BasicProjector (rademacher entries
# +-1, fixed seed, same matrix for featurize and score); the bit-level random
# stream differs from fast_jl, a projection-seed-level deviation (README.md).
# Deterministic across runs/devices of the same device type.
# ===========================================================================
class ChunkedRademacherProjector:
    def __init__(self, grad_dim, proj_dim, seed, device, chunk_rows=PROJ_CHUNK_ROWS):
        self.grad_dim = int(grad_dim)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.chunk_rows = int(chunk_rows)
        # generate 8 Rademacher signs per random byte (8x fewer RNG calls;
        # sign stream is deterministic in (seed, chunk index, proj_dim)).
        self._nbytes = (self.proj_dim + 7) // 8
        self._bit_idx = torch.arange(8, device=self.device, dtype=torch.uint8)

    def _block(self, ci, rows, dtype):
        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed * 1_000_003 + ci)
        raw = torch.randint(0, 256, (rows, self._nbytes), generator=g,
                            device=self.device, dtype=torch.uint8)
        bits = (raw.unsqueeze(-1) >> self._bit_idx) & 1          # (rows, nbytes, 8)
        blk = bits.reshape(rows, self._nbytes * 8)[:, :self.proj_dim]
        return blk.to(dtype).mul_(2).sub_(1)                     # {0,1} -> {-1,+1}

    @torch.no_grad()
    def project(self, G):
        """G: (B, grad_dim) on self.device -> (B, proj_dim) float32."""
        assert G.shape[1] == self.grad_dim, (G.shape, self.grad_dim)
        out = torch.zeros(G.shape[0], self.proj_dim, device=self.device,
                          dtype=torch.float32)
        for ci, start in enumerate(range(0, self.grad_dim, self.chunk_rows)):
            rows = min(self.chunk_rows, self.grad_dim - start)
            P = self._block(ci, rows, G.dtype)
            out += (G[:, start:start + rows] @ P).float()
        return out

    @torch.no_grad()
    def project_multi(self, Gs):
        """List of (Bi, grad_dim) -> (sum Bi, proj_dim) float32. Same math as
        project(torch.cat(Gs, 0)) (blocks are deterministic in chunk index and
        rows accumulate independently) up to BLAS batch-shape reduction order
        (~1e-6 relative on CPU fp32). Never materializes the concatenated
        gradient matrix, whose 2x transient memory peak does not fit a 48 GB GPU."""
        for g in Gs:
            assert g.shape[1] == self.grad_dim, (g.shape, self.grad_dim)
        outs = [torch.zeros(g.shape[0], self.proj_dim, device=self.device,
                            dtype=torch.float32) for g in Gs]
        for ci, start in enumerate(range(0, self.grad_dim, self.chunk_rows)):
            rows = min(self.chunk_rows, self.grad_dim - start)
            P = self._block(ci, rows, Gs[0].dtype)
            for g, out in zip(Gs, outs):
                out += (g[:, start:start + rows] @ P).float()
        return torch.cat(outs, dim=0)


# ===========================================================================
# Functional forward of robomimic-0.5 DiffusionPolicyUNet.
# Mirrors DiffusionPolicyUNet.train_on_batch (robomimic/algo/diffusion_policy.py,
# lines 182-217 @ d309eaec): obs encode -> flatten cond -> add_noise -> eps_pred.
# ===========================================================================
class DPForward(nn.Module):
    """Wraps nets['policy'] (ModuleDict{obs_encoder, noise_pred_net}) with a
    single forward so torch.func.functional_call can substitute parameters for
    the whole computation."""

    def __init__(self, policy_moduledict):
        super().__init__()
        self.policy = policy_moduledict   # params registered under "policy."

    def forward(self, obs_dict, noisy_actions, timesteps):
        # obs_dict: {k: (B, To, Dk)}; noisy_actions: (B, Tp, Da); timesteps: (B,)
        some = next(iter(obs_dict.values()))
        B, To = some.shape[0], some.shape[1]
        flat = {k: v.reshape(B * To, *v.shape[2:]) for k, v in obs_dict.items()}
        feats = self.policy["obs_encoder"](obs=flat)           # (B*To, D)
        obs_cond = feats.reshape(B, -1)                        # == (B,To,D).flatten(1)
        return self.policy["noise_pred_net"](
            noisy_actions, timesteps, global_cond=obs_cond)


def make_get_output(wrapper, scheduler, obs_keys):
    """CUPID model-output f(z; theta) = mean(eps_pred^2) for one sample
    (vmapped over the batch), averaged over K sampled diffusion timesteps.

    Mirrors CUPID's .../data_attribution/modelout_functions.py
    ::DiffusionLowdimFunctionalModelOutput.get_output (loss_fn "square"), with
    the robomimic DP loss convention: no inpainting/condition mask
    (loss_mask == 1 everywhere, cf. robomimic train_on_batch which computes
    plain F.mse_loss).

    Structural difference from CUPID's gradient computer (README.md): CUPID
    averages the K per-timestep GRADIENTS outside torch.func.grad
    (_average_vectorize); we average the K per-timestep LOSSES inside f and
    take one gradient, identical by linearity of the gradient and K-fold
    cheaper in gradient materialization.
    """

    def one_step(params, buffers, obs_dict, action, t):
        noise = torch.randn(action.shape, device=action.device,
                            dtype=action.dtype)
        noisy = scheduler.add_noise(action, noise, t)
        eps = torch.func.functional_call(
            wrapper, (params, buffers), (obs_dict, noisy, t))
        return (eps ** 2).mean()

    def get_output(params, buffers, tsteps, action, *obs):
        # tsteps (K,); action (Tp, Da); obs one tensor (To, Dk) per key
        obs_dict = {k: o.unsqueeze(0) for k, o in zip(obs_keys, obs)}
        action = action.unsqueeze(0)
        K = tsteps.shape[0]
        total = action.new_zeros(())
        for k in range(K):
            total = total + one_step(params, buffers, obs_dict, action,
                                     tsteps[k:k + 1].long())
        return total / K

    return get_output


class RobomimicDPGradientComputer:
    """Per-sample grads of f wrt a parameter subset, averaged over K sampled
    diffusion timesteps.  Mirrors CUPID's .../gradient_computers.py
    ::DiffusionLowdimFunctionalGradientComputer (vmap of torch.func.grad,
    randomness='different', accumulate/K, flatten to (B, grad_dim))."""

    def __init__(self, wrapper, scheduler, obs_keys, grad_keys_prefix, device):
        self.wrapper = wrapper
        self.obs_keys = list(obs_keys)
        self.device = device
        all_params = dict(wrapper.named_parameters())
        if grad_keys_prefix:
            prefixes = tuple(p for p in grad_keys_prefix.split(",") if p)
            self.params = {k: v for k, v in all_params.items()
                           if k.startswith(prefixes)}
            if not self.params:
                raise ValueError(
                    f"--grad_keys {grad_keys_prefix!r} matched no parameters; "
                    f"examples: {list(all_params)[:5]}")
        else:
            self.params = all_params
        self.buffers = dict(wrapper.named_buffers())
        self.param_order = list(self.params.keys())
        self.grad_dim = sum(v.numel() for v in self.params.values())
        self.get_output = make_get_output(wrapper, scheduler, self.obs_keys)
        self._grad_fn = torch.func.grad(self.get_output, argnums=0)
        in_dims = (None, None, 0, 0) + (0,) * len(self.obs_keys)
        self._vmapped = torch.func.vmap(self._grad_fn, in_dims=in_dims,
                                        randomness="different")

    def _flatten(self, gdict, out):
        ptr = 0
        for name in self.param_order:
            g = gdict[name]
            n = g[0].numel()
            out[:, ptr:ptr + n] = g.reshape(g.shape[0], -1).detach()
            ptr += n

    def compute(self, obs_dict, actions, timesteps):
        """obs_dict {k:(B,To,Dk)}, actions (B,Tp,Da), timesteps (B,K) -> (B,D).

        The K-timestep average happens inside the model-output function (see
        make_get_output); one vmapped grad call per batch."""
        B = timesteps.shape[0]
        obs = [obs_dict[k] for k in self.obs_keys]
        g = self._vmapped(self.params, self.buffers, timesteps, actions, *obs)
        grads = torch.zeros(B, self.grad_dim, device=self.device,
                            dtype=torch.float32)
        self._flatten(g, grads)
        return grads


# ===========================================================================
# Dataset plumbing (mirrors robomimic train_utils.dataset_factory kwargs).
# ===========================================================================
def build_sequence_dataset(hdf5_path, config, all_obs_keys, filter_key=None,
                           normalize_obs=False):
    from robomimic.utils.dataset import SequenceDataset
    action_config = {k: dict(v) for k, v in dict(config.train.action_config).items()}
    for k in action_config:           # normalization applied manually via ckpt
        action_config[k]["normalization"] = None
    return SequenceDataset(
        hdf5_path=str(hdf5_path),
        obs_keys=list(all_obs_keys),
        action_keys=list(config.train.action_keys),
        dataset_keys=list(config.train.action_keys),  # actions only; rewards/dones not needed
        action_config=action_config,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="low_dim",
        hdf5_use_swmr=True,
        hdf5_normalize_obs=normalize_obs,
        filter_by_attribute=filter_key,
        load_next_obs=False,
    )


def dataset_ep_info(ds):
    """Demo names (hdf5 keys, numeric order) and per-demo sample counts."""
    names = list(ds.demos)
    lens = [int(ds._demo_id_to_demo_length[d]) for d in names]
    # with pad_frame_stack & pad_seq_length both True, samples per demo == length
    assert sum(lens) == len(ds), "padding disabled? samples != sum(lengths)"
    return names, lens


# ===========================================================================
# Featurization loop with resumable memmaps.
# ===========================================================================
def featurize(loader, algo, computer, projector, mmap_path, done_path, n_total,
              num_timesteps, T_train, gen, proj_batch, device, desc,
              obs_norm_stats=None):
    import robomimic.utils.obs_utils as ObsUtils
    from tqdm import tqdm

    if obs_norm_stats is not None:
        # tensorize once, on device (ckpt stats arrive as np arrays)
        obs_norm_stats = {
            m: {s: torch.as_tensor(np.asarray(v), device=device,
                                   dtype=torch.float32)
                for s, v in kv.items()}
            for m, kv in obs_norm_stats.items()}

    k = projector.proj_dim
    mode = "r+" if Path(mmap_path).exists() else "w+"
    phi = np.lib.format.open_memmap(mmap_path, mode=mode, dtype=np.float32,
                                    shape=(n_total, k))
    if Path(done_path).exists():
        done = np.load(done_path)
        assert done.shape == (n_total,)
    else:
        done = np.zeros(n_total, dtype=np.uint8)
    normalize = math.sqrt(computer.grad_dim)   # trak traker.py:167 & 426

    buf, buf_rows = [], []
    row = 0

    def flush():
        if not buf:
            return
        proj = (projector.project_multi(buf) / normalize).cpu().numpy()
        r0 = 0
        for (s, e) in buf_rows:
            phi[s:e] = proj[r0:r0 + (e - s)]
            done[s:e] = 1
            r0 += e - s
        np.save(done_path, done)
        buf.clear()
        buf_rows.clear()

    buf_dtype = torch.float16 if device.type == "cuda" else torch.float32
    for batch in tqdm(loader, desc=desc):
        B = batch["actions"].shape[0]
        s, e = row, row + B
        row = e
        # timesteps are consumed from the generator even for skipped batches so
        # resumed runs stay on the same RNG stream for later batches.
        tsteps = torch.randint(0, T_train, (B, num_timesteps),
                               generator=gen, device=device)
        if done[s:e].all():
            continue
        proc = algo.process_batch_for_training(batch)   # slices To/Tp, to_float, to device
        obs = proc["obs"]
        if obs_norm_stats is not None:
            obs = ObsUtils.normalize_dict(obs, normalization_stats=obs_norm_stats)
        G = computer.compute(obs, proc["actions"], tsteps)
        buf.append(G.to(buf_dtype))
        buf_rows.append((s, e))
        if sum(b.shape[0] for b in buf) >= proj_batch:
            flush()
    flush()
    assert row == n_total, (row, n_total)
    assert done.all(), "featurization incomplete"
    phi.flush()
    return phi


# ===========================================================================
# TRAK finalize + score (mirrors the trak fork exactly; see README.md):
#   traker.py:426,581   phi = P^T g / sqrt(grad_dim)
#   traker.py:491-494   xtx over TRAIN rows only (hessian_lim), features = g @ xtx_inv
#   score_computers.py:160-176  xtx_inv = inv(xtx + lambda I) fp32, /= abs().mean()
#   traker.py:672-688   scores = features @ phi_test^T   (Q == 1, num_models == 1)
# ===========================================================================
def trak_scores(phi_train, phi_test, lambda_reg, device, block=8192):
    k = phi_train.shape[1]
    xtx = torch.zeros(k, k, dtype=torch.float32, device=device)
    for s in range(0, phi_train.shape[0], block):
        g = torch.as_tensor(phi_train[s:s + block], device=device)
        xtx += g.T @ g
    xtx_reg = xtx + lambda_reg * torch.eye(k, device=device)
    xtx_inv = torch.linalg.inv(xtx_reg)
    xtx_inv /= xtx_inv.abs().mean()          # trak score_computers.py:166 (rank-neutral)
    scores = np.zeros((phi_train.shape[0], phi_test.shape[0]), dtype=np.float32)
    gt = torch.as_tensor(np.asarray(phi_test), device=device)
    for s in range(0, phi_train.shape[0], block):
        g = torch.as_tensor(phi_train[s:s + block], device=device)
        feats = g @ xtx_inv
        scores[s:s + block] = (feats @ gt.T).cpu().numpy()
    return scores        # (N_train, N_test), CUPID's on-disk orientation


# ===========================================================================
# Checkpoint loading (thin wrapper over robomimic FileUtils).
# ===========================================================================
def load_policy(ckpt_path, device):
    import robomimic.utils.file_utils as FileUtils
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(ckpt_path), device=device, verbose=False)
    algo = rollout_policy.policy
    config, _ = FileUtils.config_from_checkpoint(
        algo_name=ckpt_dict["algo_name"], ckpt_dict=ckpt_dict, verbose=False)
    return algo, config, ckpt_dict


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="training hdf5 (low_dim)")
    p.add_argument("--ckpt", required=True, help="robomimic DP last.pth")
    p.add_argument("--out", required=True, help="output scores json")
    p.add_argument("--rollouts", required=True,
                   help="rollout hdf5 from collect_rollouts.py")
    p.add_argument("--filter_key", default=None,
                   help="filter key the checkpoint was trained on "
                        "(default: all demonstrations)")
    p.add_argument("--proj_dim", type=int, default=CUPID_DEFAULTS["proj_dim"])
    p.add_argument("--lambda_reg", type=float, default=CUPID_DEFAULTS["lambda_reg"])
    p.add_argument("--num_timesteps", type=int,
                   default=CUPID_DEFAULTS["num_timesteps"])
    p.add_argument("--seed", type=int, default=CUPID_DEFAULTS["seed"])
    p.add_argument("--grad_keys", default="",
                   help="comma-separated parameter-name prefixes that restrict "
                        "the gradients; CUPID uses all parameters (the "
                        "default), smoke_test.py restricts them to stay fast")
    p.add_argument("--grad_batch", type=int, default=16,
                   help="micro-batch for vmapped per-sample grads")
    p.add_argument("--proj_batch", type=int, default=128,
                   help="samples buffered per projection round")
    p.add_argument("--device", default="auto")
    p.add_argument("--workdir", default=None,
                   help="dir for resumable feature memmaps "
                        "(default: <out>_work/)")
    a = p.parse_args()

    t0 = time.time()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    if device.type == "cpu":
        # OMP_NUM_THREADS=1 is exported for numpy/OpenBLAS safety; give torch
        # intra-op parallelism back explicitly for the CPU path.
        torch.set_num_threads(min(12, os.cpu_count() or 1))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    import robomimic
    print(f"[cupid_port] robomimic {robomimic.__version__} | torch "
          f"{torch.__version__} | device {device}")

    algo, config, ckpt_dict = load_policy(a.ckpt, device)
    if ckpt_dict["algo_name"] != "diffusion_policy":
        raise ValueError(f"checkpoint algo is {ckpt_dict['algo_name']!r}, "
                         "expected 'diffusion_policy'")
    To = config.algo.horizon.observation_horizon
    Tp = config.algo.horizon.prediction_horizon
    if config.train.seq_length < Tp or config.train.frame_stack < To:
        raise ValueError(
            f"ckpt config train.seq_length={config.train.seq_length} / "
            f"frame_stack={config.train.frame_stack} < horizons (Tp={Tp}, "
            f"To={To}); this checkpoint was trained on degenerate windows.")
    T_train = algo.noise_scheduler.config.num_train_timesteps

    use_ema = algo.ema is not None       # attribute the deployed (EMA) weights
    nets_src = algo.ema.averaged_model if use_ema else algo.nets
    module = nets_src["policy"]
    module.eval()
    wrapper = DPForward(module).to(device).eval()

    shape_meta = ckpt_dict["shape_metadata"]
    all_obs_keys = list(shape_meta["all_obs_keys"])

    computer = RobomimicDPGradientComputer(
        wrapper, algo.noise_scheduler, all_obs_keys, a.grad_keys, device)
    print(f"[cupid_port] weights={'ema' if use_ema else 'nets'} grad_dim="
          f"{computer.grad_dim:,} over {len(computer.param_order)} tensors | "
          f"To={To} Tp={Tp} T={T_train} K={a.num_timesteps} "
          f"proj_dim={a.proj_dim}")

    # ---- datasets ----------------------------------------------------------
    obs_norm_stats = None
    if ckpt_dict.get("obs_normalization_stats", None) is not None:
        # ckpt stats already np-ified by policy_from_checkpoint
        obs_norm_stats = ckpt_dict["obs_normalization_stats"]
    act_stats = ckpt_dict.get("action_normalization_stats", None)
    if act_stats is not None:
        for m in act_stats.values():
            sc, off = np.asarray(m["scale"]), np.asarray(m["offset"])
            if not (np.allclose(sc, 1.0) and np.allclose(off, 0.0)):
                raise NotImplementedError(
                    "checkpoint has non-identity action normalization "
                    f"(scale={sc.ravel()[:3]}..., offset={off.ravel()[:3]}...) "
                    "; this port supports delta actions without normalization "
                    "only. Extend build_sequence_dataset before using it.")

    train_ds = build_sequence_dataset(
        a.dataset, config, all_obs_keys, filter_key=a.filter_key,
        normalize_obs=bool(config.train.hdf5_normalize_obs))
    train_names, train_lens = dataset_ep_info(train_ds)

    test_ds = build_sequence_dataset(a.rollouts, config, all_obs_keys,
                                     filter_key=None, normalize_obs=False)
    test_names, test_lens = dataset_ep_info(test_ds)
    import h5py
    with h5py.File(a.rollouts, "r") as f:
        success = np.array(
            [bool(f["data"][n].attrs["success"]) for n in test_names])
    test_source = f"rollouts:{a.rollouts}"
    test_obs_norm = obs_norm_stats     # normalize with TRAIN stats (ckpt)

    n_train, n_test = len(train_ds), len(test_ds)
    print(f"[cupid_port] train: {len(train_names)} demos / {n_train} samples | "
          f"test ({test_source}): {len(test_names)} eps / {n_test} samples | "
          f"success rate {success.mean():.2f}")
    if success.all():
        print("[cupid_port] WARNING: all rollouts succeeded, so 'net' == 'succ' "
              "and there is no failure signal")
    if (~success).all():
        print("[cupid_port] WARNING: all rollouts failed, so 'net' == 'fail'")

    # ---- workdir / resume --------------------------------------------------
    out_path = Path(a.out)
    workdir = Path(a.workdir) if a.workdir else \
        out_path.parent / (out_path.stem + "_work")
    workdir.mkdir(parents=True, exist_ok=True)
    meta_fp = workdir / "featurize_meta.json"
    meta = dict(ckpt=str(a.ckpt), dataset=str(a.dataset),
                filter_key=a.filter_key, test_source=test_source,
                proj_dim=a.proj_dim, seed=a.seed,
                num_timesteps=a.num_timesteps, grad_dim=computer.grad_dim,
                grad_keys=a.grad_keys,
                grad_batch=a.grad_batch, chunk_rows=PROJ_CHUNK_ROWS,
                n_train=n_train, n_test=n_test)
    if meta_fp.exists():
        old = json.loads(meta_fp.read_text())
        if old != meta:
            diff = {k: (old.get(k), meta[k]) for k in meta if old.get(k) != meta[k]}
            raise RuntimeError(
                f"workdir {workdir} holds features from a different run: "
                f"{diff}. Use a fresh --workdir.")
    else:
        meta_fp.write_text(json.dumps(meta, indent=1))

    projector = ChunkedRademacherProjector(computer.grad_dim, a.proj_dim,
                                           a.seed, device)
    loader_kw = dict(batch_size=a.grad_batch, shuffle=False, num_workers=0,
                     drop_last=False)
    gen = torch.Generator(device=device)

    algo.action_check_done = False          # verify train actions in [-1,1]
    gen.manual_seed(a.seed)
    t_feat0 = time.time()
    phi_train = featurize(
        DataLoader(train_ds, **loader_kw), algo, computer, projector,
        workdir / "phi_train.npy", workdir / "done_train.npy", n_train,
        a.num_timesteps, T_train, gen, a.proj_batch, device,
        "featurize train", obs_norm_stats=None)

    algo.action_check_done = True           # DDPM clip keeps rollout acts in range
    gen.manual_seed(a.seed + 1)
    phi_test = featurize(
        DataLoader(test_ds, **loader_kw), algo, computer, projector,
        workdir / "phi_test.npy", workdir / "done_test.npy", n_test,
        a.num_timesteps, T_train, gen, a.proj_batch, device,
        "featurize test", obs_norm_stats=test_obs_norm)
    t_feat = time.time() - t_feat0

    # ---- TRAK scores + stage 3.2 ------------------------------------------
    t_sc0 = time.time()
    scores_tt = trak_scores(phi_train, phi_test, a.lambda_reg, device)
    influence = scores_tt.T                          # (N_test, N_train)
    del scores_tt

    test_idxs = ep_lens_to_idxs(test_lens)
    train_idxs = ep_lens_to_idxs(train_lens)
    all_scores, traj_canonical = {}, None
    for aggr_name, aggr_fn in AGGR_FNS.items():
        traj = pairwise_sample_to_trajectory_scores(
            influence, test_idxs, train_idxs, success, aggr_fn)
        if aggr_name == "sum_of_sum":
            traj_canonical = traj
        for metric in METRICS:
            key = f"{aggr_name}-{metric}-all"
            all_scores[key] = compute_demo_quality_scores(
                traj, success, metric).astype(np.float64).tolist()
    t_score = time.time() - t_sc0

    result = {
        "schema": "cupid_port.v1",
        "canonical_exp_key": CANONICAL_EXP_KEY,
        "demos": train_names,
        "demo_lengths": train_lens,
        "scores": all_scores,
        "test": {
            "source": test_source,
            "episodes": test_names,
            "ep_lens": test_lens,
            "success": success.astype(int).tolist(),
            "success_rate": float(success.mean()),
        },
        "traj_scores_sum_of_sum": np.asarray(traj_canonical).tolist(),
        "config": {
            **meta,
            "lambda_reg": a.lambda_reg,
            "canonical": CANONICAL_EXP_KEY,
            "robomimic_version": robomimic.__version__,
            "torch_version": torch.__version__,
            "device": str(device),
            "wall_featurize_s": round(t_feat, 1),
            "wall_score_s": round(t_score, 1),
            "wall_total_s": round(time.time() - t0, 1),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    top = np.argsort(-np.asarray(all_scores[CANONICAL_EXP_KEY]))[:5]
    print(f"[cupid_port] wrote {out_path} | canonical={CANONICAL_EXP_KEY} | "
          f"top-5 demos: {[train_names[i] for i in top]} | "
          f"featurize {t_feat:.0f}s score {t_score:.0f}s")


if __name__ == "__main__":
    main()
