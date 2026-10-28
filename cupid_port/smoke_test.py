#!/usr/bin/env python
"""End-to-end CPU smoke test for the CUPID port; no trained checkpoint needed.

Builds a tiny synthetic robomimic-style world:
  * a 5-demo low_dim hdf5 with square-MH obs keys/dims (+ train/valid masks),
  * a randomly initialized robomimic-0.5 DiffusionPolicyUNet (small UNet),
    saved in the exact TrainUtils.save_model checkpoint format,
  * a 4-episode synthetic "rollout" hdf5 with success labels [1,0,1,0],
then exercises the real pipeline:
  1. functional-forward parity: DPForward under torch.func.functional_call
     reproduces robomimic's own obs_encode->UNet noise prediction (<=1e-5);
  2. cupid_influence.py CLI on rollouts: valid json, 5 finite scores per
     exp_key, deterministic across re-runs;
  3. resume: re-run with the same workdir reuses featurized memmaps;
  4. make_cupid_masks.py: mask/cupid_{25,50,75} match top-k of the canonical
     scores with the scripts/make_masks.py k-rule.

Run inside the port venv (CPU is fine):
  cd cupid_port
  venv/bin/python smoke_test.py [--workdir DIR] [--keep]
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OBS_SPEC = {"robot0_eef_pos": 3, "robot0_eef_quat": 4,
            "robot0_gripper_qpos": 2, "object": 14}
AC_DIM = 7
DEMO_LENS = [9, 12, 16, 21, 13]
ROLLOUT_LENS = [11, 8, 14, 9]
ROLLOUT_SUCCESS = [1, 0, 1, 0]


def make_hdf5(path, lens, success=None, seed=0):
    import h5py
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, L in enumerate(lens):
            g = data.create_group(f"demo_{i}")
            og = g.create_group("obs")
            for k, d in OBS_SPEC.items():
                og.create_dataset(k, data=rng.normal(size=(L, d)).astype(np.float32))
            g.create_dataset("actions",
                             data=rng.uniform(-0.9, 0.9, size=(L, AC_DIM)).astype(np.float32))
            g.create_dataset("rewards", data=np.zeros(L, dtype=np.float32))
            g.create_dataset("dones", data=np.zeros(L, dtype=np.uint8))
            g.attrs["num_samples"] = L
            if success is not None:
                g.attrs["success"] = int(success[i])
        data.attrs["total"] = int(sum(lens))
        data.attrs["env_args"] = json.dumps(
            {"env_name": "Smoke", "type": 1, "env_kwargs": {}})
        if success is None:
            m = f.create_group("mask")
            m.create_dataset("valid", data=np.array([b"demo_3", b"demo_4"], dtype="S"))
            m.create_dataset("train",
                             data=np.array([b"demo_0", b"demo_1", b"demo_2"], dtype="S"))
    return path


def make_ckpt(dataset_path, ckpt_path):
    """Random tiny DP in the exact robomimic save_model format."""
    from robomimic.config import config_factory
    from robomimic.algo import algo_factory
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.file_utils as FileUtils

    config = config_factory("diffusion_policy")
    with config.values_unlocked():
        # robomimic-0.5 _create_networks ignores config.algo.unet.*: the
        # ConditionalUnet1D is always built with its defaults (~65.9M params),
        # so the smoke checkpoint carries the real architecture.
        config.train.data = [{"path": str(dataset_path)}]
        # scripts/train.py injects these before algo_factory (train.py:212-217);
        # the saved ckpt config therefore always carries them. Mirror that.
        config.algo.optim_params["policy"]["num_train_batches"] = 500
        config.algo.optim_params["policy"]["num_epochs"] = 100
    config.lock()
    assert config.train.seq_length == 16 and config.train.frame_stack == 2, \
        "robomimic DP config must default seq_length=16/frame_stack=2 " \
        "(needs the post-v0.5.0 fix, see README.md)"

    ObsUtils.initialize_obs_utils_with_config(config)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config={"path": str(dataset_path)},
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        verbose=False,
    )
    torch.manual_seed(7)
    algo = algo_factory("diffusion_policy", config,
                        obs_key_shapes=shape_meta["all_shapes"],
                        ac_dim=shape_meta["ac_dim"],
                        device=torch.device("cpu"))
    params = dict(
        model=algo.serialize(),
        config=config.dump(),
        algo_name="diffusion_policy",
        env_metadata={"env_name": "Smoke", "type": 1, "env_kwargs": {}},
        shape_metadata=shape_meta,
        variable_state=None,
    )
    torch.save(params, ckpt_path)
    n_unet = sum(p.numel() for p in algo.nets["policy"]["noise_pred_net"].parameters())
    n_enc = sum(p.numel() for p in algo.nets["policy"]["obs_encoder"].parameters())
    print(f"[smoke] ckpt written: UNet params {n_unet:,}, obs_encoder params {n_enc:,}")
    return ckpt_path


def check_forward_parity(ckpt_path, dataset_path):
    """DPForward(functional_call) == robomimic's own forward path."""
    import cupid_influence as ci
    import robomimic.utils.tensor_utils as TensorUtils
    from torch.utils.data import DataLoader

    torch.set_num_threads(min(12, os.cpu_count() or 1))
    algo, config, ckpt_dict = ci.load_policy(str(ckpt_path), torch.device("cpu"))
    module = (algo.ema.averaged_model if algo.ema is not None else algo.nets)["policy"]
    module.eval()
    wrapper = ci.DPForward(module).eval()
    all_obs_keys = list(ckpt_dict["shape_metadata"]["all_obs_keys"])

    ds = ci.build_sequence_dataset(dataset_path, config, all_obs_keys)
    batch = next(iter(DataLoader(ds, batch_size=6, shuffle=False)))
    proc = algo.process_batch_for_training(batch)
    obs, actions = proc["obs"], proc["actions"]
    To = config.algo.horizon.observation_horizon
    assert next(iter(obs.values())).shape[1] == To
    assert actions.shape[1] == config.algo.horizon.prediction_horizon

    torch.manual_seed(0)
    t = torch.randint(0, algo.noise_scheduler.config.num_train_timesteps, (6,))
    noise = torch.randn_like(actions)
    noisy = algo.noise_scheduler.add_noise(actions, noise, t)

    with torch.no_grad():
        # robomimic's own path (train_on_batch lines 193-214)
        feats = TensorUtils.time_distributed(
            {"obs": obs, "goal": None}, module["obs_encoder"], inputs_as_kwargs=True)
        eps_direct = module["noise_pred_net"](noisy, t, global_cond=feats.flatten(start_dim=1))
        # our functional path
        params = dict(wrapper.named_parameters())
        buffers = dict(wrapper.named_buffers())
        eps_func = torch.func.functional_call(wrapper, (params, buffers), (obs, noisy, t))
    err = (eps_direct - eps_func).abs().max().item()
    assert err <= 1e-5, f"functional forward mismatch: max abs err {err}"
    print(f"[smoke] PASS forward parity (max abs err {err:.2e})")

    # full-width per-sample grads through the REAL 65.9M-param UNet (small batch)
    computer = ci.RobomimicDPGradientComputer(
        wrapper, algo.noise_scheduler, all_obs_keys, "", torch.device("cpu"))
    assert computer.grad_dim == sum(p.numel() for p in module.parameters())
    obs4 = {k: v[:4] for k, v in obs.items()}
    g = computer.compute(obs4, actions[:4], torch.randint(0, 100, (4, 1)))
    assert g.shape == (4, computer.grad_dim) and torch.isfinite(g).all()
    assert g.abs().sum() > 0, "gradients are identically zero"
    print(f"[smoke] PASS per-sample grads: shape {tuple(g.shape)}, finite, nonzero "
          f"(grad_dim={computer.grad_dim:,})")

    # projector determinism + shape (on the full-width grads)
    proj = ci.ChunkedRademacherProjector(computer.grad_dim, 32, 0, torch.device("cpu"))
    p1, p2 = proj.project(g), proj.project(g)
    assert torch.equal(p1, p2) and p1.shape == (4, 32)
    print("[smoke] PASS projector determinism")


def run_cli(script, args):
    cmd = [sys.executable, str(HERE / script)] + [str(x) for x in args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"{script} failed (exit {r.returncode})")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--proj_dim", type=int, default=32)
    ap.add_argument("--num_timesteps", type=int, default=2)
    a = ap.parse_args()

    wd = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="cupid_smoke_"))
    wd.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] workdir: {wd}")

    ds = make_hdf5(wd / "train.hdf5", DEMO_LENS, seed=0)
    ro = make_hdf5(wd / "rollouts.hdf5", ROLLOUT_LENS, success=ROLLOUT_SUCCESS, seed=1)
    ck = make_ckpt(ds, wd / "last.pth")

    check_forward_parity(ck, ds)

    # CLI runs restrict grads to the UNet's final conv (--grad_keys) so the
    # end-to-end pipeline stays fast on CPU; full-width gradients were already
    # checked in-process above.
    common = ["--dataset", ds, "--ckpt", ck, "--proj_dim", a.proj_dim,
              "--num_timesteps", a.num_timesteps, "--grad_batch", 8,
              "--proj_batch", 16, "--device", "cpu",
              "--grad_keys", "policy.noise_pred_net.final_conv."]

    # -- rollout-mode run, twice (determinism), then resume ------------------
    out1 = wd / "scores_a.json"
    run_cli("cupid_influence.py", common + ["--rollouts", ro, "--out", out1,
                                            "--workdir", wd / "work_a"])
    blob1 = json.loads(out1.read_text())
    assert blob1["schema"] == "cupid_port.v1"
    assert blob1["demos"] == [f"demo_{i}" for i in range(5)]
    assert blob1["canonical_exp_key"] == "sum_of_sum-net-all"
    n_keys = 0
    for key, vals in blob1["scores"].items():
        assert len(vals) == 5, key
        assert np.isfinite(vals).all(), key
        n_keys += 1
    assert n_keys == 18, f"expected 6 aggr x 3 metrics = 18 exp_keys, got {n_keys}"
    canon = np.asarray(blob1["scores"]["sum_of_sum-net-all"])
    assert np.ptp(canon) > 0, "canonical scores are constant (degenerate)"
    assert blob1["test"]["success"] == ROLLOUT_SUCCESS
    # net == succ + (-fail contribution): identity check of the aggregation
    s = np.asarray(blob1["scores"]["sum_of_sum-succ-all"])
    fl = np.asarray(blob1["scores"]["sum_of_sum-fail-all"])
    assert np.allclose(canon, s + fl, atol=1e-5), "net != succ + fail"
    print("[smoke] PASS rollout-mode scores json (18 exp_keys, finite, net=succ+fail)")

    out2 = wd / "scores_b.json"
    run_cli("cupid_influence.py", common + ["--rollouts", ro, "--out", out2,
                                            "--workdir", wd / "work_b"])
    blob2 = json.loads(out2.read_text())
    assert blob1["scores"] == blob2["scores"], "scores not deterministic across runs"
    print("[smoke] PASS determinism (independent re-run, identical scores)")

    out3 = wd / "scores_c.json"
    log = run_cli("cupid_influence.py", common + ["--rollouts", ro, "--out", out3,
                                                  "--workdir", wd / "work_a"])
    blob3 = json.loads(out3.read_text())
    assert blob1["scores"] == blob3["scores"], "resume changed scores"
    print("[smoke] PASS resume from finished workdir (identical scores)")
    del log

    # -- masks ---------------------------------------------------------------
    ds_masked = wd / "train_masked.hdf5"
    shutil.copyfile(ds, ds_masked)
    run_cli("make_cupid_masks.py", ["--scores", out1, "--dataset", ds_masked,
                                    "--retentions", 0.25, 0.5, 0.75])
    import h5py
    with h5py.File(ds_masked, "r") as f:
        for ret, expect_k in ((0.25, 1), (0.5, 2), (0.75, 4)):
            tag = int(round(ret * 100))
            key = f"cupid_{tag}"
            assert key in f["mask"], key
            got = [x.decode() for x in f["mask"][key][:]]
            assert len(got) == expect_k, (key, got)
            top = set(np.argsort(-canon, kind="stable")[:expect_k].tolist())
            assert set(got) == {f"demo_{i}" for i in top}, (key, got, top)
        assert set(f["mask"]["valid"][:]) == {b"demo_3", b"demo_4"}, \
            "pre-existing masks must be untouched"
    print("[smoke] PASS make_cupid_masks (cupid_25/50/75 = top-k of canonical)")

    # -- collector import surface (no sim locally) ---------------------------
    run_cli("collect_rollouts.py", ["--help"])
    print("[smoke] PASS collect_rollouts.py --help (import surface)")

    print(f"[smoke] ALL PASS  (artifacts in {wd})")
    if not a.keep and a.workdir is None:
        shutil.rmtree(wd)
        print("[smoke] cleaned up")


if __name__ == "__main__":
    main()
