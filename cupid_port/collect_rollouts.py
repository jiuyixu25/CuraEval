#!/usr/bin/env python
"""Collect success-labelled closed-loop rollouts of a robomimic DP checkpoint
into an hdf5 that cupid_influence.py can consume as the CUPID "test" set.

This is CUPID stage 2 (their eval_save_episodes.py) on our robomimic stack:
the rollouts must come from the SAME checkpoint whose influence is being
computed (CUPID's estimator attributes the policy's own visitation; do not mix
checkpoints).  The rollout loop mirrors robomimic
utils/train_utils.py::run_rollout (policy(ob=...), env.step, is_success OR-
accumulate, terminate_on_success), and the env is wrapped exactly like
training-time eval (EnvUtils.wrap_env_from_config -> FrameStackWrapper).

Output layout (robomimic-style, readable by SequenceDataset):
  data/demo_i/obs/<key>   (L, D)  raw single frames (pre-step obs)
  data/demo_i/actions     (L, Da) executed policy actions (env action space)
  data/demo_i/rewards     (L,)
  data/demo_i/dones       (L,)
  data/demo_i.attrs: num_samples, success
  data.attrs: env_args (from ckpt), plus provenance attrs

Needs sim (robosuite + EGL) -> run on a GPU node with the training venv.
CUPID paper uses m = 100 rollouts for simulated tasks (Sec. 7.2 shows
convergence by 50-100 on harder tasks).

Example:
  python cupid_port/collect_rollouts.py \
    --ckpt $CURAEVAL_WORK/ckpt/dp_square_none_seed1/.../models/last.pth \
    --out  $CURAEVAL_WORK/cupid/rollouts_square_seed1.hdf5 \
    --n 100 --horizon 500 --seed 0
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="robomimic DP last.pth")
    p.add_argument("--out", required=True, help="output rollout hdf5")
    p.add_argument("--n", type=int, default=100, help="number of rollouts")
    p.add_argument("--horizon", type=int, default=500,
                   help="max steps per rollout (square-mh eval uses 500)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_terminate_on_success", action="store_true",
                   help="keep rolling after success (default terminates, "
                        "matching our run_dp.py eval config)")
    a = p.parse_args()

    import numpy as np
    import torch

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    import h5py
    import robomimic
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils

    print(f"[collect_rollouts] robomimic {robomimic.__version__} | device {device}")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=a.ckpt, device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(
        algo_name=ckpt_dict["algo_name"], ckpt_dict=ckpt_dict, verbose=False)
    env_meta = ckpt_dict["env_metadata"]
    obs_keys = list(ckpt_dict["shape_metadata"]["all_obs_keys"])
    frame_stacked = ("frame_stack" in config.train) and (config.train.frame_stack > 1)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    env = EnvUtils.wrap_env_from_config(env, config=config)  # FrameStackWrapper
    terminate_on_success = not a.no_terminate_on_success

    def frame(ob, k):
        """Current single-frame obs (last row of the frame stack)."""
        v = np.asarray(ob[k])
        return v[-1] if frame_stacked else v

    episodes = []
    t0 = time.time()
    for ep in range(a.n):
        policy.start_episode()
        ob = env.reset()
        obs_buf = {k: [] for k in obs_keys}
        act_buf, rew_buf, done_buf = [], [], []
        success = False
        try:
            for _ in range(a.horizon):
                for k in obs_keys:
                    obs_buf[k].append(frame(ob, k))
                ac = policy(ob=ob, goal=None)
                ob, r, done, _ = env.step(ac)
                act_buf.append(np.asarray(ac))
                rew_buf.append(float(r))
                success = success or bool(env.is_success()["task"])
                done_buf.append(bool(done) or (terminate_on_success and success))
                if done or (terminate_on_success and success):
                    break
        except env.rollout_exceptions as e:
            print(f"[collect_rollouts] WARNING ep {ep}: rollout exception {e}; "
                  f"keeping {len(act_buf)} steps as a failure")
            success = False
        if len(act_buf) == 0:
            print(f"[collect_rollouts] WARNING ep {ep}: empty episode, skipped")
            continue
        episodes.append((obs_buf, act_buf, rew_buf, done_buf, success))
        print(f"[collect_rollouts] ep {ep:3d}: len {len(act_buf):4d} "
              f"success {int(success)}")

    succ = np.array([e[-1] for e in episodes], dtype=bool)
    print(f"[collect_rollouts] {len(episodes)} episodes | success rate "
          f"{succ.mean():.3f} | {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with h5py.File(a.out, "w") as f:
        data = f.create_group("data")
        total = 0
        for i, (obs_buf, act_buf, rew_buf, done_buf, success) in enumerate(episodes):
            g = data.create_group(f"demo_{i}")
            og = g.create_group("obs")
            for k in obs_keys:
                og.create_dataset(k, data=np.asarray(obs_buf[k], dtype=np.float32))
            g.create_dataset("actions", data=np.asarray(act_buf, dtype=np.float32))
            g.create_dataset("rewards", data=np.asarray(rew_buf, dtype=np.float32))
            g.create_dataset("dones", data=np.asarray(done_buf, dtype=np.uint8))
            L = len(act_buf)
            g.attrs["num_samples"] = L
            g.attrs["success"] = int(success)
            total += L
        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps(env_meta)
        data.attrs["rollout_meta"] = json.dumps(dict(
            ckpt=os.path.abspath(a.ckpt), n=a.n, horizon=a.horizon,
            seed=a.seed, terminate_on_success=terminate_on_success,
            success_rate=float(succ.mean()),
            robomimic_version=robomimic.__version__,
        ))
    print(f"[collect_rollouts] wrote {a.out}")


if __name__ == "__main__":
    main()
