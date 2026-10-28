"""Train robomimic's Diffusion Policy on a low_dim dataset, optionally
restricted to one filter_key (a demonstration subset).

The number of gradient steps per epoch is fixed (epoch_every_n_steps), so the
total number of steps, epochs * steps_per_epoch, is the same for subsets of any
size. The defaults are the paper's settings on can and square; transport used
--epochs 300. Every run in the paper used --fast.

Training-time rollouts run only after the last epoch (--rollout_rate 0); the
final checkpoint is evaluated separately (cluster/reeval.slurm).
"""
import argparse

import torch
from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--filter_key", default="none", help="mask/<key> to train on; none = all demonstrations")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps_per_epoch", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--rollout_n", type=int, default=50, help="training-time rollouts; 0 disables them")
    p.add_argument("--rollout_rate", type=int, default=0, help="epochs between rollouts; 0 = after the last epoch only")
    p.add_argument("--horizon", type=int, default=400, help="rollout horizon: can 400, square 500, transport 700")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fast", action="store_true",
                   help="TF32 and cudnn.benchmark with non-deterministic kernels (about 2x faster)")
    a = p.parse_args()

    c = config_factory("diffusion_policy")
    with c.values_unlocked():
        c.experiment.name = a.name
        c.train.data = a.dataset
        c.train.output_dir = a.output_dir
        c.train.seed = a.seed
        c.train.num_epochs = a.epochs
        c.train.batch_size = a.batch_size
        c.experiment.epoch_every_n_steps = a.steps_per_epoch
        c.train.hdf5_cache_mode = "all"
        if a.filter_key != "none":
            c.train.hdf5_filter_key = a.filter_key
        # no validation split; success is measured by rollouts
        c.experiment.validate = False
        c.experiment.rollout.enabled = a.rollout_n > 0
        c.experiment.rollout.n = max(1, a.rollout_n)
        c.experiment.rollout.horizon = a.horizon
        c.experiment.rollout.rate = a.rollout_rate if a.rollout_rate > 0 else a.epochs
        c.experiment.rollout.terminate_on_success = True
        c.experiment.save.enabled = True
        c.experiment.save.on_best_rollout_success_rate = True
    c.lock()

    # All runs of one comparison should use one GPU type; --fast then only adds
    # numerical noise, small next to the variation across training seeds.
    if a.fast:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    train(c, device=device)


if __name__ == "__main__":
    main()
