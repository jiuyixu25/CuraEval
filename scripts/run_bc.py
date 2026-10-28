"""Train robomimic's BC-RNN-GMM on a low_dim dataset, optionally restricted to
one filter_key (a demonstration subset).

The configuration is robomimic's BC-RNN low_dim recipe for the MH datasets: the
bc template with rnn.enabled (2-layer LSTM, hidden size 400), sequence length
10, a GMM action head, no actor MLP and learning rate 1e-4. The defaults are
the paper's settings (2000 epochs x 100 steps); every run in the paper used
--fast.

The command line mirrors run_dp.py; --dry prints the locked config without
training.
"""
import argparse

from robomimic.config import config_factory
import robomimic.utils.torch_utils as TorchUtils
from robomimic.scripts.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--filter_key", default="none")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--rollout_n", type=int, default=50, help="training-time rollouts; 0 disables them")
    p.add_argument("--rollout_rate", type=int, default=0, help="epochs between rollouts; 0 = after the last epoch only")
    p.add_argument("--horizon", type=int, default=400, help="rollout horizon: can 400, square 500, transport 700")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seq", type=int, default=10, help="RNN horizon = train seq_length")
    p.add_argument("--fast", action="store_true",
                   help="TF32 and cudnn.benchmark with non-deterministic kernels")
    p.add_argument("--dry", action="store_true", help="print the config and exit without training")
    a = p.parse_args()

    c = config_factory("bc")
    with c.values_unlocked():
        c.experiment.name = a.name
        c.train.data = a.dataset
        c.train.output_dir = a.output_dir
        c.train.seed = a.seed
        c.train.num_epochs = a.epochs
        c.experiment.epoch_every_n_steps = a.steps_per_epoch
        c.train.hdf5_cache_mode = "all"
        if a.filter_key != "none":
            c.train.hdf5_filter_key = a.filter_key
        # robomimic's BC-RNN-GMM low_dim recipe
        c.algo.rnn.enabled = True
        c.algo.rnn.horizon = a.seq
        c.train.seq_length = a.seq          # must equal rnn.horizon
        c.algo.gmm.enabled = True           # GMM action head
        c.algo.actor_layer_dims = []        # no actor MLP after the LSTM
        # LR 1e-4 / hidden 400 / 2-layer LSTM are the template defaults already.
        c.experiment.validate = False
        c.experiment.rollout.enabled = a.rollout_n > 0
        c.experiment.rollout.n = max(1, a.rollout_n)
        c.experiment.rollout.horizon = a.horizon
        c.experiment.rollout.rate = a.rollout_rate if a.rollout_rate > 0 else a.epochs
        c.experiment.rollout.terminate_on_success = True
        c.experiment.save.enabled = True
        c.experiment.save.on_best_rollout_success_rate = True
    c.lock()

    if a.dry:
        print(c)
        return

    import torch
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
