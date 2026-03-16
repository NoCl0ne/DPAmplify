"""
experiments/exp_02_mnist_attack.py — DPAmplify attack on MNIST in simulation.

Runs a direct FedAvg simulation (no real Flower network — fit() called in
a loop) to measure the attack's effect on a global MLP trained on MNIST.

Architecture: MLP 784 → 128 → 10 (ReLU, no bias in hidden layer)
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from theory.dp_mechanism import DPMechanism
from attack.parameter_estimator import PassiveParameterEstimator
from attack.gradient_optimizer import GradientOptimizer
from fl_system.aggregators import fedavg_aggregate, krum_aggregate, trimmed_mean_aggregate


# ── Model ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# ── IID data split ────────────────────────────────────────────────────

def iid_split(dataset, n_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    shard_size = len(dataset) // n_clients
    return [
        Subset(dataset, indices[i * shard_size: (i + 1) * shard_size])
        for i in range(n_clients)
    ]


# ── Honest local training (single epoch, with DP clipping) ───────────

def honest_fit(
    model: MLP,
    loader: DataLoader,
    C: float,
    sigma: float,
    device: str,
) -> list:
    """Train one epoch with gradient clipping + Gaussian noise (manual DP-SGD)."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    mech = DPMechanism(C=C, sigma=sigma)
    rng = np.random.default_rng()

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Per-layer DP: clip + noise
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    g_np = p.grad.cpu().numpy().ravel()
                    g_dp = mech.apply(g_np, rng)
                    p.grad.copy_(
                        torch.tensor(
                            g_dp.reshape(p.grad.shape), dtype=p.grad.dtype
                        )
                    )
        optimizer.step()

    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


# ── Byzantine gradient construction ──────────────────────────────────

def byzantine_fit(
    global_params: list,
    estimator: PassiveParameterEstimator,
    g_target: np.ndarray,
) -> list:
    C_hat = estimator.estimate_C()
    adv_params = []
    target_iter = iter(g_target.ravel().tolist())
    for p_np in global_params:
        d = int(np.prod(p_np.shape))
        target_slice = np.array(
            [next(target_iter, 0.0) for _ in range(d)], dtype=np.float32
        )
        if np.linalg.norm(target_slice) < 1e-10:
            target_slice = np.zeros(d, dtype=np.float32)
            target_slice[0] = 1.0
        opt = GradientOptimizer(target_slice, C=C_hat)
        adv_params.append(opt.compute_g_adv().reshape(p_np.shape))
    return adv_params


# ── Global model evaluation ───────────────────────────────────────────

def evaluate(model: MLP, loader: DataLoader, device: str):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            total_loss += criterion(out, targets).item() * len(targets)
            correct += (out.argmax(1) == targets).sum().item()
            n += len(targets)
    return total_loss / max(n, 1), correct / max(n, 1)


# ── Aggregation helper ────────────────────────────────────────────────

def aggregate(param_lists: list, aggregator: str, n_byzantine: int) -> list:
    n_layers = len(param_lists[0])
    result = []
    for layer_idx in range(n_layers):
        grads = [pl[layer_idx].ravel() for pl in param_lists]
        if aggregator == "fedavg":
            agg = fedavg_aggregate(grads)
        elif aggregator == "krum":
            f = n_byzantine
            agg = krum_aggregate(grads, f=f)
        else:  # trimmed_mean
            agg = trimmed_mean_aggregate(grads, beta=0.1)
        result.append(agg.reshape(param_lists[0][layer_idx].shape))
    return result


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DPAmplify MNIST simulation")
    p.add_argument("--n_clients",    type=int,   default=20)
    p.add_argument("--n_byzantine",  type=int,   default=3)
    p.add_argument("--clipping_C",   type=float, default=1.0)
    p.add_argument("--noise_sigma",  type=float, default=0.1)
    p.add_argument("--n_rounds",     type=int,   default=100)
    p.add_argument("--T_est",        type=int,   default=20)
    p.add_argument("--aggregator",   type=str,   default="fedavg",
                   choices=["fedavg", "krum", "trimmed_mean"])
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--save_results", type=str,   default="results/exp_02_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=transform)
    shards = iid_split(train_ds, args.n_clients, args.seed)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Client loaders
    loaders = [
        DataLoader(shard, batch_size=32, shuffle=True)
        for shard in shards
    ]

    # Global model
    global_model = MLP().to(device)
    global_params = [p.detach().cpu().numpy().copy() for p in global_model.parameters()]

    # Estimators (one per Byzantine client, shared history for simplicity)
    estimator = PassiveParameterEstimator(history_window=args.T_est)

    # Attack target: e_1 in flattened parameter space
    flat_dim = sum(int(np.prod(p.shape)) for p in global_model.parameters())
    g_target = np.zeros(flat_dim, dtype=np.float32)
    g_target[0] = 1.0

    n_honest = args.n_clients - args.n_byzantine
    history = []

    for rnd in range(1, args.n_rounds + 1):
        # Load global parameters
        with torch.no_grad():
            for p, v in zip(global_model.parameters(), global_params):
                p.copy_(torch.tensor(v))

        updates = []

        # Honest clients
        for i in range(n_honest):
            up = honest_fit(
                global_model, loaders[i],
                args.clipping_C, args.noise_sigma, device
            )
            # Estimate clipping from honest gradient norm
            norms = [np.linalg.norm(u.ravel()) for u in up]
            estimator.update(float(np.mean(norms)))
            updates.append(up)

        # Byzantine clients
        for i in range(args.n_byzantine):
            if rnd <= args.T_est or not estimator.is_ready():
                # Still estimating: submit honest gradient
                up = honest_fit(
                    global_model, loaders[n_honest + i],
                    args.clipping_C, args.noise_sigma, device
                )
            else:
                up = byzantine_fit(global_params, estimator, g_target)
            updates.append(up)

        # Aggregate
        new_params = aggregate(updates, args.aggregator, args.n_byzantine)

        # Apply to global model
        with torch.no_grad():
            for p, v in zip(global_model.parameters(), new_params):
                p.copy_(torch.tensor(v))
        global_params = [p.detach().cpu().numpy().copy() for p in global_model.parameters()]

        # Evaluate every 10 rounds
        if rnd % 10 == 0 or rnd == args.n_rounds:
            loss, acc = evaluate(global_model, test_loader, device)
            history.append({"round": rnd, "loss": loss, "accuracy": acc})
            print(f"Round {rnd:3d}: loss={loss:.4f}, accuracy={acc:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    result = {
        "config": vars(args),
        "history": history,
        "final_accuracy": history[-1]["accuracy"] if history else None,
    }
    with open(args.save_results, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.save_results}")
    print(f"Final accuracy: {history[-1]['accuracy']:.4f}")


if __name__ == "__main__":
    main()
