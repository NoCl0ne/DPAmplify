"""
attack/byzantine_client.py — DPAmplify Byzantine Flower client.

Per ethical policy, full public release is scheduled 90 days after
paper publication.

Two-phase strategy:
  Phase 1 (rounds 1..T_est): Honest behaviour.  The client trains
    normally and records gradient norms to estimate C and σ via
    PassiveParameterEstimator.
  Phase 2 (rounds T_est+1..): Attack.  The client submits
    g_adv = C̃ · (g_target / ‖g_target‖₂) for each parameter layer,
    exploiting the zero-bias property of the Gaussian DP mechanism.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import flwr as fl

from attack.parameter_estimator import PassiveParameterEstimator
from attack.gradient_optimizer import GradientOptimizer
from theory.dp_mechanism import DPMechanism

logger = logging.getLogger(__name__)


class DPAmplifyClient(fl.client.NumPyClient):
    """Byzantine Flower client implementing the DPAmplify two-phase attack.

    Attributes:
        cid      (str): client identifier
        T_est    (int): number of honest estimation rounds before attack
        device   (str): torch device string
    """

    def __init__(
        self,
        cid: str,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        g_target: np.ndarray,
        estimator: PassiveParameterEstimator,
        T_est: int = 20,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            cid:         unique client identifier
            model:       PyTorch model (shared architecture with honest clients)
            train_loader: DataLoader for local training data (used in est. phase)
            g_target:    target direction for the attack (any non-zero vector)
            estimator:   PassiveParameterEstimator instance (shared or fresh)
            T_est:       number of honest rounds before switching to attack
            device:      torch device ("cpu" or "cuda")
        """
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.g_target = g_target
        self.estimator = estimator
        self.T_est = T_est
        self.device = device
        self._round: int = 0

    # ── Flower NumPyClient interface ──────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters as a list of NumPy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Participate in a training round.

        Returns honest parameters during the estimation phase; returns
        adversarial gradients during the attack phase.
        """
        self._round += 1
        self._set_parameters(parameters)

        if self._is_attack_phase():
            return self._attack_fit()
        else:
            return self._honest_fit()

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """Honest evaluation — always returns true loss and accuracy."""
        self._set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        n_samples = 0

        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * len(targets)
                correct += (outputs.argmax(dim=1) == targets).sum().item()
                n_samples += len(targets)

        avg_loss = total_loss / max(n_samples, 1)
        accuracy = correct / max(n_samples, 1)
        return avg_loss, n_samples, {"accuracy": accuracy}

    # ── Phase detection ───────────────────────────────────────────────

    def _is_attack_phase(self) -> bool:
        """True once T_est estimation rounds have elapsed and estimator is ready."""
        return self._round > self.T_est and self.estimator.is_ready()

    # ── Honest estimation phase ───────────────────────────────────────

    def _honest_fit(self) -> Tuple[List[np.ndarray], int, Dict]:
        """Train one epoch honestly, record gradient norm, return update."""
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        n_samples = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Record gradient norm for parameter estimation
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5
            self.estimator.update(grad_norm)

            optimizer.step()
            n_samples += len(targets)

        logger.info(
            "[Byzantine %s] Round %d: ESTIMATION (norm recorded)",
            self.cid,
            self._round,
        )
        return self.get_parameters({}), n_samples, {}

    # ── Attack phase ──────────────────────────────────────────────────

    def _attack_fit(self) -> Tuple[List[np.ndarray], int, Dict]:
        """Submit adversarial gradients aligned with g_target."""
        C_hat = self.estimator.estimate_C()
        adv_params: List[np.ndarray] = []

        for p in self.model.parameters():
            layer_shape = p.shape
            d = int(np.prod(layer_shape))
            # Build per-layer target slice (first d components of g_target,
            # padded/truncated to match layer size)
            g_target_flat = self.g_target.flat
            target_slice = np.array(
                [next(g_target_flat, 0.0) for _ in range(d)],
                dtype=np.float32,
            )
            if np.linalg.norm(target_slice) < 1e-10:
                # Fallback: use first canonical basis vector for this layer
                target_slice = np.zeros(d, dtype=np.float32)
                target_slice[0] = 1.0

            opt = GradientOptimizer(target_slice, C=C_hat)
            g_adv_flat = opt.compute_g_adv().astype(np.float32)
            adv_params.append(g_adv_flat.reshape(layer_shape))

        logger.info(
            "[Byzantine %s] Round %d: ATTACK (C_hat=%.4f)",
            self.cid,
            self._round,
            C_hat,
        )
        return adv_params, 1, {"attack": True, "C_hat": C_hat}

    # ── Utility ───────────────────────────────────────────────────────

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        for p, new_val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_val, dtype=p.dtype).to(self.device)
