"""
fl_system/honest_client.py — Honest Flower client with Opacus DP-SGD.

Each honest client trains its local model with differentially-private SGD
(Opacus), clipping per-sample gradients to C and adding Gaussian noise
with standard deviation σ before sending the update to the server.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import opacus
import flwr as fl

logger = logging.getLogger(__name__)


class HonestClient(fl.client.NumPyClient):
    """Flower client that trains with Opacus DP-SGD.

    Attributes:
        cid          (str):   client identifier
        C            (float): per-sample gradient clipping threshold
        sigma        (float): DP noise multiplier
        local_epochs (int):   training epochs per FL round
        device       (str):   torch device
    """

    def __init__(
        self,
        cid: str,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        C: float = 1.0,
        sigma: float = 0.1,
        local_epochs: int = 2,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            cid:          unique client identifier
            model:        PyTorch model
            train_loader: DataLoader for local training data
            test_loader:  DataLoader for local test data
            C:            per-sample gradient clipping norm (Opacus max_grad_norm)
            sigma:        DP noise multiplier (Opacus noise_multiplier)
            local_epochs: number of local training epochs per FL round
            device:       "cpu" or "cuda"
        """
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.C = C
        self.sigma = sigma
        self.local_epochs = local_epochs
        self.device = device

    # ── Flower NumPyClient interface ──────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters as a list of NumPy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Train locally with Opacus DP-SGD and return updated parameters.

        Attaches a fresh PrivacyEngine each round so that the DP
        accounting is per-round (stateless from the server's perspective).

        Args:
            parameters: global model parameters from the server
            config:     round configuration dict (currently unused)

        Returns:
            (updated_parameters, n_samples, metrics_dict)
        """
        self._set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Attach Opacus DP-SGD
        privacy_engine = opacus.PrivacyEngine()
        dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.sigma,
            max_grad_norm=self.C,
        )

        n_samples = 0
        for _ in range(self.local_epochs):
            for inputs, targets in dp_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                dp_optimizer.zero_grad()
                outputs = dp_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                dp_optimizer.step()
                n_samples += len(targets)

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        logger.info(
            "[Honest %s] local training done: ε=%.2f (δ=1e-5)",
            self.cid,
            epsilon,
        )
        return self.get_parameters({}), n_samples, {"epsilon": epsilon}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """Evaluate the global model on local test data.

        Args:
            parameters: global model parameters
            config:     configuration dict (currently unused)

        Returns:
            (loss, n_samples, {"accuracy": float})
        """
        self._set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        n_samples = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
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

    # ── Utility ───────────────────────────────────────────────────────

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load a list of NumPy arrays into the model's parameters."""
        for p, new_val in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_val, dtype=p.dtype).to(self.device)
