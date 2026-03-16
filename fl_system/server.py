"""
fl_system/server.py — Flower server for DPAmplify experiments.

Supports three aggregation strategies:
  "fedavg"       — standard Federated Averaging (FedAvg)
  "krum"         — Krum Byzantine-robust aggregation
  "trimmed_mean" — coordinate-wise trimmed mean
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Strategy

from fl_system.aggregators import (
    fedavg_aggregate,
    krum_aggregate,
    trimmed_mean_aggregate,
)

logger = logging.getLogger(__name__)


# ── Custom strategies wrapping robust aggregators ─────────────────────

class _KrumStrategy(FedAvg):
    """FedAvg subclass that replaces aggregate_fit with Krum selection."""

    def __init__(self, f: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.f = f

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        gradients = [
            parameters_to_ndarrays(fit_res.parameters)[0]
            for _, fit_res in results
        ]
        selected = krum_aggregate(gradients, f=self.f)
        return ndarrays_to_parameters([selected]), {}


class _TrimmedMeanStrategy(FedAvg):
    """FedAvg subclass that replaces aggregate_fit with trimmed mean."""

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        gradients = [
            parameters_to_ndarrays(fit_res.parameters)[0]
            for _, fit_res in results
        ]
        aggregated = trimmed_mean_aggregate(gradients, beta=self.beta)
        return ndarrays_to_parameters([aggregated]), {}


# ── Public factory ────────────────────────────────────────────────────

def create_strategy(
    aggregator_name: str,
    min_clients: int,
    fraction_fit: float = 1.0,
) -> Strategy:
    """Create a Flower aggregation strategy by name.

    Args:
        aggregator_name: one of "fedavg", "krum", "trimmed_mean"
        min_clients:     minimum number of clients required per round
        fraction_fit:    fraction of available clients to sample per round

    Returns:
        A Flower Strategy instance

    Raises:
        ValueError: if aggregator_name is not recognised
    """
    common_kwargs = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )

    name = aggregator_name.lower()
    if name == "fedavg":
        return FedAvg(**common_kwargs)
    elif name == "krum":
        return _KrumStrategy(f=1, **common_kwargs)
    elif name == "trimmed_mean":
        return _TrimmedMeanStrategy(beta=0.1, **common_kwargs)
    else:
        raise ValueError(
            f"Unknown aggregator '{aggregator_name}'. "
            "Choose from: fedavg, krum, trimmed_mean"
        )


# ── Server entry point ────────────────────────────────────────────────

def run_server(
    aggregator_name: str = "fedavg",
    n_rounds: int = 100,
    min_clients: int = 2,
    server_address: str = "[::]:8080",
) -> None:
    """Start a Flower server for a DPAmplify experiment.

    Logs the aggregated gradient norm to logs/gradient_norms.csv
    at the end of each round.

    Args:
        aggregator_name: aggregation strategy (see create_strategy)
        n_rounds:        total number of FL rounds
        min_clients:     minimum clients required to start a round
        server_address:  gRPC address for the Flower server
    """
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "gradient_norms.csv")

    strategy = create_strategy(aggregator_name, min_clients)

    logger.info(
        "Starting Flower server: aggregator=%s, rounds=%d, address=%s",
        aggregator_name,
        n_rounds,
        server_address,
    )

    # Wrap strategy to log gradient norms
    original_aggregate_fit = strategy.aggregate_fit

    def _logging_aggregate_fit(server_round, results, failures):
        agg_params, metrics = original_aggregate_fit(
            server_round, results, failures
        )
        if agg_params is not None:
            arrays = parameters_to_ndarrays(agg_params)
            norm = float(np.linalg.norm(arrays[0])) if arrays else 0.0
            with open(log_path, "a", newline="") as fh:
                csv.writer(fh).writerow([server_round, norm])
            logger.info("Round %d: aggregated norm = %.4f", server_round, norm)
        return agg_params, metrics

    strategy.aggregate_fit = _logging_aggregate_fit

    # Write CSV header if file is new
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["round", "aggregated_norm"])

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )
