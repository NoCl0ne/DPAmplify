"""Aggregation strategies for DP-FedAvg experiments."""

from fl_system.aggregators.fedavg import fedavg_aggregate
from fl_system.aggregators.krum import krum_aggregate
from fl_system.aggregators.trimmed_mean import trimmed_mean_aggregate

__all__ = ["fedavg_aggregate", "krum_aggregate", "trimmed_mean_aggregate"]
