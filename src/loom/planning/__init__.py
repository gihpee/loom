# Adapted from Parallax (https://github.com/GradientHQ/parallax, arXiv:2509.26182).
# Original: src/scheduling/__init__.py (empty) + public surface assembled from
# src/scheduling/{model_info,node,layer_allocation,node_management,request_routing,scheduler}.py.
# Изменения: собран явный программный API библиотеки планирования; capacity (c_i)
# передаётся явным параметром ShardCapacity от Resource Broker.
"""Loom planning library: Phase-1 layer allocation + Phase-2 request routing.

This is a programmatic (non-CLI) library. One ``Scheduler`` instance manages
exactly one model over the sub-pool of nodes granted by the Resource Broker;
the orchestrator's Scheduler Pool creates one instance per model.
"""

from loom.planning.capacity import ShardCapacity
from loom.planning.layer_allocation import (
    BaseLayerAllocator,
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
)
from loom.planning.model_info import ModelInfo
from loom.planning.node import Node, NodeHardwareInfo, RequestSignal
from loom.planning.node_management import NodeManager, NodeState, Pipeline
from loom.planning.request_routing import (
    DynamicProgrammingRouting,
    RandomizedOverDynamicPipelinesRouting,
    RequestRoutingStrategy,
    RoundRobinOverFixedPipelinesRouting,
)
from loom.planning.scheduler import Scheduler

__all__ = [
    "ShardCapacity",
    "ModelInfo",
    "Node",
    "NodeHardwareInfo",
    "RequestSignal",
    "NodeManager",
    "NodeState",
    "Pipeline",
    "BaseLayerAllocator",
    "GreedyLayerAllocator",
    "DynamicProgrammingLayerAllocator",
    "RequestRoutingStrategy",
    "DynamicProgrammingRouting",
    "RandomizedOverDynamicPipelinesRouting",
    "RoundRobinOverFixedPipelinesRouting",
    "Scheduler",
]
