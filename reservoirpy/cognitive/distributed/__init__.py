"""
Distributed Cognitive Mesh API & Embodiment Layer

This module implements the distributed cognitive mesh infrastructure for
multi-agent cognitive systems with embodied cognition capabilities.

Components:
- API Server: REST/WebSocket endpoints for cognitive mesh operations
- Mesh Coordinator: Distributed reservoir coordination and state sync
- Embodiment: Interfaces for Unity3D, ROS, and web-based agents
- Orchestration: Multi-agent task distribution and load balancing
"""

from .api_server import CognitiveMeshAPI
from .mesh_coordinator import MeshCoordinator, ReservoirNode
from .orchestrator import AgentOrchestrator, TaskDistributor, TaskPriority

__all__ = [
    'CognitiveMeshAPI',
    'MeshCoordinator', 
    'ReservoirNode',
    'AgentOrchestrator',
    'TaskDistributor',
    'TaskPriority'
]