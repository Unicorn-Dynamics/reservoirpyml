"""
Mesh Coordinator for Distributed Reservoir Networks

Manages coordination, synchronization, and topology discovery for
distributed reservoir computing nodes in the cognitive mesh.
"""

import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np

from ..ggml.core import GGMLTensor, GGMLContext
from ..ggml.reservoir_backend import GGMLReservoir
from ..attention.ecan import ECANAttentionSystem


@dataclass
class ReservoirNode:
    """Represents a distributed reservoir node in the cognitive mesh."""
    
    node_id: str
    host: str = "localhost"
    port: int = 8080
    reservoir: Optional[GGMLReservoir] = None
    state: Optional[np.ndarray] = None
    last_update: float = field(default_factory=time.time)
    status: str = "active"
    capabilities: Set[str] = field(default_factory=set)
    load: float = 0.0
    connected_nodes: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize reservoir if not provided."""
        if self.reservoir is None:
            # Create a simple mock reservoir for demo purposes
            class SimpleReservoir:
                def __init__(self, units):
                    self.units = units
                    self.output_dim = units
                    
                def __call__(self, input_data):
                    # Simple linear transformation for demo
                    return np.random.randn(1, self.units) * 0.1
            
            self.reservoir = SimpleReservoir(100)
            
        # Initialize state if not provided
        if self.state is None:
            output_dim = getattr(self.reservoir, 'output_dim', 100)
            self.state = np.zeros((1, output_dim))
    
    def update_state(self, new_state: np.ndarray):
        """Update node state and timestamp."""
        self.state = new_state.copy()
        self.last_update = time.time()
    
    def process_input(self, input_data: np.ndarray) -> np.ndarray:
        """Process input through this node's reservoir."""
        output = self.reservoir(input_data)
        self.update_state(output)
        return output
    
    def get_connection_weight(self, target_node_id: str) -> float:
        """Get connection weight to target node."""
        # Simple distance-based weighting for now
        if target_node_id in self.connected_nodes:
            return 1.0 / (len(self.connected_nodes) + 1)
        return 0.0
    
    def add_connection(self, node_id: str):
        """Add connection to another node."""
        self.connected_nodes.add(node_id)
    
    def remove_connection(self, node_id: str):
        """Remove connection to another node."""
        self.connected_nodes.discard(node_id)


class MeshCoordinator:
    """
    Coordinates distributed reservoir nodes in the cognitive mesh.
    
    Responsibilities:
    - Node registration and discovery
    - State synchronization across nodes
    - Topology management and optimization
    - Load balancing and fault tolerance
    - Task distribution and coordination
    """
    
    def __init__(self):
        self.nodes: Dict[str, ReservoirNode] = {}
        self.topology: Dict[str, Set[str]] = {}
        self.global_state: Dict[str, Any] = {}
        self.sync_lock = threading.Lock()
        
        # Performance monitoring
        self.performance_history: List[Dict[str, Any]] = []
        self.sync_timestamps: Dict[str, float] = {}
        
        # Task distribution
        self.task_queue: List[Dict[str, Any]] = []
        self.task_assignments: Dict[str, List[str]] = {}  # task_id -> node_ids
        
        # Initialize with local node
        self._create_local_node()
    
    def _create_local_node(self):
        """Create the local reservoir node."""
        local_id = f"local_{uuid.uuid4().hex[:8]}"
        local_node = ReservoirNode(
            node_id=local_id,
            host="localhost",
            port=8080
        )
        self.register_node(local_node)
    
    def register_node(self, node: ReservoirNode) -> bool:
        """Register a new node in the mesh."""
        with self.sync_lock:
            if node.node_id in self.nodes:
                return False
            
            self.nodes[node.node_id] = node
            self.topology[node.node_id] = set()
            self.sync_timestamps[node.node_id] = time.time()
            
            # Auto-connect to existing nodes (simple mesh topology)
            for existing_id in list(self.nodes.keys()):
                if existing_id != node.node_id:
                    self._create_connection(node.node_id, existing_id)
            
            return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node from the mesh."""
        with self.sync_lock:
            if node_id not in self.nodes:
                return False
            
            # Remove all connections
            for connected_id in list(self.topology[node_id]):
                self._remove_connection(node_id, connected_id)
            
            # Remove node
            del self.nodes[node_id]
            del self.topology[node_id]
            del self.sync_timestamps[node_id]
            
            return True
    
    def _create_connection(self, node1_id: str, node2_id: str):
        """Create bidirectional connection between nodes."""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.topology[node1_id].add(node2_id)
            self.topology[node2_id].add(node1_id)
            self.nodes[node1_id].add_connection(node2_id)
            self.nodes[node2_id].add_connection(node1_id)
    
    def _remove_connection(self, node1_id: str, node2_id: str):
        """Remove bidirectional connection between nodes."""
        self.topology[node1_id].discard(node2_id)
        self.topology[node2_id].discard(node1_id)
        if node1_id in self.nodes:
            self.nodes[node1_id].remove_connection(node2_id)
        if node2_id in self.nodes:
            self.nodes[node2_id].remove_connection(node1_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get mesh coordinator status."""
        with self.sync_lock:
            return {
                "total_nodes": len(self.nodes),
                "active_nodes": len([n for n in self.nodes.values() if n.status == "active"]),
                "total_connections": sum(len(connections) for connections in self.topology.values()) // 2,
                "sync_status": self._calculate_sync_health(),
                "performance_summary": self._get_performance_summary()
            }
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get information about all nodes."""
        with self.sync_lock:
            return [
                {
                    "node_id": node.node_id,
                    "host": node.host,
                    "port": node.port,
                    "status": node.status,
                    "load": node.load,
                    "last_update": node.last_update,
                    "connections": len(node.connected_nodes),
                    "state_shape": list(node.state.shape) if node.state is not None else None
                }
                for node in self.nodes.values()
            ]
    
    def get_topology(self) -> Dict[str, List[str]]:
        """Get mesh topology as adjacency list."""
        with self.sync_lock:
            return {
                node_id: list(connections)
                for node_id, connections in self.topology.items()
            }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status across the mesh."""
        with self.sync_lock:
            current_time = time.time()
            return {
                "sync_timestamps": dict(self.sync_timestamps),
                "sync_delays": {
                    node_id: current_time - timestamp
                    for node_id, timestamp in self.sync_timestamps.items()
                },
                "overall_health": self._calculate_sync_health()
            }
    
    def get_node_state(self, node_id: str) -> Dict[str, Any]:
        """Get state of specific node."""
        with self.sync_lock:
            if node_id not in self.nodes:
                return {"error": f"Node {node_id} not found"}
            
            node = self.nodes[node_id]
            return {
                "node_id": node_id,
                "state": node.state.tolist() if node.state is not None else None,
                "state_shape": list(node.state.shape) if node.state is not None else None,
                "last_update": node.last_update,
                "status": node.status,
                "load": node.load
            }
    
    def get_all_states(self) -> Dict[str, Any]:
        """Get states of all nodes."""
        with self.sync_lock:
            return {
                node_id: {
                    "state": node.state.tolist() if node.state is not None else None,
                    "last_update": node.last_update,
                    "load": node.load
                }
                for node_id, node in self.nodes.items()
            }
    
    def get_global_state(self) -> Dict[str, Any]:
        """Get aggregated global mesh state."""
        with self.sync_lock:
            if not self.nodes:
                return {"error": "No nodes available"}
            
            # Aggregate states
            all_states = []
            total_load = 0.0
            
            for node in self.nodes.values():
                if node.state is not None and node.status == "active":
                    all_states.append(node.state)
                    total_load += node.load
            
            if not all_states:
                return {"error": "No active node states available"}
            
            # Calculate global aggregations
            mean_state = np.mean(all_states, axis=0)
            variance_state = np.var(all_states, axis=0)
            
            return {
                "mean_state": mean_state.tolist(),
                "variance_state": variance_state.tolist(),
                "active_nodes": len(all_states),
                "total_load": total_load,
                "average_load": total_load / len(all_states) if all_states else 0.0
            }
    
    def update_node_state(self, node_id: str, new_state: np.ndarray) -> Dict[str, Any]:
        """Update state of specific node and propagate changes."""
        with self.sync_lock:
            if node_id not in self.nodes:
                return {"error": f"Node {node_id} not found"}
            
            # Update node state
            self.nodes[node_id].update_state(new_state)
            self.sync_timestamps[node_id] = time.time()
            
            # Propagate to connected nodes (simplified)
            propagation_results = {}
            for connected_id in self.topology[node_id]:
                if connected_id in self.nodes:
                    # Simple state influence (weighted average)
                    weight = self.nodes[node_id].get_connection_weight(connected_id)
                    connected_node = self.nodes[connected_id]
                    if connected_node.state is not None:
                        influenced_state = (
                            connected_node.state * (1 - weight) + 
                            new_state * weight
                        )
                        connected_node.update_state(influenced_state)
                        propagation_results[connected_id] = "influenced"
            
            return {
                "status": "updated",
                "node_id": node_id,
                "state_shape": list(new_state.shape),
                "propagated_to": list(propagation_results.keys())
            }
    
    def process_distributed_input(self, input_tensor: GGMLTensor, agent_id: str) -> Dict[str, Any]:
        """Process input across the distributed mesh."""
        with self.sync_lock:
            if not self.nodes:
                return {"error": "No nodes available"}
            
            # Select optimal node for processing
            selected_node_id = self._select_optimal_node(input_tensor, agent_id)
            selected_node = self.nodes[selected_node_id]
            
            # Process input
            input_data = input_tensor.data
            output_data = selected_node.process_input(input_data)
            
            # Update node load
            selected_node.load += 0.1  # Simple load increment
            
            # Synchronize with connected nodes
            sync_results = self._synchronize_node(selected_node_id)
            
            return {
                "status": "processed",
                "processing_node": selected_node_id,
                "input_shape": list(input_data.shape),
                "output_shape": list(output_data.shape),
                "agent_id": agent_id,
                "synchronization": sync_results
            }
    
    def distribute_task(self, task_id: str, task_data: Dict[str, Any], target_agents: List[str]) -> Dict[str, Any]:
        """Distribute a task across the mesh."""
        with self.sync_lock:
            # Add task to queue
            task_info = {
                "task_id": task_id,
                "data": task_data,
                "target_agents": target_agents,
                "timestamp": time.time(),
                "status": "queued"
            }
            self.task_queue.append(task_info)
            
            # Select nodes for task execution
            selected_nodes = self._select_nodes_for_task(task_data, target_agents)
            self.task_assignments[task_id] = selected_nodes
            
            # Update node loads
            load_increment = 1.0 / len(selected_nodes) if selected_nodes else 1.0
            for node_id in selected_nodes:
                if node_id in self.nodes:
                    self.nodes[node_id].load += load_increment
            
            return {
                "status": "distributed",
                "task_id": task_id,
                "assigned_nodes": selected_nodes,
                "load_distribution": {
                    node_id: self.nodes[node_id].load 
                    for node_id in selected_nodes if node_id in self.nodes
                }
            }
    
    def _select_optimal_node(self, input_tensor: GGMLTensor, agent_id: str) -> str:
        """Select optimal node for processing based on load and capabilities."""
        active_nodes = [
            (node_id, node) for node_id, node in self.nodes.items() 
            if node.status == "active"
        ]
        
        if not active_nodes:
            return list(self.nodes.keys())[0]  # Fallback to first node
        
        # Simple load-based selection
        min_load_node = min(active_nodes, key=lambda x: x[1].load)
        return min_load_node[0]
    
    def _select_nodes_for_task(self, task_data: Dict[str, Any], target_agents: List[str]) -> List[str]:
        """Select nodes for task execution."""
        active_nodes = [
            node_id for node_id, node in self.nodes.items() 
            if node.status == "active"
        ]
        
        # Simple round-robin selection
        num_nodes = min(len(active_nodes), max(1, len(target_agents)))
        return active_nodes[:num_nodes]
    
    def _synchronize_node(self, node_id: str) -> Dict[str, Any]:
        """Synchronize node with its neighbors."""
        if node_id not in self.nodes:
            return {"error": "Node not found"}
        
        node = self.nodes[node_id]
        sync_results = {}
        
        # Synchronize with connected nodes
        for connected_id in self.topology[node_id]:
            if connected_id in self.nodes:
                connected_node = self.nodes[connected_id]
                
                # Simple synchronization: average states
                if node.state is not None and connected_node.state is not None:
                    sync_weight = 0.1  # Small synchronization influence
                    avg_state = (node.state + connected_node.state) / 2
                    
                    # Apply synchronization
                    node.state = node.state * (1 - sync_weight) + avg_state * sync_weight
                    connected_node.state = connected_node.state * (1 - sync_weight) + avg_state * sync_weight
                    
                    sync_results[connected_id] = "synchronized"
        
        # Update timestamp
        self.sync_timestamps[node_id] = time.time()
        
        return sync_results
    
    def _calculate_sync_health(self) -> float:
        """Calculate overall synchronization health (0-1)."""
        if not self.sync_timestamps:
            return 0.0
        
        current_time = time.time()
        max_delay = 10.0  # Maximum acceptable delay in seconds
        
        delays = [
            current_time - timestamp 
            for timestamp in self.sync_timestamps.values()
        ]
        
        avg_delay = sum(delays) / len(delays)
        health = max(0.0, 1.0 - (avg_delay / max_delay))
        
        return health
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across the mesh."""
        if not self.nodes:
            return {"error": "No nodes available"}
        
        loads = [node.load for node in self.nodes.values()]
        
        return {
            "average_load": sum(loads) / len(loads),
            "max_load": max(loads),
            "min_load": min(loads),
            "load_variance": np.var(loads),
            "active_tasks": len(self.task_queue),
            "node_count": len(self.nodes)
        }