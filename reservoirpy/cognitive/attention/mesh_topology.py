"""
Dynamic Mesh Topology Management
===============================

Implements dynamic mesh topology modification based on attention flow,
adaptive network structures, and distributed reservoir agent coordination.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .ecan import ECANAttentionSystem, AttentionValue
from .attention_reservoir import AttentionReservoir, AttentionFlow


class TopologyChangeType(Enum):
    """Types of topology modifications."""
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node" 
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    MODIFY_WEIGHT = "modify_weight"
    CLUSTER_FORMATION = "cluster_formation"
    CLUSTER_DISSOLUTION = "cluster_dissolution"


@dataclass
class TopologyChange:
    """
    Record of a topology modification event.
    
    Parameters
    ----------
    change_id : str
        Unique identifier for the change
    change_type : TopologyChangeType
        Type of topology modification
    source_node : str, optional
        Source node for connection changes
    target_node : str, optional
        Target node for connection changes
    attention_driver : float, optional
        Attention value that triggered the change
    timestamp : float
        When the change occurred
    success : bool, default=True
        Whether the change was successfully applied
    metadata : dict, optional
        Additional metadata about the change
    """
    change_id: str
    change_type: TopologyChangeType
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    attention_driver: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class MeshTopology:
    """
    Dynamic mesh topology manager for attention-driven reservoir networks.
    
    Manages network topology changes based on attention flow patterns,
    implements adaptive clustering, and coordinates distributed reservoir agents.
    
    Parameters
    ----------
    attention_system : ECANAttentionSystem
        ECAN attention allocation system
    attention_flow : AttentionFlow
        Attention flow management system
    max_nodes : int, default=100
        Maximum number of nodes in the mesh
    topology_update_interval : float, default=5.0
        Seconds between topology update cycles
    connection_threshold : float, default=3.0
        Attention threshold for creating new connections
    removal_threshold : float, default=0.5
        Attention threshold below which connections are removed
    clustering_threshold : float, default=10.0
        Attention threshold for cluster formation
    """
    
    def __init__(
        self,
        attention_system: ECANAttentionSystem,
        attention_flow: AttentionFlow,
        max_nodes: int = 100,
        topology_update_interval: float = 5.0,
        connection_threshold: float = 3.0,
        removal_threshold: float = 0.5,
        clustering_threshold: float = 10.0
    ):
        self.attention_system = attention_system
        self.attention_flow = attention_flow
        self.max_nodes = max_nodes
        self.topology_update_interval = topology_update_interval
        self.connection_threshold = connection_threshold
        self.removal_threshold = removal_threshold
        self.clustering_threshold = clustering_threshold
        
        # Network topology
        self.topology_graph = nx.DiGraph()
        self.reservoir_nodes: Dict[str, AttentionReservoir] = {}
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Dynamic topology state
        self.topology_history: List[TopologyChange] = []
        self.clusters: Dict[str, Set[str]] = {}
        self.cluster_history: List[Dict[str, Any]] = []
        
        # Update management
        self.last_topology_update: float = 0.0
        self.update_lock = threading.Lock()
        self.auto_update_enabled: bool = False
        self.update_thread: Optional[threading.Thread] = None
        
        # Performance metrics
        self.topology_stats = {
            'total_changes': 0,
            'successful_changes': 0,
            'failed_changes': 0,
            'nodes_added': 0,
            'nodes_removed': 0,
            'connections_added': 0,
            'connections_removed': 0,
            'clusters_formed': 0,
            'clusters_dissolved': 0
        }
    
    def add_reservoir_node(
        self,
        node_id: str,
        reservoir: AttentionReservoir,
        initial_connections: Optional[List[str]] = None,
        node_properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a new reservoir node to the mesh topology."""
        
        if len(self.reservoir_nodes) >= self.max_nodes:
            return False
        
        if node_id in self.reservoir_nodes:
            return False
        
        with self.update_lock:
            # Add to topology
            self.topology_graph.add_node(node_id)
            self.reservoir_nodes[node_id] = reservoir
            self.node_metadata[node_id] = node_properties or {}
            
            # Create initial connections
            if initial_connections:
                for target_id in initial_connections:
                    if target_id in self.reservoir_nodes:
                        self._add_connection(node_id, target_id, 1.0)
            
            # Record topology change
            change = TopologyChange(
                change_id=str(uuid.uuid4()),
                change_type=TopologyChangeType.ADD_NODE,
                source_node=node_id,
                metadata={'initial_connections': initial_connections or []}
            )
            self.topology_history.append(change)
            self.topology_stats['nodes_added'] += 1
            self.topology_stats['total_changes'] += 1
            self.topology_stats['successful_changes'] += 1
        
        return True
    
    def remove_reservoir_node(self, node_id: str) -> bool:
        """Remove a reservoir node from the mesh topology."""
        
        if node_id not in self.reservoir_nodes:
            return False
        
        with self.update_lock:
            # Remove all connections
            connected_nodes = list(self.topology_graph.predecessors(node_id)) + \
                            list(self.topology_graph.successors(node_id))
            
            for connected_node in connected_nodes:
                self._remove_connection(node_id, connected_node)
                self._remove_connection(connected_node, node_id)
            
            # Remove from topology
            self.topology_graph.remove_node(node_id)
            del self.reservoir_nodes[node_id]
            if node_id in self.node_metadata:
                del self.node_metadata[node_id]
            
            # Remove from clusters
            for cluster_id, cluster_nodes in self.clusters.items():
                if node_id in cluster_nodes:
                    cluster_nodes.remove(node_id)
            
            # Clean up empty clusters
            empty_clusters = [cid for cid, nodes in self.clusters.items() if len(nodes) == 0]
            for cid in empty_clusters:
                del self.clusters[cid]
            
            # Record topology change
            change = TopologyChange(
                change_id=str(uuid.uuid4()),
                change_type=TopologyChangeType.REMOVE_NODE,
                source_node=node_id,
                metadata={'connected_nodes': connected_nodes}
            )
            self.topology_history.append(change)
            self.topology_stats['nodes_removed'] += 1
            self.topology_stats['total_changes'] += 1
            self.topology_stats['successful_changes'] += 1
        
        return True
    
    def _add_connection(self, source_id: str, target_id: str, weight: float) -> bool:
        """Add a connection between two nodes."""
        if source_id not in self.reservoir_nodes or target_id not in self.reservoir_nodes:
            return False
        
        # Add to topology graph
        self.topology_graph.add_edge(source_id, target_id, weight=weight)
        
        # Register with attention flow system
        self.attention_flow.register_flow_connection(source_id, target_id)
        
        # Record change
        change = TopologyChange(
            change_id=str(uuid.uuid4()),
            change_type=TopologyChangeType.ADD_CONNECTION,
            source_node=source_id,
            target_node=target_id,
            metadata={'weight': weight}
        )
        self.topology_history.append(change)
        self.topology_stats['connections_added'] += 1
        self.topology_stats['total_changes'] += 1
        
        return True
    
    def _remove_connection(self, source_id: str, target_id: str) -> bool:
        """Remove a connection between two nodes."""
        if not self.topology_graph.has_edge(source_id, target_id):
            return False
        
        # Remove from topology graph
        self.topology_graph.remove_edge(source_id, target_id)
        
        # Record change
        change = TopologyChange(
            change_id=str(uuid.uuid4()),
            change_type=TopologyChangeType.REMOVE_CONNECTION,
            source_node=source_id,
            target_node=target_id
        )
        self.topology_history.append(change)
        self.topology_stats['connections_removed'] += 1
        self.topology_stats['total_changes'] += 1
        
        return True
    
    def update_topology_based_on_attention(self):
        """Update mesh topology based on current attention patterns."""
        
        current_time = time.time()
        
        with self.update_lock:
            changes_made = []
            
            # Analyze attention patterns
            attention_matrix = self._compute_attention_matrix()
            
            # Add connections for high attention pairs
            for source_id in self.reservoir_nodes:
                for target_id in self.reservoir_nodes:
                    if source_id == target_id:
                        continue
                    
                    attention_strength = attention_matrix.get((source_id, target_id), 0.0)
                    
                    # Add connection if attention is high and no connection exists
                    if (attention_strength > self.connection_threshold and
                        not self.topology_graph.has_edge(source_id, target_id)):
                        
                        if self._add_connection(source_id, target_id, attention_strength):
                            changes_made.append(f"Added connection {source_id} -> {target_id}")
                    
                    # Remove connection if attention is low
                    elif (attention_strength < self.removal_threshold and
                          self.topology_graph.has_edge(source_id, target_id)):
                        
                        if self._remove_connection(source_id, target_id):
                            changes_made.append(f"Removed connection {source_id} -> {target_id}")
                    
                    # Update weight for existing connections
                    elif self.topology_graph.has_edge(source_id, target_id):
                        self.topology_graph[source_id][target_id]['weight'] = attention_strength
            
            # Update clusters based on attention
            self._update_clusters_based_on_attention(attention_matrix)
            
            # Update statistics
            if changes_made:
                self.topology_stats['successful_changes'] += len(changes_made)
            
            self.last_topology_update = current_time
        
        return changes_made
    
    def _compute_attention_matrix(self) -> Dict[Tuple[str, str], float]:
        """Compute attention relationships between all node pairs."""
        attention_matrix = {}
        
        for source_id in self.reservoir_nodes:
            source_av = self.attention_system.get_attention_value(source_id)
            
            for target_id in self.reservoir_nodes:
                if source_id == target_id:
                    continue
                
                target_av = self.attention_system.get_attention_value(target_id)
                
                # Calculate attention relationship strength
                # Consider both individual attention values and historical flow
                base_strength = np.sqrt(source_av.total_importance * target_av.total_importance)
                
                # Factor in historical attention flow
                flow_bonus = 0.0
                for flow_event in self.attention_flow.flow_history[-10:]:  # Recent flows
                    if (flow_event.get('source_atom') == source_id and 
                        target_id in flow_event.get('flow_results', {})):
                        flow_bonus += flow_event['flow_results'][target_id] * 0.1
                
                total_strength = base_strength + flow_bonus
                attention_matrix[(source_id, target_id)] = total_strength
        
        return attention_matrix
    
    def _update_clusters_based_on_attention(self, attention_matrix: Dict[Tuple[str, str], float]):
        """Update node clusters based on attention patterns."""
        
        # Find strongly connected attention communities
        strongly_connected_pairs = []
        
        for (source_id, target_id), strength in attention_matrix.items():
            if strength > self.clustering_threshold:
                reverse_strength = attention_matrix.get((target_id, source_id), 0.0)
                if reverse_strength > self.clustering_threshold:
                    strongly_connected_pairs.append((source_id, target_id, strength))
        
        # Form clusters from strongly connected components
        cluster_graph = nx.Graph()
        for source_id, target_id, _ in strongly_connected_pairs:
            cluster_graph.add_edge(source_id, target_id)
        
        # Find connected components as clusters
        new_clusters = {}
        for i, component in enumerate(nx.connected_components(cluster_graph)):
            if len(component) >= 2:  # Minimum cluster size
                cluster_id = f"cluster_{int(time.time())}_{i}"
                new_clusters[cluster_id] = set(component)
        
        # Compare with existing clusters and update
        old_cluster_nodes = set()
        for cluster_nodes in self.clusters.values():
            old_cluster_nodes.update(cluster_nodes)
        
        new_cluster_nodes = set()
        for cluster_nodes in new_clusters.values():
            new_cluster_nodes.update(cluster_nodes)
        
        # Record cluster changes
        if new_cluster_nodes != old_cluster_nodes:
            cluster_event = {
                'timestamp': time.time(),
                'old_clusters': {cid: list(nodes) for cid, nodes in self.clusters.items()},
                'new_clusters': {cid: list(nodes) for cid, nodes in new_clusters.items()},
                'formed_clusters': len(new_clusters) - len(self.clusters),
                'attention_threshold': self.clustering_threshold
            }
            self.cluster_history.append(cluster_event)
            
            # Update statistics
            if len(new_clusters) > len(self.clusters):
                self.topology_stats['clusters_formed'] += len(new_clusters) - len(self.clusters)
            elif len(new_clusters) < len(self.clusters):
                self.topology_stats['clusters_dissolved'] += len(self.clusters) - len(new_clusters)
        
        # Update clusters
        self.clusters = new_clusters
    
    def start_auto_update(self):
        """Start automatic topology updates based on attention patterns."""
        if self.auto_update_enabled:
            return
        
        self.auto_update_enabled = True
        self.update_thread = threading.Thread(target=self._auto_update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_auto_update(self):
        """Stop automatic topology updates."""
        self.auto_update_enabled = False
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
    
    def _auto_update_loop(self):
        """Automatic topology update loop."""
        while self.auto_update_enabled:
            try:
                current_time = time.time()
                
                if current_time - self.last_topology_update >= self.topology_update_interval:
                    changes = self.update_topology_based_on_attention()
                    if changes:
                        print(f"Topology updated: {len(changes)} changes made")
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                print(f"Auto-update error: {e}")
    
    def get_topology_metrics(self) -> Dict[str, Any]:
        """Get comprehensive topology metrics and statistics."""
        
        # Basic network metrics
        num_nodes = len(self.reservoir_nodes)
        num_edges = self.topology_graph.number_of_edges()
        
        # Calculate network properties
        if num_nodes > 0:
            density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
            
            # Average clustering coefficient
            clustering_coeffs = []
            for node_id in self.reservoir_nodes:
                neighbors = list(self.topology_graph.neighbors(node_id))
                if len(neighbors) >= 2:
                    subgraph = self.topology_graph.subgraph(neighbors)
                    possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                    actual_edges = subgraph.number_of_edges()
                    clustering_coeffs.append(actual_edges / possible_edges if possible_edges > 0 else 0.0)
            
            avg_clustering = np.mean(clustering_coeffs) if clustering_coeffs else 0.0
            
            # Path length (for connected components)
            try:
                if nx.is_strongly_connected(self.topology_graph):
                    avg_path_length = nx.average_shortest_path_length(self.topology_graph)
                else:
                    # Calculate for largest strongly connected component
                    largest_scc = max(nx.strongly_connected_components(self.topology_graph), 
                                    key=len, default=set())
                    if len(largest_scc) > 1:
                        scc_subgraph = self.topology_graph.subgraph(largest_scc)
                        avg_path_length = nx.average_shortest_path_length(scc_subgraph)
                    else:
                        avg_path_length = float('inf')
            except:
                avg_path_length = float('inf')
        else:
            density = 0.0
            avg_clustering = 0.0
            avg_path_length = float('inf')
        
        # Attention distribution across nodes
        attention_values = []
        for node_id in self.reservoir_nodes:
            av = self.attention_system.get_attention_value(node_id)
            attention_values.append(av.total_importance)
        
        return {
            'network_structure': {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'density': density,
                'average_clustering': avg_clustering,
                'average_path_length': avg_path_length if avg_path_length != float('inf') else None
            },
            'clusters': {
                'num_clusters': len(self.clusters),
                'cluster_sizes': [len(nodes) for nodes in self.clusters.values()],
                'largest_cluster_size': max([len(nodes) for nodes in self.clusters.values()], default=0),
                'clustered_nodes': sum(len(nodes) for nodes in self.clusters.values()),
                'clustering_threshold': self.clustering_threshold
            },
            'attention_distribution': {
                'min_attention': min(attention_values) if attention_values else 0.0,
                'max_attention': max(attention_values) if attention_values else 0.0,
                'mean_attention': np.mean(attention_values) if attention_values else 0.0,
                'std_attention': np.std(attention_values) if attention_values else 0.0
            },
            'topology_stats': self.topology_stats.copy(),
            'recent_changes': self.topology_history[-10:] if self.topology_history else [],
            'recent_cluster_events': self.cluster_history[-5:] if self.cluster_history else [],
            'update_status': {
                'auto_update_enabled': self.auto_update_enabled,
                'last_update': self.last_topology_update,
                'update_interval': self.topology_update_interval
            }
        }
    
    def visualize_topology(self, include_attention: bool = True) -> Dict[str, Any]:
        """Generate topology visualization data."""
        
        # Node data
        nodes = []
        for node_id in self.reservoir_nodes:
            av = self.attention_system.get_attention_value(node_id)
            
            node_data = {
                'id': node_id,
                'label': node_id,
                'size': max(5, av.total_importance * 2) if include_attention else 10,
                'color': self._get_node_color(node_id, av.total_importance) if include_attention else 'blue'
            }
            nodes.append(node_data)
        
        # Edge data
        edges = []
        for source_id, target_id, edge_data in self.topology_graph.edges(data=True):
            weight = edge_data.get('weight', 1.0)
            
            edge_data = {
                'source': source_id,
                'target': target_id,
                'weight': weight,
                'width': max(1, weight * 2) if include_attention else 1,
                'color': self._get_edge_color(weight) if include_attention else 'gray'
            }
            edges.append(edge_data)
        
        # Cluster data
        cluster_data = []
        for cluster_id, cluster_nodes in self.clusters.items():
            cluster_data.append({
                'id': cluster_id,
                'nodes': list(cluster_nodes),
                'size': len(cluster_nodes)
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'clusters': cluster_data,
            'metadata': {
                'include_attention': include_attention,
                'timestamp': time.time(),
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'total_clusters': len(cluster_data)
            }
        }
    
    def _get_node_color(self, node_id: str, attention_value: float) -> str:
        """Get color for a node based on attention value."""
        if attention_value > 10.0:
            return 'red'  # High attention
        elif attention_value > 5.0:
            return 'orange'  # Medium attention
        elif attention_value > 1.0:
            return 'yellow'  # Low attention
        else:
            return 'lightblue'  # Very low attention
    
    def _get_edge_color(self, weight: float) -> str:
        """Get color for an edge based on weight."""
        if weight > 5.0:
            return 'darkred'  # Strong connection
        elif weight > 2.0:
            return 'orange'  # Medium connection
        else:
            return 'lightgray'  # Weak connection


class TopologyModifier:
    """
    Advanced topology modification algorithms for attention-driven networks.
    
    Implements sophisticated topology adaptation strategies including
    genetic algorithms, reinforcement learning, and attention-driven optimization.
    
    Parameters
    ----------
    mesh_topology : MeshTopology
        Mesh topology manager
    modification_strategy : str, default='attention_driven'
        Strategy for topology modifications
    optimization_window : int, default=100
        Window size for optimization metrics
    """
    
    def __init__(
        self,
        mesh_topology: MeshTopology,
        modification_strategy: str = 'attention_driven',
        optimization_window: int = 100
    ):
        self.mesh_topology = mesh_topology
        self.modification_strategy = modification_strategy
        self.optimization_window = optimization_window
        
        # Optimization history
        self.performance_history: List[Dict[str, float]] = []
        self.modification_success_rate: float = 0.0
        
        # Strategy parameters
        self.strategy_params = {
            'attention_driven': {
                'attention_weight': 0.7,
                'performance_weight': 0.3,
                'exploration_rate': 0.1
            },
            'genetic': {
                'population_size': 20,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7
            },
            'reinforcement': {
                'learning_rate': 0.01,
                'discount_factor': 0.95,
                'epsilon': 0.1
            }
        }
    
    def optimize_topology(self, performance_metrics: Dict[str, float]) -> List[TopologyChange]:
        """
        Optimize topology based on performance metrics and attention patterns.
        
        Returns a list of recommended topology changes.
        """
        
        # Record performance
        self.performance_history.append({
            'timestamp': time.time(),
            **performance_metrics
        })
        
        # Limit history size
        if len(self.performance_history) > self.optimization_window:
            self.performance_history = self.performance_history[-self.optimization_window:]
        
        # Apply optimization strategy
        if self.modification_strategy == 'attention_driven':
            return self._attention_driven_optimization(performance_metrics)
        elif self.modification_strategy == 'genetic':
            return self._genetic_optimization(performance_metrics)
        elif self.modification_strategy == 'reinforcement':
            return self._reinforcement_optimization(performance_metrics)
        else:
            return []
    
    def _attention_driven_optimization(self, performance_metrics: Dict[str, float]) -> List[TopologyChange]:
        """Attention-driven topology optimization."""
        
        changes = []
        
        # Analyze attention imbalances
        attention_system = self.mesh_topology.attention_system
        
        # Find nodes with attention imbalances
        attention_values = {}
        for node_id in self.mesh_topology.reservoir_nodes:
            av = attention_system.get_attention_value(node_id)
            attention_values[node_id] = av.total_importance
        
        if not attention_values:
            return changes
        
        mean_attention = np.mean(list(attention_values.values()))
        std_attention = np.std(list(attention_values.values()))
        
        # Identify underperforming or overloaded nodes
        for node_id, attention in attention_values.items():
            # Very high attention nodes might benefit from load distribution
            if attention > mean_attention + 2 * std_attention:
                # Suggest adding connections to distribute load
                potential_targets = [
                    nid for nid, att in attention_values.items()
                    if nid != node_id and att < mean_attention
                ]
                
                if potential_targets:
                    target_id = min(potential_targets, key=lambda x: attention_values[x])
                    
                    change = TopologyChange(
                        change_id=str(uuid.uuid4()),
                        change_type=TopologyChangeType.ADD_CONNECTION,
                        source_node=node_id,
                        target_node=target_id,
                        attention_driver=attention,
                        metadata={'reason': 'load_distribution', 'expected_benefit': 'attention_balancing'}
                    )
                    changes.append(change)
            
            # Very low attention nodes might need better connectivity
            elif attention < mean_attention - std_attention:
                # Suggest connecting to high-attention nodes
                potential_sources = [
                    nid for nid, att in attention_values.items()
                    if nid != node_id and att > mean_attention
                ]
                
                if potential_sources:
                    source_id = max(potential_sources, key=lambda x: attention_values[x])
                    
                    change = TopologyChange(
                        change_id=str(uuid.uuid4()),
                        change_type=TopologyChangeType.ADD_CONNECTION,
                        source_node=source_id,
                        target_node=node_id,
                        attention_driver=attention_values[source_id],
                        metadata={'reason': 'attention_boost', 'expected_benefit': 'improved_connectivity'}
                    )
                    changes.append(change)
        
        return changes[:5]  # Limit to 5 changes per optimization cycle
    
    def _genetic_optimization(self, performance_metrics: Dict[str, float]) -> List[TopologyChange]:
        """Genetic algorithm-based topology optimization."""
        # Placeholder for genetic algorithm implementation
        # This would involve creating a population of topology configurations,
        # evaluating their fitness, and evolving better configurations
        return []
    
    def _reinforcement_optimization(self, performance_metrics: Dict[str, float]) -> List[TopologyChange]:
        """Reinforcement learning-based topology optimization."""
        # Placeholder for reinforcement learning implementation
        # This would involve learning optimal topology modifications based on
        # reward signals from performance metrics
        return []
    
    def evaluate_modification_impact(
        self,
        changes: List[TopologyChange],
        evaluation_period: float = 10.0
    ) -> Dict[str, Any]:
        """
        Evaluate the impact of topology modifications on system performance.
        
        Returns analysis of how the changes affected attention flow and performance.
        """
        
        # Record pre-modification state
        pre_state = {
            'timestamp': time.time(),
            'topology_metrics': self.mesh_topology.get_topology_metrics(),
            'attention_stats': self.mesh_topology.attention_system.get_attention_statistics()
        }
        
        # Apply changes (this would be done by the caller)
        # Here we just record that evaluation is starting
        
        evaluation_result = {
            'evaluation_id': str(uuid.uuid4()),
            'changes_evaluated': len(changes),
            'pre_modification_state': pre_state,
            'evaluation_period': evaluation_period,
            'start_time': time.time()
        }
        
        return evaluation_result
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get topology optimization statistics and performance trends."""
        
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        # Analyze performance trends
        recent_performance = self.performance_history[-10:]
        
        performance_trends = {}
        if len(recent_performance) > 1:
            for metric in recent_performance[0].keys():
                if metric != 'timestamp':
                    values = [p.get(metric, 0.0) for p in recent_performance]
                    performance_trends[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'trend': 'improving' if values[-1] > values[0] else 'declining',
                        'change_rate': (values[-1] - values[0]) / len(values) if len(values) > 1 else 0.0
                    }
        
        return {
            'optimization_strategy': self.modification_strategy,
            'performance_history_length': len(self.performance_history),
            'performance_trends': performance_trends,
            'modification_success_rate': self.modification_success_rate,
            'strategy_parameters': self.strategy_params.get(self.modification_strategy, {}),
            'optimization_window': self.optimization_window
        }