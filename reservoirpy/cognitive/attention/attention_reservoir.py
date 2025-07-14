"""
Attention-Aware Reservoir Computing Nodes
=========================================

ReservoirPy nodes enhanced with ECAN attention mechanisms for dynamic
adaptation of spectral radius, learning rates, and connection weights.
"""

import numpy as np
from typing import Optional, Dict, List, Any, Union, Callable
import time

# Import ReservoirPy components
from ...nodes import Reservoir, Ridge
from ...node import Node
from .. import AtomSpace, HypergraphNode
from .ecan import ECANAttentionSystem, AttentionValue


class AttentionReservoir(Reservoir):
    """
    Attention-aware reservoir node with ECAN integration.
    
    Extends ReservoirPy Reservoir with attention-driven dynamics including:
    - Attention-modulated spectral radius
    - Attention-based learning rate adaptation  
    - Connection weight scaling based on attention flow
    - Dynamic reservoir pruning based on attention values
    
    Parameters
    ----------
    units : int
        Number of reservoir units
    attention_system : ECANAttentionSystem
        ECAN attention allocation system
    atom_id : str, optional
        Unique identifier for this reservoir's hypergraph atom
    base_sr : float, default=0.9
        Base spectral radius before attention modulation
    attention_modulation_factor : float, default=0.3
        How much attention affects spectral radius (0-1)
    attention_learning_factor : float, default=0.2
        How much attention affects learning rate (0-1)
    pruning_threshold : float, default=0.1
        Attention threshold below which connections are pruned
    **kwargs
        Additional arguments passed to Reservoir
    """
    
    def __init__(
        self,
        units: int,
        attention_system: ECANAttentionSystem,
        atom_id: Optional[str] = None,
        base_sr: float = 0.9,
        attention_modulation_factor: float = 0.3,
        attention_learning_factor: float = 0.2,
        pruning_threshold: float = 0.1,
        **kwargs
    ):
        # Initialize base reservoir
        super().__init__(units=units, sr=base_sr, **kwargs)
        
        # Attention integration
        self.attention_system = attention_system
        self.atom_id = atom_id or f"reservoir_{id(self)}"
        self.base_sr = base_sr
        self.attention_modulation_factor = attention_modulation_factor
        self.attention_learning_factor = attention_learning_factor
        self.pruning_threshold = pruning_threshold
        
        # Attention state tracking
        self.attention_history: List[float] = []
        self.connection_attention: Optional[np.ndarray] = None
        self.last_attention_update: float = 0.0
        
        # Performance metrics
        self.attention_stats = {
            'spectral_radius_history': [],
            'learning_rate_history': [],
            'pruning_events': 0,
            'attention_updates': 0,
            'average_attention': 0.0
        }
        
        # Register with attention system
        self._register_with_attention_system()
    
    def _register_with_attention_system(self):
        """Register this reservoir with the attention system."""
        # Set initial attention values
        self.attention_system.set_sti(self.atom_id, 5.0)  # Initial moderate attention
        self.attention_system.set_lti(self.atom_id, 2.0)  # Some long-term importance
        
        # Initialize connection attention weights
        if hasattr(self, 'W') and self.W is not None:
            self.connection_attention = np.ones_like(self.W)
    
    def _update_attention_dynamics(self):
        """Update reservoir dynamics based on current attention values."""
        current_time = time.time()
        
        # Get current attention value
        av = self.attention_system.get_attention_value(self.atom_id)
        current_attention = av.total_importance
        
        # Record attention history
        self.attention_history.append(current_attention)
        if len(self.attention_history) > 1000:
            self.attention_history = self.attention_history[-500:]
        
        # Update spectral radius based on attention
        attention_factor = np.tanh(current_attention / 10.0)  # Normalize attention
        sr_adjustment = self.attention_modulation_factor * attention_factor
        new_sr = self.base_sr + sr_adjustment
        new_sr = np.clip(new_sr, 0.1, 1.5)  # Keep within reasonable bounds
        
        if hasattr(self, 'sr'):
            self.sr = new_sr
        
        # Update learning rate if applicable
        if hasattr(self, 'lr') and self.lr is not None:
            base_lr = getattr(self, '_base_lr', self.lr)
            if not hasattr(self, '_base_lr'):
                self._base_lr = self.lr
            
            lr_adjustment = self.attention_learning_factor * attention_factor
            new_lr = base_lr * (1.0 + lr_adjustment)
            self.lr = np.clip(new_lr, 0.001, 1.0)
        
        # Update connection attention weights
        self._update_connection_attention()
        
        # Update statistics
        self.attention_stats['spectral_radius_history'].append(new_sr)
        if hasattr(self, 'lr'):
            self.attention_stats['learning_rate_history'].append(self.lr)
        self.attention_stats['attention_updates'] += 1
        self.attention_stats['average_attention'] = np.mean(self.attention_history)
        
        self.last_attention_update = current_time
    
    def _update_connection_attention(self):
        """Update connection weights based on attention flow."""
        if not hasattr(self, 'W') or self.W is None:
            return
        
        av = self.attention_system.get_attention_value(self.atom_id)
        
        # Simple attention-based weight modulation
        # In a full implementation, this would consider attention flow between connected atoms
        attention_factor = np.tanh(av.total_importance / 10.0)
        
        if self.connection_attention is None:
            self.connection_attention = np.ones_like(self.W)
        
        # Gradually adjust connection attention
        target_attention = 0.5 + 0.5 * attention_factor
        self.connection_attention = (
            0.9 * self.connection_attention + 
            0.1 * target_attention
        )
        
        # Apply attention-based pruning
        self._apply_attention_pruning()
    
    def _apply_attention_pruning(self):
        """Prune connections with low attention values."""
        if self.connection_attention is None:
            return
        
        # Identify connections to prune
        prune_mask = self.connection_attention < self.pruning_threshold
        
        if np.any(prune_mask):
            # Apply pruning by zeroing out low-attention connections
            self.W = self.W * (1 - prune_mask.astype(float))
            self.connection_attention[prune_mask] = 0.0
            
            # Update statistics
            self.attention_stats['pruning_events'] += 1
    
    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward pass with attention-driven dynamics.
        
        Updates attention-based parameters before processing input.
        """
        # Update attention dynamics periodically
        current_time = time.time()
        if current_time - self.last_attention_update > 1.0:  # Update every second
            self._update_attention_dynamics()
        
        # Apply attention modulation to reservoir weights
        if hasattr(self, 'W') and self.W is not None and self.connection_attention is not None:
            # Temporarily modulate weights based on attention
            original_W = self.W.copy()
            self.W = self.W * self.connection_attention
            
            # Perform forward pass
            result = super().forward(x, **kwargs)
            
            # Restore original weights
            self.W = original_W
            
            return result
        else:
            return super().forward(x, **kwargs)
    
    def stimulate_attention(self, stimulus_strength: float):
        """Apply external attention stimulus to this reservoir."""
        self.attention_system.stimulate_atom(self.atom_id, stimulus_strength)
        
        # Trigger immediate attention update
        self._update_attention_dynamics()
    
    def get_attention_status(self) -> Dict[str, Any]:
        """Get current attention status and statistics."""
        av = self.attention_system.get_attention_value(self.atom_id)
        
        current_sr = getattr(self, 'sr', None)
        current_lr = getattr(self, 'lr', None)
        
        return {
            'atom_id': self.atom_id,
            'attention_value': {
                'sti': av.sti,
                'lti': av.lti,
                'total_importance': av.total_importance,
                'attention_budget': av.attention_budget
            },
            'reservoir_params': {
                'spectral_radius': current_sr,
                'learning_rate': current_lr,
                'base_sr': self.base_sr
            },
            'attention_stats': self.attention_stats.copy(),
            'connection_stats': {
                'total_connections': np.size(self.W) if hasattr(self, 'W') and self.W is not None else 0,
                'active_connections': np.count_nonzero(self.connection_attention) if self.connection_attention is not None else 0,
                'pruning_ratio': 1.0 - (np.count_nonzero(self.connection_attention) / np.size(self.connection_attention)) if self.connection_attention is not None else 0.0
            }
        }


class AttentionRidge(Ridge):
    """
    Attention-aware readout layer with ECAN integration.
    
    Extends ReservoirPy Ridge with attention-driven learning rate adaptation
    and output weight scaling based on attention flow.
    
    Parameters
    ----------
    output_dim : int
        Number of output dimensions
    attention_system : ECANAttentionSystem
        ECAN attention allocation system
    atom_id : str, optional
        Unique identifier for this readout's hypergraph atom
    attention_learning_factor : float, default=0.2
        How much attention affects learning rate
    **kwargs
        Additional arguments passed to Ridge
    """
    
    def __init__(
        self,
        output_dim: int,
        attention_system: ECANAttentionSystem,
        atom_id: Optional[str] = None,
        attention_learning_factor: float = 0.2,
        **kwargs
    ):
        super().__init__(output_dim=output_dim, **kwargs)
        
        self.attention_system = attention_system
        self.atom_id = atom_id or f"readout_{id(self)}"
        self.attention_learning_factor = attention_learning_factor
        
        # Attention state
        self.attention_history: List[float] = []
        self.last_attention_update: float = 0.0
        
        # Statistics
        self.attention_stats = {
            'learning_rate_history': [],
            'attention_updates': 0,
            'average_attention': 0.0
        }
        
        # Register with attention system
        self._register_with_attention_system()
    
    def _register_with_attention_system(self):
        """Register this readout with the attention system."""
        self.attention_system.set_sti(self.atom_id, 3.0)  # Initial attention for readout
        self.attention_system.set_lti(self.atom_id, 5.0)  # Higher long-term importance for output layer
    
    def _update_attention_dynamics(self):
        """Update readout dynamics based on attention values."""
        current_time = time.time()
        
        # Get attention value
        av = self.attention_system.get_attention_value(self.atom_id)
        current_attention = av.total_importance
        
        # Record attention
        self.attention_history.append(current_attention)
        if len(self.attention_history) > 1000:
            self.attention_history = self.attention_history[-500:]
        
        # Update learning rate based on attention
        if hasattr(self, 'ridge') and self.ridge is not None:
            base_ridge = getattr(self, '_base_ridge', self.ridge)
            if not hasattr(self, '_base_ridge'):
                self._base_ridge = self.ridge
            
            attention_factor = np.tanh(current_attention / 10.0)
            ridge_adjustment = self.attention_learning_factor * attention_factor
            new_ridge = base_ridge * (1.0 - ridge_adjustment)  # Lower ridge = higher learning
            self.ridge = np.clip(new_ridge, 1e-8, 1.0)
        
        # Update statistics
        if hasattr(self, 'ridge'):
            self.attention_stats['learning_rate_history'].append(1.0 / self.ridge)
        self.attention_stats['attention_updates'] += 1
        self.attention_stats['average_attention'] = np.mean(self.attention_history)
        
        self.last_attention_update = current_time
    
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """Fit with attention-modulated learning rate."""
        # Update attention dynamics
        self._update_attention_dynamics()
        
        return super().fit(x, y, **kwargs)
    
    def get_attention_status(self) -> Dict[str, Any]:
        """Get current attention status for readout."""
        av = self.attention_system.get_attention_value(self.atom_id)
        
        return {
            'atom_id': self.atom_id,
            'attention_value': {
                'sti': av.sti,
                'lti': av.lti,
                'total_importance': av.total_importance
            },
            'readout_params': {
                'ridge_parameter': getattr(self, 'ridge', None),
                'effective_learning_rate': 1.0 / self.ridge if hasattr(self, 'ridge') and self.ridge else None
            },
            'attention_stats': self.attention_stats.copy()
        }


class AttentionFlow:
    """
    Attention cascade and propagation algorithms for reservoir networks.
    
    Manages attention flow between connected reservoir nodes and implements
    attention-driven activation spreading through reservoir networks.
    
    Parameters
    ----------
    attention_system : ECANAttentionSystem
        ECAN attention allocation system
    flow_decay : float, default=0.8
        Decay factor for attention flow
    cascade_threshold : float, default=2.0
        Threshold for triggering attention cascades
    max_cascade_depth : int, default=5
        Maximum depth for attention cascades
    """
    
    def __init__(
        self,
        attention_system: ECANAttentionSystem,
        flow_decay: float = 0.8,
        cascade_threshold: float = 2.0,
        max_cascade_depth: int = 5
    ):
        self.attention_system = attention_system
        self.flow_decay = flow_decay
        self.cascade_threshold = cascade_threshold
        self.max_cascade_depth = max_cascade_depth
        
        # Flow tracking
        self.flow_network: Dict[str, List[str]] = {}
        self.flow_history: List[Dict[str, Any]] = []
        self.cascade_events: List[Dict[str, Any]] = []
    
    def register_flow_connection(self, source_atom: str, target_atom: str):
        """Register a flow connection between atoms."""
        if source_atom not in self.flow_network:
            self.flow_network[source_atom] = []
        
        if target_atom not in self.flow_network[source_atom]:
            self.flow_network[source_atom].append(target_atom)
    
    def propagate_attention_flow(
        self,
        source_atom: str,
        flow_strength: float,
        max_depth: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Propagate attention flow through the network.
        
        Returns a dictionary mapping atom IDs to the attention flow they received.
        """
        max_depth = max_depth or self.max_cascade_depth
        
        # Track flow propagation
        flow_received = {source_atom: flow_strength}
        propagation_queue = [(source_atom, flow_strength, 0)]
        
        while propagation_queue and max_depth > 0:
            current_atom, current_flow, depth = propagation_queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get connected atoms
            connected_atoms = self.flow_network.get(current_atom, [])
            
            if not connected_atoms:
                continue
            
            # Calculate flow to each connected atom
            flow_per_connection = current_flow * self.flow_decay / len(connected_atoms)
            
            for target_atom in connected_atoms:
                # Apply flow
                if target_atom not in flow_received:
                    flow_received[target_atom] = 0.0
                
                flow_received[target_atom] += flow_per_connection
                
                # Continue propagation if above threshold
                if flow_per_connection > self.cascade_threshold * (self.flow_decay ** (depth + 1)):
                    propagation_queue.append((target_atom, flow_per_connection, depth + 1))
        
        # Apply flows to attention system
        for atom_id, flow_amount in flow_received.items():
            if atom_id != source_atom:  # Don't re-stimulate source
                av = self.attention_system.get_attention_value(atom_id)
                av.update_sti(flow_amount)
        
        # Record flow event
        flow_event = {
            'timestamp': time.time(),
            'source_atom': source_atom,
            'initial_strength': flow_strength,
            'atoms_affected': len(flow_received),
            'total_flow_distributed': sum(flow_received.values()),
            'max_depth_reached': max(depth for _, _, depth in 
                                   [(source_atom, flow_strength, 0)] + 
                                   [(a, f, 0) for a, f in flow_received.items() if a != source_atom])
        }
        self.flow_history.append(flow_event)
        
        return flow_received
    
    def trigger_attention_cascade(
        self,
        trigger_atoms: List[str],
        cascade_strength: float = 5.0
    ) -> Dict[str, Any]:
        """
        Trigger a coordinated attention cascade across multiple atoms.
        
        Returns cascade event information and propagation results.
        """
        cascade_id = f"cascade_{int(time.time())}_{id(self)}"
        cascade_start = time.time()
        
        # Initial stimulation
        total_flow_results = {}
        
        for atom_id in trigger_atoms:
            # Stimulate initial atom
            self.attention_system.stimulate_atom(atom_id, cascade_strength)
            
            # Propagate flow
            flow_results = self.propagate_attention_flow(atom_id, cascade_strength)
            
            # Merge results
            for target_atom, flow_amount in flow_results.items():
                if target_atom not in total_flow_results:
                    total_flow_results[target_atom] = 0.0
                total_flow_results[target_atom] += flow_amount
        
        # Create cascade event record
        cascade_event = {
            'cascade_id': cascade_id,
            'timestamp': cascade_start,
            'trigger_atoms': trigger_atoms,
            'cascade_strength': cascade_strength,
            'atoms_affected': len(total_flow_results),
            'total_attention_distributed': sum(total_flow_results.values()),
            'flow_results': total_flow_results,
            'duration': time.time() - cascade_start
        }
        
        self.cascade_events.append(cascade_event)
        
        # Limit history size
        if len(self.cascade_events) > 100:
            self.cascade_events = self.cascade_events[-50:]
        
        return cascade_event
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """Get attention flow network statistics."""
        total_connections = sum(len(targets) for targets in self.flow_network.values())
        
        recent_flows = self.flow_history[-10:] if self.flow_history else []
        recent_cascades = self.cascade_events[-5:] if self.cascade_events else []
        
        return {
            'network_size': len(self.flow_network),
            'total_connections': total_connections,
            'average_connections_per_node': total_connections / len(self.flow_network) if self.flow_network else 0,
            'total_flow_events': len(self.flow_history),
            'total_cascade_events': len(self.cascade_events),
            'recent_flows': recent_flows,
            'recent_cascades': recent_cascades,
            'flow_parameters': {
                'flow_decay': self.flow_decay,
                'cascade_threshold': self.cascade_threshold,
                'max_cascade_depth': self.max_cascade_depth
            }
        }