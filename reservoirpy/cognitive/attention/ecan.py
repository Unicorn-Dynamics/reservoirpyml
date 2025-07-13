"""
Economic Cognitive Attention Networks (ECAN) Implementation
===========================================================

Core ECAN attention allocation system with Short-term Importance (STI) and 
Long-term Importance (LTI) values, activation spreading, and attention dynamics.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
import uuid

from ..hypergraph import AtomSpace, HypergraphNode, HypergraphLink


@dataclass
class AttentionValue:
    """
    Attention Value (AV) container for ECAN attention allocation.
    
    Contains Short-term Importance (STI) and Long-term Importance (LTI)
    values that drive economic attention allocation and activation spreading.
    
    Parameters
    ----------
    sti : float, default=0.0
        Short-term Importance - immediate attention priority
    lti : float, default=0.0  
        Long-term Importance - persistent attention value
    vlti : float, default=0.0
        Very Long-term Importance - structural importance
    confidence : float, default=1.0
        Confidence in the attention value estimates
    rent : float, default=0.0
        Current attention rent being paid
    wage : float, default=0.0
        Wages earned for cognitive processing
    """
    sti: float = 0.0
    lti: float = 0.0
    vlti: float = 0.0
    confidence: float = 1.0
    rent: float = 0.0
    wage: float = 0.0
    
    @property
    def total_importance(self) -> float:
        """Calculate total attention importance."""
        return self.sti + self.lti + self.vlti
    
    @property
    def attention_budget(self) -> float:
        """Calculate available attention budget after rent."""
        return max(0.0, self.total_importance - self.rent)
    
    def update_sti(self, delta: float, decay_rate: float = 0.01):
        """Update STI with decay."""
        self.sti = max(0.0, self.sti + delta - self.sti * decay_rate)
    
    def update_lti(self, delta: float):
        """Update LTI (no automatic decay)."""
        self.lti = max(0.0, self.lti + delta)
    
    def pay_rent(self, amount: float) -> bool:
        """Pay attention rent if sufficient budget."""
        if self.attention_budget >= amount:
            self.rent += amount
            return True
        return False
    
    def earn_wage(self, amount: float):
        """Earn attention wage for processing."""
        self.wage += amount
        self.sti += amount * 0.5  # Half goes to STI
        self.lti += amount * 0.3  # Smaller portion to LTI


class ECANAttentionSystem:
    """
    Main ECAN attention allocation and spreading system.
    
    Implements economic attention markets, activation spreading through hypergraphs,
    and dynamic attention-based resource allocation for cognitive architectures.
    
    Parameters
    ----------
    atomspace : AtomSpace
        Hypergraph atomspace for attention spreading
    max_attention_focus : int, default=20
        Maximum number of atoms in attention focus
    min_sti_threshold : float, default=1.0
        Minimum STI to remain in focus
    spreading_factor : float, default=0.7
        Attention spreading decay factor
    economic_cycles : int, default=10
        Cycles between economic attention market operations
    """
    
    def __init__(
        self,
        atomspace: AtomSpace,
        max_attention_focus: int = 20,
        min_sti_threshold: float = 1.0,
        spreading_factor: float = 0.7,
        economic_cycles: int = 10
    ):
        self.atomspace = atomspace
        self.max_attention_focus = max_attention_focus
        self.min_sti_threshold = min_sti_threshold
        self.spreading_factor = spreading_factor
        self.economic_cycles = economic_cycles
        
        # Attention tracking
        self.attention_values: Dict[str, AttentionValue] = {}
        self.attention_focus: Set[str] = set()
        self.attention_history: List[Dict[str, float]] = []
        
        # Economic parameters
        self.attention_bank_funds: float = 1000.0
        self.base_rent_rate: float = 0.1
        self.wage_rate: float = 0.05
        self.cycle_count: int = 0
        
        # Performance metrics
        self.allocation_stats = {
            'total_cycles': 0,
            'focus_changes': 0,
            'attention_redistributions': 0,
            'economic_transactions': 0
        }
    
    def get_attention_value(self, atom_id: str) -> AttentionValue:
        """Get or create attention value for an atom."""
        if atom_id not in self.attention_values:
            self.attention_values[atom_id] = AttentionValue()
        return self.attention_values[atom_id]
    
    def set_sti(self, atom_id: str, sti: float):
        """Set Short-term Importance for an atom."""
        av = self.get_attention_value(atom_id)
        av.sti = sti
        self._update_attention_focus()
    
    def set_lti(self, atom_id: str, lti: float):
        """Set Long-term Importance for an atom."""
        av = self.get_attention_value(atom_id)
        av.lti = lti
    
    def spread_attention(self, source_atom_id: str, spreading_amount: float):
        """
        Spread attention from a source atom to connected atoms.
        
        Implements ECAN attention spreading through hypergraph connections
        with economic considerations and decay factors.
        """
        if source_atom_id not in self.attention_values:
            return
        
        source_av = self.attention_values[source_atom_id]
        if source_av.attention_budget < spreading_amount:
            spreading_amount = source_av.attention_budget
        
        # Find connected atoms
        connected_atoms = self._get_connected_atoms(source_atom_id)
        if not connected_atoms:
            return
        
        # Calculate spreading distribution
        spread_per_atom = spreading_amount / len(connected_atoms)
        decay_factor = self.spreading_factor
        
        for target_id in connected_atoms:
            target_av = self.get_attention_value(target_id)
            spread_amount = spread_per_atom * decay_factor
            
            # Transfer attention with economic accounting
            source_av.sti -= spread_amount
            target_av.update_sti(spread_amount)
            
            # Economic transaction
            self.allocation_stats['economic_transactions'] += 1
        
        self._update_attention_focus()
    
    def _get_connected_atoms(self, atom_id: str) -> List[str]:
        """Get atoms connected to the given atom through hypergraph links."""
        connected = []
        
        # Find all links involving this atom
        for link in self.atomspace.links:
            if any(node.id == atom_id for node in link.nodes):
                # Add all other nodes in the link
                for node in link.nodes:
                    if node.id != atom_id:
                        connected.append(node.id)
        
        return list(set(connected))  # Remove duplicates
    
    def _update_attention_focus(self):
        """Update the set of atoms currently in attention focus."""
        previous_focus = self.attention_focus.copy()
        
        # Sort atoms by total attention value
        attention_sorted = sorted(
            self.attention_values.items(),
            key=lambda x: x[1].total_importance,
            reverse=True
        )
        
        # Update focus based on thresholds and limits
        new_focus = set()
        for atom_id, av in attention_sorted[:self.max_attention_focus]:
            if av.total_importance >= self.min_sti_threshold:
                new_focus.add(atom_id)
        
        # Track focus changes
        if new_focus != previous_focus:
            self.allocation_stats['focus_changes'] += 1
        
        self.attention_focus = new_focus
    
    def run_economic_cycle(self):
        """
        Run one cycle of the economic attention market.
        
        Includes rent collection, wage distribution, and attention redistribution
        based on economic principles and cognitive resource demands.
        """
        self.cycle_count += 1
        
        # Collect rent from attention focus
        total_rent_collected = 0.0
        for atom_id in self.attention_focus:
            av = self.attention_values[atom_id]
            rent_due = av.total_importance * self.base_rent_rate
            if av.pay_rent(rent_due):
                total_rent_collected += rent_due
        
        # Distribute wages based on processing activity
        total_wages_distributed = 0.0
        for atom_id, av in self.attention_values.items():
            # Simplified wage calculation based on attention activity
            wage_amount = av.total_importance * self.wage_rate
            if self.attention_bank_funds >= wage_amount:
                av.earn_wage(wage_amount)
                total_wages_distributed += wage_amount
                self.attention_bank_funds -= wage_amount
        
        # Update bank funds
        self.attention_bank_funds += total_rent_collected
        
        # Record allocation statistics
        self.allocation_stats['total_cycles'] += 1
        self.allocation_stats['economic_transactions'] += len(self.attention_focus)
        
        # Store attention history
        current_attention = {
            atom_id: av.total_importance 
            for atom_id, av in self.attention_values.items()
        }
        self.attention_history.append(current_attention)
        
        # Trigger attention redistribution if needed
        if self.cycle_count % self.economic_cycles == 0:
            self._redistribute_attention()
    
    def _redistribute_attention(self):
        """Redistribute attention based on economic and cognitive factors."""
        self.allocation_stats['attention_redistributions'] += 1
        
        # Calculate total available attention
        total_attention = sum(av.total_importance for av in self.attention_values.values())
        
        if total_attention <= 0:
            return
        
        # Redistribute based on recent activity and economic factors
        for atom_id, av in self.attention_values.items():
            # Boost attention for atoms that paid rent successfully
            if av.rent > 0:
                boost = min(av.rent * 0.1, av.attention_budget * 0.05)
                av.update_sti(boost)
            
            # Reset rent and wage counters
            av.rent = 0.0
            av.wage = 0.0
        
        self._update_attention_focus()
    
    def stimulate_atom(self, atom_id: str, stimulus_strength: float):
        """Apply external stimulus to increase an atom's attention."""
        av = self.get_attention_value(atom_id)
        av.update_sti(stimulus_strength)
        
        # Trigger attention spreading
        if stimulus_strength > 1.0:
            spread_amount = stimulus_strength * 0.3
            self.spread_attention(atom_id, spread_amount)
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attention allocation statistics."""
        if not self.attention_values:
            return {'error': 'No attention values recorded'}
        
        sti_values = [av.sti for av in self.attention_values.values()]
        lti_values = [av.lti for av in self.attention_values.values()]
        
        return {
            'total_atoms': len(self.attention_values),
            'focus_size': len(self.attention_focus),
            'average_sti': np.mean(sti_values) if sti_values else 0.0,
            'average_lti': np.mean(lti_values) if lti_values else 0.0,
            'max_sti': max(sti_values) if sti_values else 0.0,
            'max_lti': max(lti_values) if lti_values else 0.0,
            'attention_bank_funds': self.attention_bank_funds,
            'allocation_stats': self.allocation_stats.copy(),
            'economic_cycles': self.cycle_count
        }
    
    def get_focus_atoms(self) -> List[str]:
        """Get list of atoms currently in attention focus."""
        return list(self.attention_focus)
    
    def reset_attention(self):
        """Reset all attention values and economic state."""
        self.attention_values.clear()
        self.attention_focus.clear()
        self.attention_history.clear()
        self.attention_bank_funds = 1000.0
        self.cycle_count = 0
        self.allocation_stats = {
            'total_cycles': 0,
            'focus_changes': 0,
            'attention_redistributions': 0,
            'economic_transactions': 0
        }