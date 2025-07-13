"""
Hypergraph Primitives for Cognitive Encoding
===========================================

Basic hypergraph data structures for representing ReservoirPy nodes and states
as interconnected cognitive patterns.
"""

import hashlib
import uuid
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np


class HypergraphNode:
    """
    Basic hypergraph node representation for cognitive primitives.
    
    A hypergraph node represents an atomic cognitive concept with associated
    properties and tensor data.
    
    Parameters
    ----------
    name : str
        Unique identifier for the node
    node_type : str
        Type classification (e.g., 'reservoir', 'readout', 'activation')
    properties : dict, optional
        Additional properties and metadata
    tensor_data : np.ndarray, optional
        Associated tensor data for the cognitive state
    """
    
    def __init__(
        self, 
        name: str, 
        node_type: str, 
        properties: Optional[Dict[str, Any]] = None,
        tensor_data: Optional[np.ndarray] = None
    ):
        self.name = name
        self.node_type = node_type
        self.properties = properties or {}
        self.tensor_data = tensor_data
        self.id = str(uuid.uuid4())
        
    def __repr__(self) -> str:
        return f"HypergraphNode(name='{self.name}', type='{self.node_type}', id='{self.id[:8]}...')"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HypergraphNode):
            return False
        return self.id == other.id
    
    def get_signature(self) -> str:
        """Generate a content-based signature for the node."""
        content = f"{self.name}:{self.node_type}:{str(sorted(self.properties.items()))}"
        if self.tensor_data is not None:
            content += f":{self.tensor_data.shape}:{np.sum(self.tensor_data)}"
        return hashlib.md5(content.encode()).hexdigest()


class HypergraphLink:
    """
    Hypergraph link connecting multiple nodes with semantic relationships.
    
    A hypergraph link can connect any number of nodes and represents
    complex relationships between cognitive primitives.
    
    Parameters
    ----------
    nodes : List[HypergraphNode]
        List of nodes connected by this link
    link_type : str
        Type of relationship (e.g., 'connection', 'feedback', 'state_flow')
    properties : dict, optional
        Additional properties and metadata
    strength : float, optional
        Connection strength or weight
    """
    
    def __init__(
        self,
        nodes: List[HypergraphNode],
        link_type: str,
        properties: Optional[Dict[str, Any]] = None,
        strength: float = 1.0
    ):
        if len(nodes) < 2:
            raise ValueError("A hypergraph link must connect at least 2 nodes")
        
        self.nodes = list(nodes)
        self.link_type = link_type
        self.properties = properties or {}
        self.strength = strength
        self.id = str(uuid.uuid4())
        
    def __repr__(self) -> str:
        node_names = [node.name for node in self.nodes]
        return f"HypergraphLink(type='{self.link_type}', nodes={node_names}, id='{self.id[:8]}...')"
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HypergraphLink):
            return False
        return self.id == other.id
    
    def contains_node(self, node: HypergraphNode) -> bool:
        """Check if the link contains a specific node."""
        return node in self.nodes
    
    def get_connected_nodes(self, node: HypergraphNode) -> List[HypergraphNode]:
        """Get all nodes connected to the given node through this link."""
        if node not in self.nodes:
            return []
        return [n for n in self.nodes if n != node]


class AtomSpace:
    """
    Simple AtomSpace-like hypergraph container for cognitive patterns.
    
    The AtomSpace maintains a collection of hypergraph nodes and links,
    providing operations for querying and manipulating the cognitive
    representation.
    
    Parameters
    ----------
    name : str, optional
        Name identifier for this AtomSpace
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: Set[HypergraphNode] = set()
        self.links: Set[HypergraphLink] = set()
        self._node_index: Dict[str, HypergraphNode] = {}
        self._type_index: Dict[str, Set[HypergraphNode]] = {}
        
    def add_node(self, node: HypergraphNode) -> None:
        """Add a node to the AtomSpace."""
        self.nodes.add(node)
        self._node_index[node.name] = node
        
        if node.node_type not in self._type_index:
            self._type_index[node.node_type] = set()
        self._type_index[node.node_type].add(node)
    
    def add_link(self, link: HypergraphLink) -> None:
        """Add a link to the AtomSpace."""
        # Ensure all linked nodes are in the space
        for node in link.nodes:
            if node not in self.nodes:
                self.add_node(node)
        
        self.links.add(link)
    
    def get_node(self, name: str) -> Optional[HypergraphNode]:
        """Get a node by name."""
        return self._node_index.get(name)
    
    def get_nodes_by_type(self, node_type: str) -> Set[HypergraphNode]:
        """Get all nodes of a specific type."""
        return self._type_index.get(node_type, set()).copy()
    
    def get_links_for_node(self, node: HypergraphNode) -> List[HypergraphLink]:
        """Get all links that contain the specified node."""
        return [link for link in self.links if link.contains_node(node)]
    
    def get_connected_nodes(self, node: HypergraphNode) -> Set[HypergraphNode]:
        """Get all nodes connected to the given node."""
        connected = set()
        for link in self.get_links_for_node(node):
            connected.update(link.get_connected_nodes(node))
        return connected
    
    def query_pattern(
        self, 
        node_types: List[str], 
        link_type: Optional[str] = None
    ) -> List[List[HypergraphNode]]:
        """
        Query for patterns matching the given node types and link type.
        
        Parameters
        ----------
        node_types : List[str]
            List of node types to match in order
        link_type : str, optional
            Type of link that should connect the nodes
            
        Returns
        -------
        List[List[HypergraphNode]]
            List of node combinations that match the pattern
        """
        if len(node_types) < 2:
            return []
            
        # Find all possible combinations
        patterns = []
        
        # Get nodes of first type
        first_type_nodes = self.get_nodes_by_type(node_types[0])
        
        for first_node in first_type_nodes:
            self._find_pattern_recursive(
                [first_node], 
                node_types[1:], 
                link_type, 
                patterns
            )
        
        return patterns
    
    def _find_pattern_recursive(
        self, 
        current_path: List[HypergraphNode],
        remaining_types: List[str],
        link_type: Optional[str],
        patterns: List[List[HypergraphNode]]
    ) -> None:
        """Recursively find pattern matches."""
        if not remaining_types:
            patterns.append(current_path.copy())
            return
            
        current_node = current_path[-1]
        next_type = remaining_types[0]
        
        # Find connected nodes of the next type
        for link in self.get_links_for_node(current_node):
            if link_type and link.link_type != link_type:
                continue
                
            for connected_node in link.get_connected_nodes(current_node):
                if (connected_node.node_type == next_type and 
                    connected_node not in current_path):
                    
                    self._find_pattern_recursive(
                        current_path + [connected_node],
                        remaining_types[1:],
                        link_type,
                        patterns
                    )
    
    def clear(self) -> None:
        """Clear all nodes and links from the AtomSpace."""
        self.nodes.clear()
        self.links.clear()
        self._node_index.clear()
        self._type_index.clear()
    
    def size(self) -> Dict[str, int]:
        """Get the size statistics of the AtomSpace."""
        return {
            "nodes": len(self.nodes),
            "links": len(self.links),
            "node_types": len(self._type_index)
        }
    
    def __repr__(self) -> str:
        stats = self.size()
        return f"AtomSpace(name='{self.name}', nodes={stats['nodes']}, links={stats['links']})"