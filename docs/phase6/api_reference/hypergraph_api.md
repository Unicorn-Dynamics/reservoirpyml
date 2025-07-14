# Hypergraph Primitives API Reference

## Module: `reservoirpy.cognitive.hypergraph`

### Overview

Hypergraph Primitives for Cognitive Encoding
===========================================

Basic hypergraph data structures for representing ReservoirPy nodes and states
as interconnected cognitive patterns.


### Classes and Functions

#### `Any`

**Type:** Class

**Description:** Special type indicating an unconstrained type.

    - Any is compatible with every type.
    - Any assumed to have all methods.
    - All values assumed to be instances of Any.

    Note that all the above statements are true from the point of view of
    static type checkers. At runtime, Any should not be used with instance
    checks.
    

#### `AtomSpace`

**Type:** Class

**Description:** 
    Simple AtomSpace-like hypergraph container for cognitive patterns.
    
    The AtomSpace maintains a collection of hypergraph nodes and links,
    providing operations for querying and manipulating the cognitive
    representation.
    
    Parameters
    ----------
    name : str, optional
        Name identifier for this AtomSpace
    

**Methods:**
- `add_link()`: Add a link to the AtomSpace....
- `add_node()`: Add a node to the AtomSpace....
- `clear()`: Clear all nodes and links from the AtomSpace....
- `get_connected_nodes()`: Get all nodes connected to the given node....
- `get_links_for_node()`: Get all links that contain the specified node....

#### `HypergraphLink`

**Type:** Class

**Description:** 
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
    

**Methods:**
- `contains_node()`: Check if the link contains a specific node....
- `get_connected_nodes()`: Get all nodes connected to the given node through this link....

#### `HypergraphNode`

**Type:** Class

**Description:** 
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
    

**Methods:**
- `get_signature()`: Generate a content-based signature for the node....

