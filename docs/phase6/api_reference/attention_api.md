# ECAN Attention System API Reference

## Module: `reservoirpy.cognitive.attention`

### Overview

====================================
ECAN Attention Allocation System
====================================

This module implements Economic Cognitive Attention Networks (ECAN) for dynamic
attention allocation and resource management in ReservoirPy cognitive architectures.

Core ECAN Components
===================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ECANAttentionSystem - Main ECAN attention allocation and spreading system
   AttentionValue - Attention value container (STI/LTI)
   ResourceAllocator - Economic attention markets and resource scheduling
   AttentionReservoir - Attention-aware reservoir node implementation

Dynamic Mesh Integration
=======================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   MeshTopology - Dynamic mesh topology management
   AttentionFlow - Attention cascade and propagation algorithms
   TopologyModifier - Adaptive topology modification based on attention

Scheduling & Economics
=====================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   AttentionMarket - Economic attention market mechanisms
   ResourceScheduler - Real-time cognitive resource scheduling
   AttentionBank - Attention banking and lending operations


### Classes and Functions

#### `AttentionBank`

**Type:** Class

**Description:** 
    Attention banking system for lending and borrowing attention resources.
    
    Provides credit mechanisms for cognitive systems to borrow attention
    when needed and pay it back over time with interest.
    
    Parameters
    ----------
    initial_reserves : float, default=1000.0
        Initial attention reserves
    base_interest_rate : float, default=0.05
        Base interest rate for loans
    max_loan_ratio : float, default=0.8
        Maximum loan-to-collateral ratio
    

**Methods:**
- `check_defaults()`: Check for loan defaults and seize collateral if necessary....
- `get_banking_status()`: Get current banking status and loan portfolio....
- `make_payment()`: Make a payment on an active loan....
- `request_loan()`: Request an attention loan with atom collateral....

#### `AttentionFlow`

**Type:** Class

**Description:** 
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
    

**Methods:**
- `get_flow_statistics()`: Get attention flow network statistics....
- `propagate_attention_flow()`: 
        Propagate attention flow through the network.
        
        Returns a dictionary mapping...
- `register_flow_connection()`: Register a flow connection between atoms....
- `trigger_attention_cascade()`: 
        Trigger a coordinated attention cascade across multiple atoms.
        
        Returns cas...

#### `AttentionMarket`

**Type:** Class

**Description:** 
    Economic attention market for resource allocation.
    
    Implements auction-based allocation of cognitive resources using attention
    values as currency, with dynamic pricing and market mechanisms.
    
    Parameters
    ----------
    initial_resources : Dict[ResourceType, float]
        Initial resource capacities by type
    base_prices : Dict[ResourceType, float], optional
        Base prices for each resource type
    price_elasticity : float, default=0.5
        How responsive prices are to demand
    market_efficiency : float, default=0.9
        Market efficiency factor (0-1)
    

**Methods:**
- `cleanup_expired_allocations()`: Remove expired allocations and free up resources....
- `get_market_status()`: Get current market status and statistics....
- `release_allocation()`: Release an active resource allocation....
- `submit_request()`: Submit a resource request to the market....

#### `AttentionReservoir`

**Type:** Class

**Description:** 
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
    

**Methods:**
- `call()`: Call the Node forward function on a single step of data.
        Can update the state of the
       ...
- `clean_buffers()`: Clean Node's buffer arrays....
- `copy()`: Returns a copy of the Node.

        Parameters
        ----------
        name : str
            Na...
- `create_buffer()`: Create a buffer array on disk, using numpy.memmap. This can be
        used to store transient varia...

#### `AttentionValue`

**Type:** Class

**Description:** 
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
    

**Methods:**
- `earn_wage()`: Earn attention wage for processing....
- `pay_rent()`: Pay attention rent if sufficient budget....

#### `ECANAttentionSystem`

**Type:** Class

**Description:** 
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
    

**Methods:**
- `get_attention_statistics()`: Get comprehensive attention allocation statistics....
- `get_attention_value()`: Get or create attention value for an atom....
- `get_focus_atoms()`: Get list of atoms currently in attention focus....
- `reset_attention()`: Reset all attention values and economic state....
- `run_economic_cycle()`: 
        Run one cycle of the economic attention market.
        
        Includes rent collection, ...

#### `MeshTopology`

**Type:** Class

**Description:** 
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
    

**Methods:**
- `add_reservoir_node()`: Add a new reservoir node to the mesh topology....
- `get_topology_metrics()`: Get comprehensive topology metrics and statistics....
- `remove_reservoir_node()`: Remove a reservoir node from the mesh topology....
- `start_auto_update()`: Start automatic topology updates based on attention patterns....
- `stop_auto_update()`: Stop automatic topology updates....

#### `ResourceAllocator`

**Type:** Class

**Description:** 
    Main resource allocation coordinator integrating ECAN attention with economic markets.
    
    Coordinates between attention systems, resource markets, task scheduling,
    and attention banking to provide comprehensive cognitive resource management.
    
    Parameters
    ----------
    attention_system : ECANAttentionSystem
        ECAN attention allocation system
    initial_resources : Dict[ResourceType, float], optional
        Initial resource capacities
    

**Methods:**
- `allocate_resources()`: 
        Allocate resources for a cognitive task.
        
        Integrates attention-based priori...
- `get_system_status()`: Get comprehensive system status across all components....
- `run_economic_cycle()`: Run one cycle of economic attention allocation....
- `shutdown()`: Shutdown the resource allocation system....

#### `ResourceScheduler`

**Type:** Class

**Description:** 
    Real-time cognitive resource scheduler with attention-driven priorities.
    
    Manages task scheduling, deadline enforcement, and resource optimization
    for cognitive processing workloads.
    
    Parameters
    ----------
    attention_system : ECANAttentionSystem
        ECAN system for attention-based prioritization  
    market : AttentionMarket
        Economic market for resource allocation
    max_concurrent_tasks : int, default=10
        Maximum number of concurrent tasks
    scheduling_quantum : float, default=0.1
        Time quantum for round-robin scheduling
    

**Methods:**
- `get_scheduler_status()`: Get current scheduler status and performance metrics....
- `schedule_task()`: Schedule a cognitive task with attention-driven priority....
- `start_scheduler()`: Start the real-time task scheduler....
- `stop_scheduler()`: Stop the task scheduler....

#### `TopologyModifier`

**Type:** Class

**Description:** 
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
    

**Methods:**
- `evaluate_modification_impact()`: 
        Evaluate the impact of topology modifications on system performance.
        
        Retur...
- `get_optimization_statistics()`: Get topology optimization statistics and performance trends....
- `optimize_topology()`: 
        Optimize topology based on performance metrics and attention patterns.
        
        Ret...

