# Meta-Optimization API Reference

## Module: `reservoirpy.cognitive.meta_optimization`

### Overview

==================================================
Phase 5: Recursive Meta-Cognition & Evolutionary Optimization
==================================================

This module enables the system to observe, analyze, and recursively improve itself
using evolutionary algorithms, with ReservoirPy serving as both the computational
substrate and the target for optimization.

.. currentmodule:: reservoirpy.cognitive.meta_optimization

Core Components
===============

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   MetaCognitiveSystem - Main meta-cognitive analysis and feedback system
   EvolutionaryOptimizer - Evolutionary algorithms for reservoir optimization
   PerformanceBenchmarker - Continuous performance monitoring and metrics
   RecursiveImprovement - Recursive feedback loops for self-improvement

Meta-Cognitive Components
========================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ReservoirIntrospector - Reservoir performance monitoring and analysis
   MetaAttentionAllocator - Meta-cognitive attention allocation system
   SelfAnalysisModule - Feedback-driven self-analysis capabilities

Evolutionary Components
======================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   TopologyEvolver - Evolutionary optimization of reservoir topology
   ParameterEvolver - Adaptive evolution of spectral radius and learning rates
   ConnectionEvolver - Genetic programming for reservoir connection patterns
   ArchitectureMutator - Real-time reservoir architecture mutation system

Optimization Components
======================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   ContinuousBenchmarker - Real-time performance benchmarking system
   SelfTuner - Adaptive self-tuning of kernels and attention mechanisms
   MultiObjectiveOptimizer - Multi-objective optimization for cognitive trade-offs
   FitnessEvaluator - Fitness evaluation for evolutionary optimization

Recursive Improvement
====================

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   EmbodiedFeedback - Feedback mechanisms from embodied agent performance
   EnsembleEvolver - Reservoir ensemble evolution for improved robustness
   MetaLearner - Meta-learning algorithms for rapid adaptation
   HierarchicalOptimizer - Hierarchical optimization across cognitive mesh layers


### Classes and Functions

#### `ArchitectureMutator`

**Type:** Class

**Description:** 
    Real-time reservoir architecture mutation system.
    
    Provides dynamic mutation capabilities for live reservoir systems,
    enabling continuous adaptation during operation.
    

**Methods:**
- `analyze_mutation_impact()`: Analyze impact of recent mutations on performance....
- `mutate_architecture()`: Apply real-time mutations to reservoir architecture....

#### `ConnectionEvolver`

**Type:** Class

**Description:** 
    Genetic programming for reservoir connection patterns.
    
    Evolves specific connection patterns, motifs, and structural features
    that improve reservoir computing performance.
    

**Methods:**
- `evolve_connection_patterns()`: Evolve connection patterns for improved performance....

#### `ContinuousBenchmarker`

**Type:** Class

**Description:** 
    Real-time continuous benchmarking system.
    
    Provides continuous monitoring and benchmarking of system performance
    with automatic baseline updates and trend detection.
    

**Methods:**
- `get_continuous_summary()`: Get summary of continuous benchmarking results....
- `set_alert_threshold()`: Set alert threshold for a specific metric....
- `start_continuous_benchmarking()`: Start continuous benchmarking in background thread....
- `stop_continuous_benchmarking()`: Stop continuous benchmarking....

#### `EmbodiedFeedback`

**Type:** Class

**Description:** 
    Feedback mechanisms from embodied agent performance.
    
    Collects and processes feedback from physical and virtual agents
    to guide system optimization and adaptation.
    

**Methods:**
- `collect_agent_feedback()`: Collect performance feedback from embodied agent....
- `process_embodied_feedback()`: Process collected feedback to generate system improvements....
- `register_embodied_agent()`: Register embodied agent for feedback collection....

#### `EnsembleEvolver`

**Type:** Class

**Description:** 
    Reservoir ensemble evolution for improved robustness.
    
    Evolves collections of reservoir instances for better performance
    through diversity, specialization, and cooperative learning.
    

**Methods:**
- `create_ensemble()`: Create new reservoir ensemble....
- `evolve_ensemble()`: Evolve reservoir ensemble for improved performance....

#### `EvolutionaryOptimizer`

**Type:** Class

**Description:** 
    Main evolutionary optimizer coordinating all evolutionary processes.
    
    Integrates topology evolution, parameter evolution, connection patterns,
    and architecture mutations for comprehensive reservoir optimization.
    

**Methods:**
- `continuous_optimization()`: Continuous real-time optimization using mutations....
- `optimize_reservoir()`: Comprehensive evolutionary optimization of reservoir....

#### `FitnessEvaluator`

**Type:** Class

**Description:** 
    Fitness evaluation system for evolutionary optimization.
    
    Provides comprehensive fitness evaluation combining performance metrics,
    stability measures, and optimization objectives.
    

**Methods:**
- `evaluate_reservoir_fitness()`: Evaluate comprehensive fitness of reservoir configuration....
- `get_fitness_trends()`: Get fitness trends for analysis....
- `register_fitness_component()`: Register a fitness component for evaluation....

#### `HierarchicalOptimizer`

**Type:** Class

**Description:** 
    Hierarchical optimization across cognitive mesh layers.
    
    Implements multi-level optimization that coordinates improvements
    across different hierarchical levels of the cognitive system.
    

**Methods:**
- `hierarchical_optimize()`: Perform hierarchical optimization across all levels....
- `register_optimization_level()`: Register optimization level in hierarchy....

#### `MetaAttentionAllocator`

**Type:** Class

**Description:** 
    Meta-cognitive attention allocation for self-improvement.
    
    Manages attention resources for meta-cognitive processes including
    self-analysis, performance monitoring, and improvement planning.
    

**Methods:**
- `allocate_meta_attention()`: Dynamically allocate attention for meta-cognitive processes....

#### `MetaCognitiveSystem`

**Type:** Class

**Description:** 
    Main meta-cognitive system coordinating all meta-cognitive processes.
    
    Integrates reservoir introspection, attention allocation, and self-analysis
    for comprehensive recursive self-improvement capabilities.
    

**Methods:**
- `generate_feedback_loops()`: Generate feedback loops for recursive improvement....
- `update_meta_cognitive_state()`: Update and return current meta-cognitive state....

#### `MetaLearner`

**Type:** Class

**Description:** 
    Meta-learning algorithms for rapid adaptation.
    
    Implements learning-to-learn capabilities that enable the system
    to quickly adapt to new tasks and environments.
    

**Methods:**
- `learn_adaptation_strategy()`: Learn adaptation strategy for specific task type....
- `rapid_adapt()`: Rapidly adapt system to new task using meta-learning....

#### `MultiObjectiveOptimizer`

**Type:** Class

**Description:** 
    Multi-objective optimization for cognitive trade-offs.
    
    Handles optimization problems with multiple competing objectives
    such as performance vs. stability, speed vs. accuracy, etc.
    

**Methods:**
- `add_objective()`: Add optimization objective....
- `evaluate_solution()`: Evaluate a solution against all objectives....
- `optimize_multi_objective()`: Perform multi-objective optimization....
- `set_trade_off_weights()`: Set trade-off weights between objectives....

#### `ParameterEvolver`

**Type:** Class

**Description:** 
    Adaptive evolution of spectral radius and learning rates.
    
    Evolves reservoir parameters like spectral radius, leaking rate,
    and other critical parameters for optimal performance.
    

**Methods:**
- `evolve_parameters()`: Evolve reservoir parameters using evolutionary strategy....

#### `PerformanceBenchmarker`

**Type:** Class

**Description:** 
    Main performance benchmarking system.
    
    Provides comprehensive benchmarking capabilities for reservoir performance,
    cognitive metrics, and system efficiency measurements.
    

**Methods:**
- `get_performance_trends()`: Get performance trends for a specific metric....
- `register_benchmark_task()`: Register a benchmark task for regular execution....
- `run_benchmark_suite()`: Run comprehensive benchmark suite....

#### `RecursiveImprovement`

**Type:** Class

**Description:** 
    Main recursive improvement system coordinating all improvement loops.
    
    Integrates embodied feedback, ensemble evolution, meta-learning, and 
    hierarchical optimization for comprehensive recursive self-improvement.
    

**Methods:**
- `get_improvement_summary()`: Get summary of recursive improvement performance....
- `register_feedback_loop()`: Register recursive feedback loop....
- `start_recursive_improvement()`: Start recursive improvement process....

#### `ReservoirIntrospector`

**Type:** Class

**Description:** 
    Reservoir introspection mechanisms for performance monitoring.
    
    Provides deep analysis of reservoir dynamics, states, and performance
    characteristics for meta-cognitive self-improvement.
    

**Methods:**
- `analyze_reservoir_dynamics()`: Analyze reservoir dynamics and extract performance insights....
- `detect_performance_patterns()`: Detect patterns in performance history for meta-analysis....
- `monitor_cognitive_performance()`: Monitor overall cognitive mesh performance....

#### `SelfAnalysisModule`

**Type:** Class

**Description:** 
    Feedback-driven self-analysis module.
    
    Provides comprehensive self-analysis capabilities including performance
    reflection, weakness identification, and improvement opportunity discovery.
    

**Methods:**
- `perform_self_analysis()`: Perform comprehensive self-analysis....

#### `SelfTuner`

**Type:** Class

**Description:** 
    Adaptive self-tuning system for kernels and attention mechanisms.
    
    Automatically adjusts system parameters based on performance feedback
    and optimization objectives.
    

**Methods:**
- `get_tuning_summary()`: Get summary of tuning performance and trends....
- `register_tunable_parameter()`: Register a parameter for automatic tuning....
- `set_performance_target()`: Set performance target for tuning optimization....
- `tune_parameters()`: Perform adaptive parameter tuning based on current performance....

#### `TopologyEvolver`

**Type:** Class

**Description:** 
    Evolutionary optimization of reservoir topology.
    
    Evolves reservoir connection matrices, sparsity patterns, and 
    structural properties for improved performance.
    

**Methods:**
- `evolve_topology()`: Evolve reservoir topology using genetic algorithm....
- `initialize_population()`: Initialize population with diverse reservoir topologies....

