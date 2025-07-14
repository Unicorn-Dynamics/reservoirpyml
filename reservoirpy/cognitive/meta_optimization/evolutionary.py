"""
Evolutionary Optimization for Reservoir Self-Improvement
=======================================================

This module implements evolutionary algorithms for reservoir topology optimization,
adaptive parameter evolution, and genetic programming for connection patterns.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import random
import time
from collections import deque
from copy import deepcopy

try:
    from ...nodes import Reservoir
    from ...model import Model
    from ... import mat_gen
except ImportError:
    # Fallback for minimal dependencies
    pass


@dataclass
class Individual:
    """Individual in evolutionary population."""
    genome: Dict[str, Any]
    fitness: float
    age: int
    performance_history: List[float]


@dataclass
class EvolutionaryParameters:
    """Parameters for evolutionary optimization."""
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 2.0
    elitism_ratio: float = 0.1
    max_generations: int = 100
    convergence_threshold: float = 1e-6


class TopologyEvolver:
    """
    Evolutionary optimization of reservoir topology.
    
    Evolves reservoir connection matrices, sparsity patterns, and 
    structural properties for improved performance.
    """
    
    def __init__(self, target_size: int = 100, evolution_params: Optional[EvolutionaryParameters] = None):
        self.target_size = target_size
        self.params = evolution_params or EvolutionaryParameters()
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = deque(maxlen=1000)
        
    def initialize_population(self) -> List[Individual]:
        """Initialize population with diverse reservoir topologies."""
        population = []
        
        for i in range(self.params.population_size):
            # Generate diverse connection matrices
            genome = self._generate_random_topology()
            
            individual = Individual(
                genome=genome,
                fitness=0.0,
                age=0,
                performance_history=[]
            )
            
            population.append(individual)
        
        self.population = population
        return population
    
    def evolve_topology(self, fitness_function: Callable, generations: Optional[int] = None) -> Individual:
        """Evolve reservoir topology using genetic algorithm."""
        if not self.population:
            self.initialize_population()
        
        max_gen = generations or self.params.max_generations
        
        for generation in range(max_gen):
            self.generation = generation
            
            # Evaluate fitness for all individuals
            self._evaluate_population(fitness_function)
            
            # Selection and reproduction
            new_population = self._reproduce_population()
            
            # Update population
            self.population = new_population
            
            # Track best individual
            self._update_best_individual()
            
            # Check convergence
            if self._check_convergence():
                break
        
        return self.best_individual
    
    def _generate_random_topology(self) -> Dict[str, Any]:
        """Generate random reservoir topology genome."""
        # Connection matrix parameters
        sparsity = random.uniform(0.1, 0.9)
        spectral_radius = random.uniform(0.5, 1.2)
        
        # Generate random connection matrix
        W = mat_gen.uniform(self.target_size, self.target_size, 
                           spectral_radius=spectral_radius, 
                           sparsity=sparsity)
        
        # Topology features
        genome = {
            "W": W,
            "sparsity": sparsity,
            "spectral_radius": spectral_radius,
            "small_world_coeff": random.uniform(0.0, 1.0),
            "clustering_coeff": random.uniform(0.0, 0.5),
            "hub_ratio": random.uniform(0.05, 0.3),
            "modularity": random.uniform(0.0, 0.8),
            "hierarchy_level": random.randint(1, 5)
        }
        
        return genome
    
    def _evaluate_population(self, fitness_function: Callable):
        """Evaluate fitness for all individuals in population."""
        for individual in self.population:
            try:
                fitness = fitness_function(individual.genome)
                individual.fitness = fitness
                individual.performance_history.append(fitness)
                
                # Limit history size
                if len(individual.performance_history) > 20:
                    individual.performance_history = individual.performance_history[-20:]
                    
            except Exception as e:
                # Assign low fitness for problematic individuals
                individual.fitness = 0.01
                individual.performance_history.append(0.01)
    
    def _reproduce_population(self) -> List[Individual]:
        """Create new population through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism - keep best individuals
        elite_count = int(self.params.population_size * self.params.elitism_ratio)
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_count]
        new_population.extend(deepcopy(elite))
        
        # Generate rest of population
        while len(new_population) < self.params.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.params.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            # Mutation
            if random.random() < self.params.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.params.mutation_rate:
                child2 = self._mutate(child2)
            
            # Add children
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        new_population = new_population[:self.params.population_size]
        
        # Age population
        for individual in new_population:
            individual.age += 1
        
        return new_population
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection for parent selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover operation for topology evolution."""
        child1_genome = {}
        child2_genome = {}
        
        # Blend numerical parameters
        for key in ["sparsity", "spectral_radius", "small_world_coeff", 
                   "clustering_coeff", "hub_ratio", "modularity"]:
            if key in parent1.genome and key in parent2.genome:
                alpha = random.random()
                child1_genome[key] = alpha * parent1.genome[key] + (1-alpha) * parent2.genome[key]
                child2_genome[key] = (1-alpha) * parent1.genome[key] + alpha * parent2.genome[key]
        
        # Crossover connection matrices
        if "W" in parent1.genome and "W" in parent2.genome:
            W1, W2 = self._crossover_matrices(parent1.genome["W"], parent2.genome["W"])
            child1_genome["W"] = W1
            child2_genome["W"] = W2
        
        # Handle discrete parameters
        child1_genome["hierarchy_level"] = random.choice([parent1.genome.get("hierarchy_level", 1),
                                                         parent2.genome.get("hierarchy_level", 1)])
        child2_genome["hierarchy_level"] = random.choice([parent1.genome.get("hierarchy_level", 1),
                                                         parent2.genome.get("hierarchy_level", 1)])
        
        child1 = Individual(genome=child1_genome, fitness=0.0, age=0, performance_history=[])
        child2 = Individual(genome=child2_genome, fitness=0.0, age=0, performance_history=[])
        
        return child1, child2
    
    def _crossover_matrices(self, W1: np.ndarray, W2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crossover operation for connection matrices."""
        if W1.shape != W2.shape:
            return W1.copy(), W2.copy()
        
        # Block-wise crossover
        rows, cols = W1.shape
        split_row = random.randint(1, rows-1)
        split_col = random.randint(1, cols-1)
        
        child1 = W1.copy()
        child2 = W2.copy()
        
        # Exchange blocks
        child1[split_row:, split_col:] = W2[split_row:, split_col:]
        child2[split_row:, split_col:] = W1[split_row:, split_col:]
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutation operation for topology evolution."""
        mutated_genome = deepcopy(individual.genome)
        
        # Mutate numerical parameters
        for key in ["sparsity", "spectral_radius", "small_world_coeff",
                   "clustering_coeff", "hub_ratio", "modularity"]:
            if key in mutated_genome and random.random() < 0.3:
                mutation_strength = random.uniform(-0.1, 0.1)
                mutated_genome[key] = np.clip(mutated_genome[key] + mutation_strength, 0.0, 1.0)
        
        # Mutate connection matrix
        if "W" in mutated_genome:
            mutated_genome["W"] = self._mutate_matrix(mutated_genome["W"])
        
        # Mutate discrete parameters
        if random.random() < 0.2:
            mutated_genome["hierarchy_level"] = max(1, min(5, 
                mutated_genome.get("hierarchy_level", 1) + random.randint(-1, 1)))
        
        # Regenerate connection matrix if spectral radius changed significantly
        if "spectral_radius" in mutated_genome:
            try:
                new_W = mat_gen.uniform(self.target_size, self.target_size,
                                      spectral_radius=mutated_genome["spectral_radius"],
                                      sparsity=mutated_genome.get("sparsity", 0.5))
                mutated_genome["W"] = new_W
            except:
                pass  # Keep original matrix if generation fails
        
        return Individual(genome=mutated_genome, fitness=0.0, age=0, performance_history=[])
    
    def _mutate_matrix(self, W: np.ndarray, mutation_rate: float = 0.01) -> np.ndarray:
        """Mutate connection matrix."""
        mutated_W = W.copy()
        
        # Random weight mutations
        mask = np.random.random(W.shape) < mutation_rate
        mutations = np.random.normal(0, 0.1, W.shape)
        mutated_W[mask] += mutations[mask]
        
        # Connection addition/removal
        if random.random() < 0.1:
            # Add new connections
            zero_mask = (W == 0)
            if np.any(zero_mask):
                add_indices = np.where(zero_mask)
                if len(add_indices[0]) > 0:
                    idx = random.randint(0, len(add_indices[0])-1)
                    row, col = add_indices[0][idx], add_indices[1][idx]
                    mutated_W[row, col] = random.uniform(-0.5, 0.5)
        
        if random.random() < 0.1:
            # Remove connections
            nonzero_mask = (W != 0)
            if np.any(nonzero_mask):
                remove_indices = np.where(nonzero_mask)
                if len(remove_indices[0]) > 0:
                    idx = random.randint(0, len(remove_indices[0])-1)
                    row, col = remove_indices[0][idx], remove_indices[1][idx]
                    mutated_W[row, col] = 0.0
        
        return mutated_W
    
    def _update_best_individual(self):
        """Update best individual in population."""
        current_best = max(self.population, key=lambda x: x.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = deepcopy(current_best)
        
        # Record fitness history
        self.fitness_history.append(current_best.fitness)
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self.fitness_history) < 10:
            return False
        
        recent_fitnesses = list(self.fitness_history)[-10:]
        fitness_std = np.std(recent_fitnesses)
        
        return fitness_std < self.params.convergence_threshold


class ParameterEvolver:
    """
    Adaptive evolution of spectral radius and learning rates.
    
    Evolves reservoir parameters like spectral radius, leaking rate,
    and other critical parameters for optimal performance.
    """
    
    def __init__(self, evolution_params: Optional[EvolutionaryParameters] = None):
        self.params = evolution_params or EvolutionaryParameters()
        self.parameter_ranges = {
            "spectral_radius": (0.1, 1.5),
            "leaking_rate": (0.01, 1.0),
            "input_scaling": (0.1, 2.0),
            "bias_scaling": (0.0, 1.0),
            "noise_level": (0.0, 0.1),
            "feedback_scaling": (0.0, 0.5)
        }
        self.population = []
        self.best_parameters = None
        
    def evolve_parameters(self, fitness_function: Callable, 
                         target_parameters: List[str] = None) -> Dict[str, float]:
        """Evolve reservoir parameters using evolutionary strategy."""
        if target_parameters is None:
            target_parameters = list(self.parameter_ranges.keys())
        
        # Initialize population
        self._initialize_parameter_population(target_parameters)
        
        for generation in range(self.params.max_generations):
            # Evaluate fitness
            for individual in self.population:
                individual.fitness = fitness_function(individual.genome)
            
            # Evolution step
            self._evolve_parameter_step(target_parameters)
            
            # Update best
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_parameters is None or current_best.fitness > self.best_parameters.fitness:
                self.best_parameters = deepcopy(current_best)
        
        return self.best_parameters.genome if self.best_parameters else {}
    
    def _initialize_parameter_population(self, target_parameters: List[str]):
        """Initialize population for parameter evolution."""
        population = []
        
        for _ in range(self.params.population_size):
            genome = {}
            for param in target_parameters:
                if param in self.parameter_ranges:
                    low, high = self.parameter_ranges[param]
                    genome[param] = random.uniform(low, high)
            
            individual = Individual(
                genome=genome,
                fitness=0.0,
                age=0,
                performance_history=[]
            )
            population.append(individual)
        
        self.population = population
    
    def _evolve_parameter_step(self, target_parameters: List[str]):
        """Single evolution step for parameter optimization."""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep elite
        elite_count = int(self.params.population_size * self.params.elitism_ratio)
        new_population = self.population[:elite_count]
        
        # Generate offspring
        while len(new_population) < self.params.population_size:
            # Select parents
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Create offspring
            child = self._parameter_crossover(parent1, parent2, target_parameters)
            child = self._parameter_mutation(child, target_parameters)
            
            new_population.append(child)
        
        self.population = new_population
    
    def _select_parent(self) -> Individual:
        """Select parent for parameter evolution."""
        # Tournament selection
        tournament_size = 3
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _parameter_crossover(self, parent1: Individual, parent2: Individual, 
                           target_parameters: List[str]) -> Individual:
        """Crossover for parameter evolution."""
        child_genome = {}
        
        for param in target_parameters:
            if param in parent1.genome and param in parent2.genome:
                # Blend crossover
                alpha = random.random()
                child_genome[param] = alpha * parent1.genome[param] + (1-alpha) * parent2.genome[param]
            elif param in parent1.genome:
                child_genome[param] = parent1.genome[param]
            elif param in parent2.genome:
                child_genome[param] = parent2.genome[param]
        
        return Individual(genome=child_genome, fitness=0.0, age=0, performance_history=[])
    
    def _parameter_mutation(self, individual: Individual, 
                          target_parameters: List[str]) -> Individual:
        """Mutation for parameter evolution."""
        mutated_genome = deepcopy(individual.genome)
        
        for param in target_parameters:
            if param in mutated_genome and random.random() < self.params.mutation_rate:
                if param in self.parameter_ranges:
                    low, high = self.parameter_ranges[param]
                    # Gaussian mutation
                    mutation_std = (high - low) * 0.1
                    mutation = random.gauss(0, mutation_std)
                    mutated_genome[param] = np.clip(mutated_genome[param] + mutation, low, high)
        
        return Individual(genome=mutated_genome, fitness=0.0, age=0, performance_history=[])


class ConnectionEvolver:
    """
    Genetic programming for reservoir connection patterns.
    
    Evolves specific connection patterns, motifs, and structural features
    that improve reservoir computing performance.
    """
    
    def __init__(self, reservoir_size: int = 100):
        self.reservoir_size = reservoir_size
        self.connection_patterns = []
        self.pattern_fitness = {}
        
    def evolve_connection_patterns(self, base_matrix: np.ndarray, 
                                 fitness_function: Callable,
                                 num_patterns: int = 10) -> List[Dict[str, Any]]:
        """Evolve connection patterns for improved performance."""
        patterns = []
        
        # Generate initial patterns
        for _ in range(num_patterns):
            pattern = self._generate_random_pattern()
            patterns.append(pattern)
        
        # Evolve patterns
        for generation in range(50):  # Reduced generations for patterns
            # Evaluate patterns
            for pattern in patterns:
                modified_matrix = self._apply_pattern(base_matrix, pattern)
                pattern["fitness"] = fitness_function({"W": modified_matrix})
            
            # Sort by fitness
            patterns.sort(key=lambda x: x["fitness"], reverse=True)
            
            # Keep best patterns and generate new ones
            elite_count = max(1, num_patterns // 3)
            new_patterns = patterns[:elite_count]
            
            # Generate offspring from best patterns
            while len(new_patterns) < num_patterns:
                parent1 = random.choice(patterns[:elite_count])
                parent2 = random.choice(patterns[:elite_count])
                
                child = self._crossover_patterns(parent1, parent2)
                child = self._mutate_pattern(child)
                new_patterns.append(child)
            
            patterns = new_patterns
        
        return sorted(patterns, key=lambda x: x["fitness"], reverse=True)
    
    def _generate_random_pattern(self) -> Dict[str, Any]:
        """Generate random connection pattern."""
        pattern_type = random.choice(["hub", "cluster", "path", "ring", "star"])
        
        pattern = {
            "type": pattern_type,
            "size": random.randint(3, min(20, self.reservoir_size // 4)),
            "strength": random.uniform(0.1, 2.0),
            "sparsity": random.uniform(0.1, 0.9),
            "nodes": [],
            "fitness": 0.0
        }
        
        # Generate node positions for pattern
        pattern["nodes"] = random.sample(range(self.reservoir_size), pattern["size"])
        
        return pattern
    
    def _apply_pattern(self, base_matrix: np.ndarray, pattern: Dict[str, Any]) -> np.ndarray:
        """Apply connection pattern to base matrix."""
        modified_matrix = base_matrix.copy()
        nodes = pattern["nodes"]
        strength = pattern["strength"]
        pattern_type = pattern["type"]
        
        if pattern_type == "hub":
            # Create hub pattern
            hub_node = nodes[0]
            for node in nodes[1:]:
                if random.random() < pattern["sparsity"]:
                    modified_matrix[hub_node, node] = strength * random.uniform(-1, 1)
                    modified_matrix[node, hub_node] = strength * random.uniform(-1, 1)
        
        elif pattern_type == "cluster":
            # Create fully connected cluster
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i != j and random.random() < pattern["sparsity"]:
                        modified_matrix[node1, node2] = strength * random.uniform(-1, 1)
        
        elif pattern_type == "path":
            # Create path pattern
            for i in range(len(nodes) - 1):
                modified_matrix[nodes[i], nodes[i+1]] = strength * random.uniform(-1, 1)
        
        elif pattern_type == "ring":
            # Create ring pattern
            for i in range(len(nodes)):
                next_node = (i + 1) % len(nodes)
                modified_matrix[nodes[i], nodes[next_node]] = strength * random.uniform(-1, 1)
        
        elif pattern_type == "star":
            # Create star pattern
            center = nodes[0]
            for node in nodes[1:]:
                modified_matrix[center, node] = strength * random.uniform(-1, 1)
                modified_matrix[node, center] = strength * random.uniform(-1, 1)
        
        return modified_matrix
    
    def _crossover_patterns(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover operation for connection patterns."""
        child = {}
        
        # Inherit pattern type from one parent
        child["type"] = random.choice([parent1["type"], parent2["type"]])
        
        # Blend numerical parameters
        child["size"] = random.choice([parent1["size"], parent2["size"]])
        child["strength"] = (parent1["strength"] + parent2["strength"]) / 2
        child["sparsity"] = (parent1["sparsity"] + parent2["sparsity"]) / 2
        
        # Combine node sets
        all_nodes = list(set(parent1["nodes"] + parent2["nodes"]))
        child["nodes"] = random.sample(all_nodes, min(child["size"], len(all_nodes)))
        
        child["fitness"] = 0.0
        
        return child
    
    def _mutate_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation for connection patterns."""
        mutated = deepcopy(pattern)
        
        # Mutate parameters
        if random.random() < 0.3:
            mutated["strength"] += random.uniform(-0.2, 0.2)
            mutated["strength"] = max(0.1, min(2.0, mutated["strength"]))
        
        if random.random() < 0.3:
            mutated["sparsity"] += random.uniform(-0.1, 0.1)
            mutated["sparsity"] = max(0.1, min(0.9, mutated["sparsity"]))
        
        # Mutate nodes
        if random.random() < 0.2:
            # Replace some nodes
            num_replace = max(1, len(mutated["nodes"]) // 4)
            available_nodes = [i for i in range(self.reservoir_size) if i not in mutated["nodes"]]
            if available_nodes:
                replace_indices = random.sample(range(len(mutated["nodes"])), 
                                              min(num_replace, len(mutated["nodes"])))
                new_nodes = random.sample(available_nodes, min(num_replace, len(available_nodes)))
                
                for i, new_node in zip(replace_indices, new_nodes):
                    mutated["nodes"][i] = new_node
        
        return mutated


class ArchitectureMutator:
    """
    Real-time reservoir architecture mutation system.
    
    Provides dynamic mutation capabilities for live reservoir systems,
    enabling continuous adaptation during operation.
    """
    
    def __init__(self, mutation_rate: float = 0.01):
        self.mutation_rate = mutation_rate
        self.mutation_history = deque(maxlen=1000)
        self.performance_impact = {}
        
    def mutate_architecture(self, reservoir, mutation_strength: float = 1.0) -> Dict[str, Any]:
        """Apply real-time mutations to reservoir architecture."""
        mutations_applied = []
        
        if not hasattr(reservoir, 'W') or reservoir.W is None:
            return {"mutations": [], "success": False}
        
        original_W = reservoir.W.copy()
        
        # Weight mutations
        if random.random() < self.mutation_rate * mutation_strength:
            mutation = self._mutate_weights(reservoir.W, mutation_strength)
            mutations_applied.append(mutation)
        
        # Connection mutations
        if random.random() < self.mutation_rate * mutation_strength * 0.5:
            mutation = self._mutate_connections(reservoir.W, mutation_strength)
            mutations_applied.append(mutation)
        
        # Structural mutations
        if random.random() < self.mutation_rate * mutation_strength * 0.2:
            mutation = self._mutate_structure(reservoir, mutation_strength)
            mutations_applied.append(mutation)
        
        # Record mutation
        mutation_record = {
            "timestamp": time.time(),
            "mutations": mutations_applied,
            "mutation_strength": mutation_strength,
            "original_spectral_radius": np.max(np.abs(np.linalg.eigvals(original_W))),
            "new_spectral_radius": np.max(np.abs(np.linalg.eigvals(reservoir.W))),
        }
        
        self.mutation_history.append(mutation_record)
        
        return {
            "mutations": mutations_applied,
            "success": True,
            "mutation_record": mutation_record
        }
    
    def _mutate_weights(self, W: np.ndarray, strength: float) -> Dict[str, Any]:
        """Mutate connection weights."""
        mutation_mask = np.random.random(W.shape) < self.mutation_rate * strength
        weight_changes = np.random.normal(0, 0.1 * strength, W.shape)
        
        # Apply mutations only to existing connections
        existing_connections = W != 0
        final_mask = mutation_mask & existing_connections
        
        W[final_mask] += weight_changes[final_mask]
        
        return {
            "type": "weight_mutation",
            "num_changes": np.sum(final_mask),
            "avg_change": np.mean(np.abs(weight_changes[final_mask])) if np.any(final_mask) else 0.0
        }
    
    def _mutate_connections(self, W: np.ndarray, strength: float) -> Dict[str, Any]:
        """Add or remove connections."""
        changes = {"added": 0, "removed": 0}
        
        # Add connections
        zero_connections = W == 0
        add_mask = np.random.random(W.shape) < self.mutation_rate * strength * 0.1
        add_mask = add_mask & zero_connections
        
        if np.any(add_mask):
            new_weights = np.random.uniform(-0.5, 0.5, W.shape)
            W[add_mask] = new_weights[add_mask]
            changes["added"] = np.sum(add_mask)
        
        # Remove connections
        nonzero_connections = W != 0
        remove_mask = np.random.random(W.shape) < self.mutation_rate * strength * 0.05
        remove_mask = remove_mask & nonzero_connections
        
        if np.any(remove_mask):
            W[remove_mask] = 0.0
            changes["removed"] = np.sum(remove_mask)
        
        return {
            "type": "connection_mutation",
            "connections_added": changes["added"],
            "connections_removed": changes["removed"]
        }
    
    def _mutate_structure(self, reservoir, strength: float) -> Dict[str, Any]:
        """Mutate higher-level structural properties."""
        mutations = []
        
        # Spectral radius adjustment
        if hasattr(reservoir, 'W') and random.random() < 0.3:
            current_sr = np.max(np.abs(np.linalg.eigvals(reservoir.W)))
            target_sr = current_sr + random.uniform(-0.1, 0.1) * strength
            target_sr = max(0.1, min(1.5, target_sr))
            
            # Scale matrix to achieve target spectral radius
            if current_sr > 0:
                reservoir.W *= target_sr / current_sr
                mutations.append(f"spectral_radius_adjusted_to_{target_sr:.3f}")
        
        # Sparsity adjustment
        if hasattr(reservoir, 'W') and random.random() < 0.2:
            # Randomly sparsify or densify
            if random.random() < 0.5:
                # Sparsify
                nonzero_mask = reservoir.W != 0
                sparsify_mask = np.random.random(reservoir.W.shape) < 0.1 * strength
                final_mask = nonzero_mask & sparsify_mask
                reservoir.W[final_mask] = 0.0
                mutations.append(f"sparsified_{np.sum(final_mask)}_connections")
            else:
                # Densify
                zero_mask = reservoir.W == 0
                densify_mask = np.random.random(reservoir.W.shape) < 0.05 * strength
                final_mask = zero_mask & densify_mask
                new_weights = np.random.uniform(-0.3, 0.3, reservoir.W.shape)
                reservoir.W[final_mask] = new_weights[final_mask]
                mutations.append(f"densified_{np.sum(final_mask)}_connections")
        
        return {
            "type": "structural_mutation",
            "mutations": mutations
        }
    
    def analyze_mutation_impact(self, performance_before: float, 
                              performance_after: float) -> Dict[str, Any]:
        """Analyze impact of recent mutations on performance."""
        if not self.mutation_history:
            return {"no_mutations": True}
        
        recent_mutation = self.mutation_history[-1]
        impact = performance_after - performance_before
        
        # Record performance impact
        mutation_id = f"{recent_mutation['timestamp']}"
        self.performance_impact[mutation_id] = {
            "impact": impact,
            "performance_before": performance_before,
            "performance_after": performance_after,
            "mutations": recent_mutation["mutations"]
        }
        
        analysis = {
            "performance_change": impact,
            "improvement": impact > 0,
            "mutation_beneficial": impact > 0.01,
            "mutation_harmful": impact < -0.01,
            "mutation_neutral": abs(impact) <= 0.01,
            "recent_mutation": recent_mutation
        }
        
        return analysis


class EvolutionaryOptimizer:
    """
    Main evolutionary optimizer coordinating all evolutionary processes.
    
    Integrates topology evolution, parameter evolution, connection patterns,
    and architecture mutations for comprehensive reservoir optimization.
    """
    
    def __init__(self, reservoir_size: int = 100):
        self.reservoir_size = reservoir_size
        self.topology_evolver = TopologyEvolver(reservoir_size)
        self.parameter_evolver = ParameterEvolver()
        self.connection_evolver = ConnectionEvolver(reservoir_size)
        self.architecture_mutator = ArchitectureMutator()
        
        self.optimization_history = deque(maxlen=500)
        self.best_configurations = []
        
    def optimize_reservoir(self, initial_reservoir, fitness_function: Callable,
                          optimization_targets: List[str] = None) -> Dict[str, Any]:
        """Comprehensive evolutionary optimization of reservoir."""
        if optimization_targets is None:
            optimization_targets = ["topology", "parameters", "connections"]
        
        optimization_results = {
            "initial_fitness": 0.0,
            "final_fitness": 0.0,
            "improvement_ratio": 0.0,
            "optimizations_performed": []
        }
        
        # Evaluate initial fitness
        try:
            if hasattr(initial_reservoir, 'W'):
                initial_fitness = fitness_function({"W": initial_reservoir.W})
            else:
                initial_fitness = 0.1
        except:
            initial_fitness = 0.1
        
        optimization_results["initial_fitness"] = initial_fitness
        current_reservoir = initial_reservoir
        current_fitness = initial_fitness
        
        # Topology optimization
        if "topology" in optimization_targets:
            topology_result = self._optimize_topology(current_reservoir, fitness_function)
            if topology_result["success"]:
                current_reservoir = topology_result["optimized_reservoir"]
                current_fitness = topology_result["fitness"]
                optimization_results["optimizations_performed"].append(topology_result)
        
        # Parameter optimization
        if "parameters" in optimization_targets:
            parameter_result = self._optimize_parameters(current_reservoir, fitness_function)
            if parameter_result["success"]:
                current_reservoir = parameter_result["optimized_reservoir"]
                current_fitness = parameter_result["fitness"]
                optimization_results["optimizations_performed"].append(parameter_result)
        
        # Connection pattern optimization
        if "connections" in optimization_targets:
            connection_result = self._optimize_connections(current_reservoir, fitness_function)
            if connection_result["success"]:
                current_reservoir = connection_result["optimized_reservoir"]
                current_fitness = connection_result["fitness"]
                optimization_results["optimizations_performed"].append(connection_result)
        
        # Final evaluation
        optimization_results["final_fitness"] = current_fitness
        optimization_results["improvement_ratio"] = current_fitness / max(0.001, initial_fitness)
        optimization_results["optimized_reservoir"] = current_reservoir
        
        # Record optimization
        self.optimization_history.append({
            "timestamp": time.time(),
            "results": optimization_results,
            "targets": optimization_targets
        })
        
        return optimization_results
    
    def _optimize_topology(self, reservoir, fitness_function: Callable) -> Dict[str, Any]:
        """Optimize reservoir topology."""
        try:
            # Create fitness function wrapper
            def topology_fitness(genome):
                return fitness_function(genome)
            
            # Run topology evolution
            best_individual = self.topology_evolver.evolve_topology(
                topology_fitness, generations=20  # Reduced for efficiency
            )
            
            if best_individual and best_individual.fitness > 0:
                # Apply best topology to reservoir
                if hasattr(reservoir, 'W') and "W" in best_individual.genome:
                    reservoir.W = best_individual.genome["W"]
                
                return {
                    "type": "topology_optimization",
                    "success": True,
                    "fitness": best_individual.fitness,
                    "optimized_reservoir": reservoir,
                    "best_genome": best_individual.genome
                }
        except Exception as e:
            pass
        
        return {"type": "topology_optimization", "success": False}
    
    def _optimize_parameters(self, reservoir, fitness_function: Callable) -> Dict[str, Any]:
        """Optimize reservoir parameters."""
        try:
            # Create parameter fitness function
            def parameter_fitness(params):
                # Create temporary reservoir with new parameters
                temp_reservoir = deepcopy(reservoir)
                
                # Apply parameters (simplified for demonstration)
                if "spectral_radius" in params and hasattr(temp_reservoir, 'W'):
                    current_sr = np.max(np.abs(np.linalg.eigvals(temp_reservoir.W)))
                    if current_sr > 0:
                        temp_reservoir.W *= params["spectral_radius"] / current_sr
                
                return fitness_function({"W": temp_reservoir.W})
            
            # Run parameter evolution
            best_params = self.parameter_evolver.evolve_parameters(
                parameter_fitness, ["spectral_radius", "leaking_rate", "input_scaling"]
            )
            
            if best_params:
                # Apply best parameters
                fitness = parameter_fitness(best_params)
                
                return {
                    "type": "parameter_optimization",
                    "success": True,
                    "fitness": fitness,
                    "optimized_reservoir": reservoir,
                    "best_parameters": best_params
                }
        except Exception as e:
            pass
        
        return {"type": "parameter_optimization", "success": False}
    
    def _optimize_connections(self, reservoir, fitness_function: Callable) -> Dict[str, Any]:
        """Optimize connection patterns."""
        try:
            if not hasattr(reservoir, 'W'):
                return {"type": "connection_optimization", "success": False}
            
            # Create connection fitness function
            def connection_fitness(genome):
                return fitness_function(genome)
            
            # Evolve connection patterns
            best_patterns = self.connection_evolver.evolve_connection_patterns(
                reservoir.W, connection_fitness, num_patterns=5
            )
            
            if best_patterns and best_patterns[0]["fitness"] > 0:
                # Apply best pattern
                best_pattern = best_patterns[0]
                optimized_matrix = self.connection_evolver._apply_pattern(reservoir.W, best_pattern)
                reservoir.W = optimized_matrix
                
                return {
                    "type": "connection_optimization",
                    "success": True,
                    "fitness": best_pattern["fitness"],
                    "optimized_reservoir": reservoir,
                    "best_pattern": best_pattern
                }
        except Exception as e:
            pass
        
        return {"type": "connection_optimization", "success": False}
    
    def continuous_optimization(self, reservoir, fitness_function: Callable,
                              mutation_interval: float = 10.0) -> Dict[str, Any]:
        """Continuous real-time optimization using mutations."""
        optimization_log = {
            "start_time": time.time(),
            "mutations_applied": 0,
            "improvements": 0,
            "performance_history": []
        }
        
        # Get baseline performance
        try:
            baseline_fitness = fitness_function({"W": reservoir.W}) if hasattr(reservoir, 'W') else 0.1
        except:
            baseline_fitness = 0.1
        
        optimization_log["performance_history"].append(baseline_fitness)
        
        # Apply continuous mutations
        mutation_count = 0
        last_mutation_time = time.time()
        
        while mutation_count < 10:  # Limit for demonstration
            current_time = time.time()
            
            if current_time - last_mutation_time >= mutation_interval:
                # Apply mutation
                performance_before = optimization_log["performance_history"][-1]
                
                mutation_result = self.architecture_mutator.mutate_architecture(
                    reservoir, mutation_strength=0.5
                )
                
                if mutation_result["success"]:
                    # Evaluate new performance
                    try:
                        performance_after = fitness_function({"W": reservoir.W})
                    except:
                        performance_after = performance_before
                    
                    # Analyze impact
                    impact_analysis = self.architecture_mutator.analyze_mutation_impact(
                        performance_before, performance_after
                    )
                    
                    optimization_log["performance_history"].append(performance_after)
                    optimization_log["mutations_applied"] += 1
                    
                    if impact_analysis.get("improvement", False):
                        optimization_log["improvements"] += 1
                
                last_mutation_time = current_time
                mutation_count += 1
        
        # Calculate final results
        final_performance = optimization_log["performance_history"][-1]
        improvement_ratio = final_performance / max(0.001, baseline_fitness)
        
        optimization_log.update({
            "final_performance": final_performance,
            "improvement_ratio": improvement_ratio,
            "duration": time.time() - optimization_log["start_time"]
        })
        
        return optimization_log