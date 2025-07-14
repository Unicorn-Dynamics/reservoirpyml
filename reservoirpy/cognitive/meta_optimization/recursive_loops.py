"""
Recursive Improvement Loops for Self-Enhancement
===============================================

This module implements recursive feedback mechanisms, ensemble evolution,
meta-learning algorithms, and hierarchical optimization for continuous improvement.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import time
from collections import deque, defaultdict
from copy import deepcopy

try:
    from ...nodes import Reservoir
    from ...model import Model
    from ..attention import ECANAttentionSystem
    from ..distributed import MeshCoordinator
    from .meta_cognitive import MetaCognitiveSystem
    from .evolutionary import EvolutionaryOptimizer
    from .benchmarking import PerformanceBenchmarker
except ImportError:
    # Fallback for minimal dependencies
    pass


@dataclass
class FeedbackLoop:
    """Recursive feedback loop for system improvement."""
    loop_id: str
    source: str
    target: str
    feedback_function: Callable
    strength: float
    active: bool = True
    history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


@dataclass
class ImprovementCycle:
    """Single cycle of recursive improvement."""
    cycle_id: int
    timestamp: float
    performance_before: float
    performance_after: float
    improvements_applied: List[Dict[str, Any]]
    feedback_loops_triggered: List[str]
    success: bool


class EmbodiedFeedback:
    """
    Feedback mechanisms from embodied agent performance.
    
    Collects and processes feedback from physical and virtual agents
    to guide system optimization and adaptation.
    """
    
    def __init__(self, feedback_window: int = 100):
        self.feedback_window = feedback_window
        self.agent_feedback = defaultdict(lambda: deque(maxlen=feedback_window))
        self.performance_correlations = {}
        self.feedback_processors = {}
        self.embodiment_metrics = {}
        
    def register_embodied_agent(self, agent_id: str, agent_type: str = "generic"):
        """Register embodied agent for feedback collection."""
        self.agent_feedback[agent_id] = deque(maxlen=self.feedback_window)
        self.embodiment_metrics[agent_id] = {
            "type": agent_type,
            "total_feedback": 0,
            "average_performance": 0.0,
            "improvement_rate": 0.0,
            "last_update": time.time()
        }
    
    def collect_agent_feedback(self, agent_id: str, performance_data: Dict[str, float],
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Collect performance feedback from embodied agent."""
        feedback_entry = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "performance": performance_data,
            "context": context or {},
            "processed": False
        }
        
        self.agent_feedback[agent_id].append(feedback_entry)
        
        # Update agent metrics
        if agent_id in self.embodiment_metrics:
            metrics = self.embodiment_metrics[agent_id]
            metrics["total_feedback"] += 1
            metrics["last_update"] = feedback_entry["timestamp"]
            
            # Update average performance
            recent_performances = [entry["performance"].get("overall", 0.5) 
                                 for entry in list(self.agent_feedback[agent_id])[-10:]]
            metrics["average_performance"] = np.mean(recent_performances)
            
            # Calculate improvement rate
            if len(self.agent_feedback[agent_id]) >= 10:
                old_performances = [entry["performance"].get("overall", 0.5) 
                                  for entry in list(self.agent_feedback[agent_id])[:5]]
                new_performances = [entry["performance"].get("overall", 0.5) 
                                  for entry in list(self.agent_feedback[agent_id])[-5:]]
                
                old_avg = np.mean(old_performances)
                new_avg = np.mean(new_performances)
                metrics["improvement_rate"] = (new_avg - old_avg) / max(old_avg, 0.001)
        
        return feedback_entry
    
    def process_embodied_feedback(self, cognitive_system) -> Dict[str, Any]:
        """Process collected feedback to generate system improvements."""
        processing_results = {
            "feedback_processed": 0,
            "insights_generated": [],
            "system_adjustments": [],
            "performance_correlations": {},
            "agent_summaries": {}
        }
        
        # Process feedback from each agent
        for agent_id, feedback_queue in self.agent_feedback.items():
            unprocessed_feedback = [entry for entry in feedback_queue if not entry["processed"]]
            
            if unprocessed_feedback:
                agent_insights = self._process_agent_feedback(agent_id, unprocessed_feedback, cognitive_system)
                processing_results["insights_generated"].extend(agent_insights)
                processing_results["feedback_processed"] += len(unprocessed_feedback)
                
                # Mark as processed
                for entry in unprocessed_feedback:
                    entry["processed"] = True
                
                # Generate agent summary
                processing_results["agent_summaries"][agent_id] = self._generate_agent_summary(agent_id)
        
        # Analyze cross-agent patterns
        cross_agent_insights = self._analyze_cross_agent_patterns()
        processing_results["insights_generated"].extend(cross_agent_insights)
        
        # Generate system adjustments
        system_adjustments = self._generate_system_adjustments(processing_results["insights_generated"])
        processing_results["system_adjustments"] = system_adjustments
        
        return processing_results
    
    def _process_agent_feedback(self, agent_id: str, feedback_entries: List[Dict[str, Any]], 
                              cognitive_system) -> List[Dict[str, Any]]:
        """Process feedback from specific agent."""
        insights = []
        
        # Analyze performance trends
        performances = [entry["performance"].get("overall", 0.5) for entry in feedback_entries]
        
        if len(performances) >= 3:
            trend = self._calculate_trend(performances)
            
            if trend > 0.01:
                insights.append({
                    "type": "performance_improvement",
                    "agent_id": agent_id,
                    "trend": trend,
                    "recommendation": "maintain_current_configuration",
                    "confidence": min(1.0, abs(trend) * 10)
                })
            elif trend < -0.01:
                insights.append({
                    "type": "performance_degradation", 
                    "agent_id": agent_id,
                    "trend": trend,
                    "recommendation": "adjust_cognitive_parameters",
                    "confidence": min(1.0, abs(trend) * 10)
                })
        
        # Analyze task-specific performance
        task_performances = defaultdict(list)
        for entry in feedback_entries:
            context = entry.get("context", {})
            task_type = context.get("task_type", "general")
            task_performances[task_type].append(entry["performance"].get("overall", 0.5))
        
        for task_type, performances in task_performances.items():
            if len(performances) >= 2:
                avg_performance = np.mean(performances)
                if avg_performance < 0.3:
                    insights.append({
                        "type": "task_difficulty",
                        "agent_id": agent_id,
                        "task_type": task_type,
                        "performance": avg_performance,
                        "recommendation": "enhance_task_specific_capabilities",
                        "confidence": 1.0 - avg_performance
                    })
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(feedback_entries)
        insights.extend(error_patterns)
        
        return insights
    
    def _analyze_cross_agent_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns across multiple agents."""
        insights = []
        
        if len(self.agent_feedback) < 2:
            return insights
        
        # Compare agent performance
        agent_performances = {}
        for agent_id, feedback_queue in self.agent_feedback.items():
            if feedback_queue:
                recent_performances = [entry["performance"].get("overall", 0.5) 
                                     for entry in list(feedback_queue)[-10:]]
                agent_performances[agent_id] = np.mean(recent_performances)
        
        if len(agent_performances) >= 2:
            best_agent = max(agent_performances, key=agent_performances.get)
            worst_agent = min(agent_performances, key=agent_performances.get)
            
            performance_gap = agent_performances[best_agent] - agent_performances[worst_agent]
            
            if performance_gap > 0.2:
                insights.append({
                    "type": "agent_performance_disparity",
                    "best_agent": best_agent,
                    "worst_agent": worst_agent,
                    "performance_gap": performance_gap,
                    "recommendation": "transfer_learning_from_best_to_worst",
                    "confidence": min(1.0, performance_gap)
                })
        
        # Analyze common failure modes
        common_failures = self._identify_common_failures()
        insights.extend(common_failures)
        
        return insights
    
    def _generate_system_adjustments(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate system adjustments based on insights."""
        adjustments = []
        
        # Group insights by type
        insight_groups = defaultdict(list)
        for insight in insights:
            insight_groups[insight["type"]].append(insight)
        
        # Generate adjustments for each insight type
        for insight_type, group_insights in insight_groups.items():
            if insight_type == "performance_degradation":
                adjustments.append({
                    "type": "parameter_adjustment",
                    "target": "cognitive_parameters",
                    "action": "increase_adaptation_rate",
                    "magnitude": min(0.2, len(group_insights) * 0.05),
                    "reason": f"Multiple agents showing performance degradation"
                })
            
            elif insight_type == "task_difficulty":
                task_types = [insight["task_type"] for insight in group_insights]
                most_common_task = max(set(task_types), key=task_types.count)
                
                adjustments.append({
                    "type": "capability_enhancement",
                    "target": "task_specific_modules",
                    "action": "enhance_task_capabilities",
                    "task_type": most_common_task,
                    "magnitude": 0.3,
                    "reason": f"Multiple agents struggling with {most_common_task} tasks"
                })
            
            elif insight_type == "agent_performance_disparity":
                adjustments.append({
                    "type": "knowledge_transfer",
                    "target": "agent_configurations",
                    "action": "transfer_best_practices",
                    "source": group_insights[0]["best_agent"],
                    "targets": [insight["worst_agent"] for insight in group_insights],
                    "magnitude": 0.4,
                    "reason": "Significant performance disparities detected"
                })
        
        return adjustments
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in performance values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _analyze_error_patterns(self, feedback_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze error patterns in feedback."""
        insights = []
        
        error_counts = defaultdict(int)
        for entry in feedback_entries:
            context = entry.get("context", {})
            errors = context.get("errors", [])
            for error in errors:
                error_counts[error] += 1
        
        # Identify frequent errors
        total_entries = len(feedback_entries)
        for error_type, count in error_counts.items():
            if count / total_entries > 0.3:  # Error occurs in >30% of cases
                insights.append({
                    "type": "frequent_error",
                    "error_type": error_type,
                    "frequency": count / total_entries,
                    "recommendation": f"address_{error_type}_error",
                    "confidence": count / total_entries
                })
        
        return insights
    
    def _identify_common_failures(self) -> List[Dict[str, Any]]:
        """Identify failure modes common across agents."""
        insights = []
        
        # Collect all failure modes
        all_failures = defaultdict(int)
        total_agents = 0
        
        for agent_id, feedback_queue in self.agent_feedback.items():
            if not feedback_queue:
                continue
                
            total_agents += 1
            agent_failures = set()
            
            for entry in feedback_queue:
                context = entry.get("context", {})
                failures = context.get("failures", [])
                agent_failures.update(failures)
            
            for failure in agent_failures:
                all_failures[failure] += 1
        
        # Identify failures affecting multiple agents
        if total_agents > 1:
            for failure_type, agent_count in all_failures.items():
                if agent_count / total_agents > 0.5:  # Affects >50% of agents
                    insights.append({
                        "type": "common_failure_mode",
                        "failure_type": failure_type,
                        "affected_agents": agent_count,
                        "prevalence": agent_count / total_agents,
                        "recommendation": "systematic_failure_mitigation",
                        "confidence": agent_count / total_agents
                    })
        
        return insights
    
    def _generate_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """Generate summary for specific agent."""
        if agent_id not in self.embodiment_metrics:
            return {"error": "Agent not found"}
        
        metrics = self.embodiment_metrics[agent_id]
        feedback_queue = self.agent_feedback[agent_id]
        
        summary = {
            "agent_id": agent_id,
            "agent_type": metrics["type"],
            "total_feedback_entries": len(feedback_queue),
            "average_performance": metrics["average_performance"],
            "improvement_rate": metrics["improvement_rate"],
            "last_update": metrics["last_update"],
            "status": "active" if time.time() - metrics["last_update"] < 300 else "inactive"
        }
        
        # Add recent performance trend
        if len(feedback_queue) >= 5:
            recent_performances = [entry["performance"].get("overall", 0.5) 
                                 for entry in list(feedback_queue)[-5:]]
            summary["recent_trend"] = self._calculate_trend(recent_performances)
            summary["performance_stability"] = 1.0 - np.std(recent_performances)
        
        return summary


class EnsembleEvolver:
    """
    Reservoir ensemble evolution for improved robustness.
    
    Evolves collections of reservoir instances for better performance
    through diversity, specialization, and cooperative learning.
    """
    
    def __init__(self, max_ensemble_size: int = 10):
        self.max_ensemble_size = max_ensemble_size
        self.ensembles = {}
        self.ensemble_performance = {}
        self.evolution_history = deque(maxlen=200)
        
    def create_ensemble(self, ensemble_id: str, base_reservoirs: List,
                       diversity_target: float = 0.7) -> Dict[str, Any]:
        """Create new reservoir ensemble."""
        ensemble = {
            "id": ensemble_id,
            "reservoirs": base_reservoirs[:self.max_ensemble_size],
            "diversity_target": diversity_target,
            "performance_history": deque(maxlen=100),
            "specializations": {},
            "cooperation_matrix": np.eye(len(base_reservoirs)),
            "creation_time": time.time()
        }
        
        # Initialize ensemble diversity
        ensemble["diversity_score"] = self._calculate_ensemble_diversity(base_reservoirs)
        
        self.ensembles[ensemble_id] = ensemble
        return ensemble
    
    def evolve_ensemble(self, ensemble_id: str, fitness_function: Callable,
                       evolution_cycles: int = 20) -> Dict[str, Any]:
        """Evolve reservoir ensemble for improved performance."""
        if ensemble_id not in self.ensembles:
            return {"error": "Ensemble not found"}
        
        ensemble = self.ensembles[ensemble_id]
        evolution_results = {
            "initial_performance": 0.0,
            "final_performance": 0.0,
            "diversity_improvement": 0.0,
            "specialization_developed": [],
            "cooperation_improved": False,
            "evolution_cycles": evolution_cycles
        }
        
        # Evaluate initial performance
        initial_performance = self._evaluate_ensemble_performance(ensemble, fitness_function)
        evolution_results["initial_performance"] = initial_performance
        
        for cycle in range(evolution_cycles):
            # Diversity-driven evolution
            self._evolve_ensemble_diversity(ensemble, fitness_function)
            
            # Specialization development
            self._develop_ensemble_specializations(ensemble, fitness_function)
            
            # Cooperation enhancement
            self._enhance_ensemble_cooperation(ensemble, fitness_function)
            
            # Evaluate progress
            current_performance = self._evaluate_ensemble_performance(ensemble, fitness_function)
            ensemble["performance_history"].append(current_performance)
            
            # Early stopping if converged
            if len(ensemble["performance_history"]) >= 5:
                recent_performances = list(ensemble["performance_history"])[-5:]
                if np.std(recent_performances) < 0.01:
                    break
        
        # Final evaluation
        final_performance = self._evaluate_ensemble_performance(ensemble, fitness_function)
        evolution_results["final_performance"] = final_performance
        evolution_results["diversity_improvement"] = (
            self._calculate_ensemble_diversity(ensemble["reservoirs"]) - ensemble["diversity_score"]
        )
        
        # Record evolution
        self.evolution_history.append({
            "timestamp": time.time(),
            "ensemble_id": ensemble_id,
            "results": evolution_results
        })
        
        return evolution_results
    
    def _calculate_ensemble_diversity(self, reservoirs: List) -> float:
        """Calculate diversity score for reservoir ensemble."""
        if len(reservoirs) < 2:
            return 0.0
        
        diversities = []
        
        for i, res1 in enumerate(reservoirs):
            for res2 in reservoirs[i+1:]:
                if hasattr(res1, 'W') and hasattr(res2, 'W') and res1.W is not None and res2.W is not None:
                    # Matrix-based diversity
                    if res1.W.shape == res2.W.shape:
                        diff = np.linalg.norm(res1.W - res2.W)
                        similarity = 1.0 / (1.0 + diff)
                        diversity = 1.0 - similarity
                        diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.5
    
    def _evaluate_ensemble_performance(self, ensemble: Dict[str, Any], 
                                     fitness_function: Callable) -> float:
        """Evaluate overall ensemble performance."""
        reservoirs = ensemble["reservoirs"]
        
        if not reservoirs:
            return 0.0
        
        # Individual performances
        individual_performances = []
        for reservoir in reservoirs:
            try:
                if hasattr(reservoir, 'W') and reservoir.W is not None:
                    performance = fitness_function({"W": reservoir.W})
                else:
                    performance = 0.1
                individual_performances.append(performance)
            except:
                individual_performances.append(0.1)
        
        # Ensemble performance combines individual and diversity benefits
        avg_performance = np.mean(individual_performances)
        diversity_bonus = ensemble.get("diversity_score", 0.5) * 0.2
        cooperation_bonus = self._calculate_cooperation_bonus(ensemble) * 0.1
        
        ensemble_performance = avg_performance + diversity_bonus + cooperation_bonus
        return min(1.0, ensemble_performance)
    
    def _evolve_ensemble_diversity(self, ensemble: Dict[str, Any], fitness_function: Callable):
        """Evolve ensemble to increase diversity while maintaining performance."""
        reservoirs = ensemble["reservoirs"]
        target_diversity = ensemble["diversity_target"]
        current_diversity = self._calculate_ensemble_diversity(reservoirs)
        
        if current_diversity < target_diversity:
            # Add diversity through mutation
            for reservoir in reservoirs:
                if hasattr(reservoir, 'W') and reservoir.W is not None and random.random() < 0.3:
                    # Apply diversity-promoting mutation
                    mutation_strength = (target_diversity - current_diversity) * 0.5
                    self._apply_diversity_mutation(reservoir, mutation_strength)
            
            # Update diversity score
            ensemble["diversity_score"] = self._calculate_ensemble_diversity(reservoirs)
    
    def _develop_ensemble_specializations(self, ensemble: Dict[str, Any], fitness_function: Callable):
        """Develop specializations within ensemble members."""
        reservoirs = ensemble["reservoirs"]
        specializations = ensemble["specializations"]
        
        # Define specialization types
        specialization_types = ["memory", "speed", "accuracy", "stability", "exploration"]
        
        for i, reservoir in enumerate(reservoirs):
            if f"reservoir_{i}" not in specializations:
                # Assign specialization
                specialization = random.choice(specialization_types)
                specializations[f"reservoir_{i}"] = specialization
                
                # Apply specialization-specific modifications
                self._apply_specialization(reservoir, specialization)
    
    def _enhance_ensemble_cooperation(self, ensemble: Dict[str, Any], fitness_function: Callable):
        """Enhance cooperation between ensemble members."""
        reservoirs = ensemble["reservoirs"]
        cooperation_matrix = ensemble["cooperation_matrix"]
        
        # Evaluate pairwise cooperation benefits
        for i in range(len(reservoirs)):
            for j in range(i+1, len(reservoirs)):
                if i < cooperation_matrix.shape[0] and j < cooperation_matrix.shape[1]:
                    # Test cooperation strength
                    cooperation_strength = self._test_cooperation(reservoirs[i], reservoirs[j], fitness_function)
                    cooperation_matrix[i, j] = cooperation_strength
                    cooperation_matrix[j, i] = cooperation_strength
        
        ensemble["cooperation_matrix"] = cooperation_matrix
    
    def _apply_diversity_mutation(self, reservoir, mutation_strength: float):
        """Apply mutation to increase reservoir diversity."""
        if hasattr(reservoir, 'W') and reservoir.W is not None:
            # Random weight perturbations
            mutation_mask = np.random.random(reservoir.W.shape) < mutation_strength * 0.1
            mutations = np.random.normal(0, mutation_strength * 0.2, reservoir.W.shape)
            reservoir.W[mutation_mask] += mutations[mutation_mask]
    
    def _apply_specialization(self, reservoir, specialization: str):
        """Apply specialization-specific modifications to reservoir."""
        if not hasattr(reservoir, 'W') or reservoir.W is None:
            return
        
        if specialization == "memory":
            # Enhance memory by reducing spectral radius slightly
            eigenvals = np.linalg.eigvals(reservoir.W)
            current_sr = np.max(np.abs(eigenvals))
            target_sr = min(current_sr, 0.95)
            if current_sr > 0:
                reservoir.W *= target_sr / current_sr
        
        elif specialization == "speed":
            # Enhance speed by increasing sparsity
            nonzero_mask = reservoir.W != 0
            sparsify_mask = np.random.random(reservoir.W.shape) < 0.1
            final_mask = nonzero_mask & sparsify_mask
            reservoir.W[final_mask] = 0.0
        
        elif specialization == "accuracy":
            # Enhance accuracy by optimizing spectral radius
            eigenvals = np.linalg.eigvals(reservoir.W)
            current_sr = np.max(np.abs(eigenvals))
            target_sr = 0.9  # Optimal for accuracy
            if current_sr > 0:
                reservoir.W *= target_sr / current_sr
        
        elif specialization == "stability":
            # Enhance stability by reducing variance
            reservoir.W *= 0.9  # Slight reduction for stability
        
        elif specialization == "exploration":
            # Enhance exploration by adding noise
            noise = np.random.normal(0, 0.05, reservoir.W.shape)
            reservoir.W += noise
    
    def _test_cooperation(self, reservoir1, reservoir2, fitness_function: Callable) -> float:
        """Test cooperation strength between two reservoirs."""
        try:
            # Simple cooperation test: average their connection matrices
            if (hasattr(reservoir1, 'W') and hasattr(reservoir2, 'W') and 
                reservoir1.W is not None and reservoir2.W is not None and
                reservoir1.W.shape == reservoir2.W.shape):
                
                combined_W = (reservoir1.W + reservoir2.W) / 2.0
                combined_performance = fitness_function({"W": combined_W})
                
                individual_perf1 = fitness_function({"W": reservoir1.W})
                individual_perf2 = fitness_function({"W": reservoir2.W})
                individual_avg = (individual_perf1 + individual_perf2) / 2.0
                
                cooperation_benefit = combined_performance - individual_avg
                return max(0.0, min(1.0, cooperation_benefit + 0.5))
        except:
            pass
        
        return 0.5  # Default cooperation strength
    
    def _calculate_cooperation_bonus(self, ensemble: Dict[str, Any]) -> float:
        """Calculate cooperation bonus for ensemble."""
        cooperation_matrix = ensemble.get("cooperation_matrix", np.eye(2))
        
        # Average off-diagonal elements (cooperation strengths)
        if cooperation_matrix.shape[0] > 1:
            off_diagonal = cooperation_matrix[np.triu_indices_from(cooperation_matrix, k=1)]
            return np.mean(off_diagonal) if len(off_diagonal) > 0 else 0.0
        
        return 0.0


class MetaLearner:
    """
    Meta-learning algorithms for rapid adaptation.
    
    Implements learning-to-learn capabilities that enable the system
    to quickly adapt to new tasks and environments.
    """
    
    def __init__(self, memory_size: int = 200):
        self.memory_size = memory_size
        self.experience_memory = deque(maxlen=memory_size)
        self.meta_models = {}
        self.adaptation_strategies = {}
        self.learning_history = deque(maxlen=100)
        
    def learn_adaptation_strategy(self, task_type: str, adaptation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn adaptation strategy for specific task type."""
        strategy = {
            "task_type": task_type,
            "patterns": self._extract_adaptation_patterns(adaptation_data),
            "success_rate": self._calculate_success_rate(adaptation_data),
            "optimal_parameters": self._find_optimal_parameters(adaptation_data),
            "learned_time": time.time()
        }
        
        self.adaptation_strategies[task_type] = strategy
        return strategy
    
    def rapid_adapt(self, new_task_context: Dict[str, Any], target_system) -> Dict[str, Any]:
        """Rapidly adapt system to new task using meta-learning."""
        task_type = new_task_context.get("task_type", "unknown")
        
        adaptation_result = {
            "task_type": task_type,
            "adaptation_time": 0.0,
            "performance_improvement": 0.0,
            "strategy_used": "none",
            "success": False
        }
        
        start_time = time.time()
        
        # Find relevant adaptation strategy
        strategy = self._find_relevant_strategy(task_type, new_task_context)
        
        if strategy:
            # Apply learned adaptation strategy
            adaptation_success = self._apply_adaptation_strategy(strategy, target_system, new_task_context)
            adaptation_result["strategy_used"] = strategy["task_type"]
            adaptation_result["success"] = adaptation_success
        else:
            # Use meta-learning for novel task
            adaptation_success = self._meta_adapt_novel_task(target_system, new_task_context)
            adaptation_result["strategy_used"] = "meta_learning"
            adaptation_result["success"] = adaptation_success
        
        adaptation_result["adaptation_time"] = time.time() - start_time
        
        # Record learning experience
        self.experience_memory.append({
            "timestamp": time.time(),
            "task_context": new_task_context,
            "adaptation_result": adaptation_result,
            "system_state": self._capture_system_state(target_system)
        })
        
        return adaptation_result
    
    def _extract_adaptation_patterns(self, adaptation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from adaptation data."""
        patterns = {
            "common_parameters": {},
            "success_indicators": [],
            "failure_indicators": [],
            "temporal_patterns": {}
        }
        
        # Analyze successful adaptations
        successful_adaptations = [data for data in adaptation_data if data.get("success", False)]
        
        if successful_adaptations:
            # Extract common parameters
            parameter_frequencies = defaultdict(list)
            for adaptation in successful_adaptations:
                parameters = adaptation.get("parameters", {})
                for param, value in parameters.items():
                    parameter_frequencies[param].append(value)
            
            for param, values in parameter_frequencies.items():
                patterns["common_parameters"][param] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "frequency": len(values) / len(adaptation_data)
                }
            
            # Extract success indicators
            for adaptation in successful_adaptations:
                indicators = adaptation.get("success_indicators", [])
                patterns["success_indicators"].extend(indicators)
        
        # Analyze failure patterns
        failed_adaptations = [data for data in adaptation_data if not data.get("success", False)]
        
        for adaptation in failed_adaptations:
            indicators = adaptation.get("failure_indicators", [])
            patterns["failure_indicators"].extend(indicators)
        
        return patterns
    
    def _calculate_success_rate(self, adaptation_data: List[Dict[str, Any]]) -> float:
        """Calculate success rate for adaptation strategy."""
        if not adaptation_data:
            return 0.0
        
        successful = sum(1 for data in adaptation_data if data.get("success", False))
        return successful / len(adaptation_data)
    
    def _find_optimal_parameters(self, adaptation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Find optimal parameters from successful adaptations."""
        successful_adaptations = [data for data in adaptation_data if data.get("success", False)]
        
        if not successful_adaptations:
            return {}
        
        # Weight parameters by performance
        parameter_values = defaultdict(list)
        performance_weights = []
        
        for adaptation in successful_adaptations:
            parameters = adaptation.get("parameters", {})
            performance = adaptation.get("performance", 0.5)
            
            for param, value in parameters.items():
                parameter_values[param].append((value, performance))
            performance_weights.append(performance)
        
        # Calculate weighted averages
        optimal_parameters = {}
        for param, value_performance_pairs in parameter_values.items():
            values = [vp[0] for vp in value_performance_pairs]
            weights = [vp[1] for vp in value_performance_pairs]
            
            if weights:
                weighted_avg = np.average(values, weights=weights)
                optimal_parameters[param] = weighted_avg
        
        return optimal_parameters
    
    def _find_relevant_strategy(self, task_type: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find most relevant adaptation strategy for task."""
        if task_type in self.adaptation_strategies:
            return self.adaptation_strategies[task_type]
        
        # Find similar task types
        similarities = {}
        for strategy_task_type, strategy in self.adaptation_strategies.items():
            similarity = self._calculate_task_similarity(task_type, strategy_task_type, context)
            similarities[strategy_task_type] = similarity
        
        if similarities:
            best_match = max(similarities, key=similarities.get)
            if similarities[best_match] > 0.6:  # Similarity threshold
                return self.adaptation_strategies[best_match]
        
        return None
    
    def _calculate_task_similarity(self, task_type1: str, task_type2: str, 
                                 context: Dict[str, Any]) -> float:
        """Calculate similarity between task types."""
        # Simple string similarity for task types
        type_similarity = len(set(task_type1.lower()) & set(task_type2.lower())) / max(len(task_type1), len(task_type2), 1)
        
        # Context similarity (simplified)
        context_similarity = 0.5  # Default
        
        return (type_similarity + context_similarity) / 2.0
    
    def _apply_adaptation_strategy(self, strategy: Dict[str, Any], target_system, 
                                 context: Dict[str, Any]) -> bool:
        """Apply learned adaptation strategy to target system."""
        try:
            optimal_params = strategy.get("optimal_parameters", {})
            
            # Apply parameters to system
            for param, value in optimal_params.items():
                if hasattr(target_system, param):
                    setattr(target_system, param, value)
                elif hasattr(target_system, f"set_{param}"):
                    getattr(target_system, f"set_{param}")(value)
            
            return True
        except Exception as e:
            return False
    
    def _meta_adapt_novel_task(self, target_system, context: Dict[str, Any]) -> bool:
        """Use meta-learning to adapt to novel task."""
        try:
            # Extract general adaptation principles from experience
            if len(self.experience_memory) < 5:
                return False
            
            # Analyze successful adaptations
            successful_experiences = [exp for exp in self.experience_memory 
                                    if exp["adaptation_result"]["success"]]
            
            if not successful_experiences:
                return False
            
            # Extract common successful patterns
            successful_patterns = self._extract_meta_patterns(successful_experiences)
            
            # Apply meta-patterns to new task
            return self._apply_meta_patterns(successful_patterns, target_system, context)
        
        except Exception as e:
            return False
    
    def _extract_meta_patterns(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract meta-level patterns from successful experiences."""
        patterns = {
            "adaptation_time_patterns": [],
            "system_state_patterns": {},
            "context_patterns": {}
        }
        
        for experience in experiences:
            adaptation_result = experience["adaptation_result"]
            patterns["adaptation_time_patterns"].append(adaptation_result["adaptation_time"])
            
            # Analyze system state patterns
            system_state = experience.get("system_state", {})
            for key, value in system_state.items():
                if key not in patterns["system_state_patterns"]:
                    patterns["system_state_patterns"][key] = []
                patterns["system_state_patterns"][key].append(value)
        
        return patterns
    
    def _apply_meta_patterns(self, patterns: Dict[str, Any], target_system, 
                           context: Dict[str, Any]) -> bool:
        """Apply meta-patterns to target system."""
        try:
            # Apply system state patterns
            system_state_patterns = patterns.get("system_state_patterns", {})
            
            for attribute, values in system_state_patterns.items():
                if values and hasattr(target_system, attribute):
                    # Use median value from successful experiences
                    optimal_value = np.median(values)
                    setattr(target_system, attribute, optimal_value)
            
            return True
        except Exception as e:
            return False
    
    def _capture_system_state(self, target_system) -> Dict[str, Any]:
        """Capture current state of target system."""
        state = {}
        
        # Capture basic attributes
        for attr in ["spectral_radius", "learning_rate", "sparsity"]:
            if hasattr(target_system, attr):
                state[attr] = getattr(target_system, attr)
        
        # Capture reservoir properties if available
        if hasattr(target_system, 'W') and target_system.W is not None:
            eigenvals = np.linalg.eigvals(target_system.W)
            state["actual_spectral_radius"] = float(np.max(np.abs(eigenvals)))
            state["matrix_size"] = target_system.W.shape[0]
            state["connection_density"] = float(np.count_nonzero(target_system.W) / target_system.W.size)
        
        return state


class HierarchicalOptimizer:
    """
    Hierarchical optimization across cognitive mesh layers.
    
    Implements multi-level optimization that coordinates improvements
    across different hierarchical levels of the cognitive system.
    """
    
    def __init__(self):
        self.optimization_levels = {}
        self.level_dependencies = {}
        self.optimization_history = deque(maxlen=200)
        self.coordination_strategies = {}
        
    def register_optimization_level(self, level_name: str, level_priority: int,
                                  optimization_function: Callable,
                                  dependencies: List[str] = None):
        """Register optimization level in hierarchy."""
        self.optimization_levels[level_name] = {
            "priority": level_priority,
            "optimization_function": optimization_function,
            "dependencies": dependencies or [],
            "last_optimization": 0.0,
            "performance_history": deque(maxlen=50),
            "optimization_count": 0
        }
        
        self.level_dependencies[level_name] = dependencies or []
    
    def hierarchical_optimize(self, target_systems: Dict[str, Any],
                            max_iterations: int = 10) -> Dict[str, Any]:
        """Perform hierarchical optimization across all levels."""
        optimization_results = {
            "iterations_completed": 0,
            "level_results": {},
            "overall_improvement": 0.0,
            "convergence_achieved": False,
            "coordination_efficiency": 0.0
        }
        
        initial_performance = self._evaluate_overall_performance(target_systems)
        
        for iteration in range(max_iterations):
            iteration_results = self._optimize_iteration(target_systems)
            optimization_results["level_results"][f"iteration_{iteration}"] = iteration_results
            
            # Check convergence
            current_performance = self._evaluate_overall_performance(target_systems)
            improvement = current_performance - initial_performance
            
            if improvement < 0.001:  # Convergence threshold
                optimization_results["convergence_achieved"] = True
                break
        
        optimization_results["iterations_completed"] = iteration + 1
        final_performance = self._evaluate_overall_performance(target_systems)
        optimization_results["overall_improvement"] = final_performance - initial_performance
        
        # Record optimization session
        self.optimization_history.append({
            "timestamp": time.time(),
            "results": optimization_results,
            "target_systems": list(target_systems.keys())
        })
        
        return optimization_results
    
    def _optimize_iteration(self, target_systems: Dict[str, Any]) -> Dict[str, Any]:
        """Perform single iteration of hierarchical optimization."""
        iteration_results = {}
        
        # Sort levels by priority
        sorted_levels = sorted(self.optimization_levels.items(), 
                             key=lambda x: x[1]["priority"])
        
        for level_name, level_info in sorted_levels:
            # Check dependencies
            if self._dependencies_satisfied(level_name):
                # Perform optimization at this level
                level_result = self._optimize_level(level_name, level_info, target_systems)
                iteration_results[level_name] = level_result
                
                # Update level info
                level_info["last_optimization"] = time.time()
                level_info["optimization_count"] += 1
                level_info["performance_history"].append(level_result.get("performance", 0.5))
        
        return iteration_results
    
    def _dependencies_satisfied(self, level_name: str) -> bool:
        """Check if dependencies for optimization level are satisfied."""
        dependencies = self.level_dependencies.get(level_name, [])
        
        for dependency in dependencies:
            if dependency in self.optimization_levels:
                dep_info = self.optimization_levels[dependency]
                # Check if dependency was optimized recently
                if dep_info["last_optimization"] == 0.0:
                    return False
        
        return True
    
    def _optimize_level(self, level_name: str, level_info: Dict[str, Any],
                       target_systems: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize specific level in hierarchy."""
        optimization_function = level_info["optimization_function"]
        
        try:
            # Call level-specific optimization function
            level_result = optimization_function(target_systems)
            
            # Add meta-information
            level_result.update({
                "level_name": level_name,
                "optimization_time": time.time(),
                "success": True
            })
            
            return level_result
        
        except Exception as e:
            return {
                "level_name": level_name,
                "error": str(e),
                "success": False,
                "performance": 0.0
            }
    
    def _evaluate_overall_performance(self, target_systems: Dict[str, Any]) -> float:
        """Evaluate overall system performance across all levels."""
        if not target_systems:
            return 0.0
        
        level_performances = []
        
        for level_name, level_info in self.optimization_levels.items():
            if level_info["performance_history"]:
                recent_performance = level_info["performance_history"][-1]
                level_performances.append(recent_performance)
        
        if level_performances:
            return np.mean(level_performances)
        else:
            return 0.5  # Default performance


class RecursiveImprovement:
    """
    Main recursive improvement system coordinating all improvement loops.
    
    Integrates embodied feedback, ensemble evolution, meta-learning, and 
    hierarchical optimization for comprehensive recursive self-improvement.
    """
    
    def __init__(self, improvement_interval: float = 30.0):
        self.improvement_interval = improvement_interval
        self.embodied_feedback = EmbodiedFeedback()
        self.ensemble_evolver = EnsembleEvolver()
        self.meta_learner = MetaLearner()
        self.hierarchical_optimizer = HierarchicalOptimizer()
        
        self.feedback_loops = {}
        self.improvement_cycles = deque(maxlen=100)
        self.system_evolution_history = deque(maxlen=200)
        self.is_active = False
        
    def register_feedback_loop(self, loop_id: str, source: str, target: str,
                             feedback_function: Callable, strength: float = 1.0):
        """Register recursive feedback loop."""
        feedback_loop = FeedbackLoop(
            loop_id=loop_id,
            source=source,
            target=target,
            feedback_function=feedback_function,
            strength=strength
        )
        
        self.feedback_loops[loop_id] = feedback_loop
    
    def start_recursive_improvement(self, target_systems: Dict[str, Any]):
        """Start recursive improvement process."""
        self.is_active = True
        
        improvement_cycle_id = len(self.improvement_cycles)
        start_time = time.time()
        
        # Measure initial performance
        initial_performance = self._measure_system_performance(target_systems)
        
        # Execute improvement cycle
        improvements_applied = []
        feedback_loops_triggered = []
        
        # Process embodied feedback
        feedback_results = self.embodied_feedback.process_embodied_feedback(
            target_systems.get("cognitive_system")
        )
        if feedback_results["system_adjustments"]:
            improvements_applied.extend(feedback_results["system_adjustments"])
            feedback_loops_triggered.append("embodied_feedback")
        
        # Evolve ensembles
        for ensemble_id in list(self.ensemble_evolver.ensembles.keys()):
            def ensemble_fitness(params):
                return self._evaluate_ensemble_fitness(params, target_systems)
            
            evolution_results = self.ensemble_evolver.evolve_ensemble(
                ensemble_id, ensemble_fitness, evolution_cycles=5
            )
            if evolution_results.get("final_performance", 0) > evolution_results.get("initial_performance", 0):
                improvements_applied.append({
                    "type": "ensemble_evolution",
                    "ensemble_id": ensemble_id,
                    "improvement": evolution_results["final_performance"] - evolution_results["initial_performance"]
                })
                feedback_loops_triggered.append("ensemble_evolution")
        
        # Apply meta-learning
        meta_context = {"task_type": "system_optimization", "timestamp": time.time()}
        meta_result = self.meta_learner.rapid_adapt(meta_context, target_systems.get("main_system"))
        if meta_result["success"]:
            improvements_applied.append({
                "type": "meta_learning",
                "adaptation_time": meta_result["adaptation_time"],
                "strategy": meta_result["strategy_used"]
            })
            feedback_loops_triggered.append("meta_learning")
        
        # Hierarchical optimization
        hierarchical_results = self.hierarchical_optimizer.hierarchical_optimize(
            target_systems, max_iterations=3
        )
        if hierarchical_results["overall_improvement"] > 0:
            improvements_applied.append({
                "type": "hierarchical_optimization",
                "improvement": hierarchical_results["overall_improvement"],
                "iterations": hierarchical_results["iterations_completed"]
            })
            feedback_loops_triggered.append("hierarchical_optimization")
        
        # Execute feedback loops
        for loop_id, feedback_loop in self.feedback_loops.items():
            if feedback_loop.active:
                try:
                    loop_result = feedback_loop.feedback_function(target_systems)
                    feedback_loop.history.append({
                        "timestamp": time.time(),
                        "result": loop_result,
                        "success": True
                    })
                    feedback_loops_triggered.append(loop_id)
                except Exception as e:
                    feedback_loop.history.append({
                        "timestamp": time.time(),
                        "error": str(e),
                        "success": False
                    })
        
        # Measure final performance
        final_performance = self._measure_system_performance(target_systems)
        
        # Create improvement cycle record
        improvement_cycle = ImprovementCycle(
            cycle_id=improvement_cycle_id,
            timestamp=start_time,
            performance_before=initial_performance,
            performance_after=final_performance,
            improvements_applied=improvements_applied,
            feedback_loops_triggered=feedback_loops_triggered,
            success=final_performance > initial_performance
        )
        
        self.improvement_cycles.append(improvement_cycle)
        
        # Record system evolution
        self.system_evolution_history.append({
            "timestamp": time.time(),
            "cycle_id": improvement_cycle_id,
            "performance_change": final_performance - initial_performance,
            "improvements_count": len(improvements_applied),
            "active_feedback_loops": len(feedback_loops_triggered)
        })
        
        return improvement_cycle
    
    def _measure_system_performance(self, target_systems: Dict[str, Any]) -> float:
        """Measure overall system performance."""
        performances = []
        
        # Basic system performance
        main_system = target_systems.get("main_system")
        if main_system:
            if hasattr(main_system, 'W') and main_system.W is not None:
                try:
                    eigenvals = np.linalg.eigvals(main_system.W)
                    spectral_radius = np.max(np.abs(eigenvals))
                    performance = 1.0 if spectral_radius < 1.0 else 0.5
                    performances.append(performance)
                except:
                    performances.append(0.5)
        
        # Cognitive system performance
        cognitive_system = target_systems.get("cognitive_system")
        if cognitive_system:
            # Simplified cognitive performance assessment
            performances.append(0.7)  # Placeholder
        
        # Default performance if no systems
        if not performances:
            performances = [0.5]
        
        return np.mean(performances)
    
    def _evaluate_ensemble_fitness(self, params: Dict[str, Any], 
                                 target_systems: Dict[str, Any]) -> float:
        """Evaluate fitness for ensemble evolution."""
        # Simplified fitness evaluation
        if "W" in params:
            try:
                eigenvals = np.linalg.eigvals(params["W"])
                spectral_radius = np.max(np.abs(eigenvals))
                
                # Fitness based on spectral properties
                if spectral_radius <= 1.0:
                    fitness = min(1.0, spectral_radius / 0.9)
                else:
                    fitness = max(0.1, 1.0 / spectral_radius)
                
                return fitness
            except:
                return 0.1
        
        return 0.5
    
    def get_improvement_summary(self, window: int = 20) -> Dict[str, Any]:
        """Get summary of recursive improvement performance."""
        if len(self.improvement_cycles) < 2:
            return {"insufficient_data": True}
        
        recent_cycles = list(self.improvement_cycles)[-window:]
        
        summary = {
            "total_cycles": len(self.improvement_cycles),
            "recent_cycles": len(recent_cycles),
            "success_rate": sum(1 for cycle in recent_cycles if cycle.success) / len(recent_cycles),
            "average_improvement": np.mean([cycle.performance_after - cycle.performance_before 
                                          for cycle in recent_cycles]),
            "total_improvements_applied": sum(len(cycle.improvements_applied) for cycle in recent_cycles),
            "most_effective_improvements": self._analyze_most_effective_improvements(recent_cycles),
            "feedback_loop_activity": self._analyze_feedback_loop_activity(recent_cycles),
            "system_evolution_trend": self._calculate_evolution_trend()
        }
        
        return summary
    
    def _analyze_most_effective_improvements(self, cycles: List[ImprovementCycle]) -> Dict[str, Any]:
        """Analyze which improvements are most effective."""
        improvement_effectiveness = defaultdict(list)
        
        for cycle in cycles:
            performance_gain = cycle.performance_after - cycle.performance_before
            
            for improvement in cycle.improvements_applied:
                improvement_type = improvement.get("type", "unknown")
                improvement_effectiveness[improvement_type].append(performance_gain)
        
        effectiveness_summary = {}
        for improvement_type, gains in improvement_effectiveness.items():
            effectiveness_summary[improvement_type] = {
                "average_gain": np.mean(gains),
                "total_applications": len(gains),
                "success_rate": sum(1 for gain in gains if gain > 0) / len(gains)
            }
        
        return effectiveness_summary
    
    def _analyze_feedback_loop_activity(self, cycles: List[ImprovementCycle]) -> Dict[str, Any]:
        """Analyze feedback loop activity and effectiveness."""
        loop_activity = defaultdict(int)
        loop_effectiveness = defaultdict(list)
        
        for cycle in cycles:
            performance_gain = cycle.performance_after - cycle.performance_before
            
            for loop_id in cycle.feedback_loops_triggered:
                loop_activity[loop_id] += 1
                loop_effectiveness[loop_id].append(performance_gain)
        
        activity_summary = {}
        for loop_id, count in loop_activity.items():
            effectiveness = loop_effectiveness[loop_id]
            activity_summary[loop_id] = {
                "activation_count": count,
                "average_effectiveness": np.mean(effectiveness),
                "activation_rate": count / len(cycles)
            }
        
        return activity_summary
    
    def _calculate_evolution_trend(self) -> float:
        """Calculate overall system evolution trend."""
        if len(self.system_evolution_history) < 3:
            return 0.0
        
        performance_changes = [entry["performance_change"] for entry in self.system_evolution_history]
        
        # Calculate trend
        x = np.arange(len(performance_changes))
        slope = np.polyfit(x, performance_changes, 1)[0]
        
        return float(slope)