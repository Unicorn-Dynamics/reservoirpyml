"""
Meta-Cognitive Pathways for Reservoir Self-Analysis
==================================================

This module implements feedback-driven self-analysis, reservoir introspection,
and meta-cognitive attention allocation for recursive system improvement.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from collections import deque, defaultdict

try:
    from ...nodes import Reservoir
    from ...model import Model
    from ..attention import ECANAttentionSystem, AttentionValue
    from ..distributed import MeshCoordinator
except ImportError:
    # Fallback for minimal dependencies
    pass


@dataclass
class PerformanceMetric:
    """Performance metric for reservoir introspection."""
    name: str
    value: float
    timestamp: float
    context: Dict[str, Any]


@dataclass
class MetaCognitiveState:
    """Current meta-cognitive state of the system."""
    introspection_depth: float
    self_awareness_level: float
    improvement_potential: float
    cognitive_load: float
    attention_allocation: Dict[str, float]


class ReservoirIntrospector:
    """
    Reservoir introspection mechanisms for performance monitoring.
    
    Provides deep analysis of reservoir dynamics, states, and performance
    characteristics for meta-cognitive self-improvement.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.state_patterns = {}
        self.performance_trends = {}
        
    def analyze_reservoir_dynamics(self, reservoir: 'Reservoir', 
                                 input_data: np.ndarray) -> Dict[str, float]:
        """Analyze reservoir dynamics and extract performance insights."""
        if not hasattr(reservoir, 'W'):
            return {"error": 1.0, "analysis_incomplete": True}
            
        metrics = {}
        
        # Spectral analysis
        try:
            eigenvals = np.linalg.eigvals(reservoir.W)
            spectral_radius = np.max(np.abs(eigenvals))
            metrics["spectral_radius"] = spectral_radius
            metrics["spectral_stability"] = 1.0 if spectral_radius < 1.0 else 0.0
        except:
            metrics["spectral_radius"] = 0.0
            metrics["spectral_stability"] = 0.0
        
        # State dynamics analysis
        if hasattr(reservoir, 'state') and reservoir.state is not None:
            state = reservoir.state
            metrics["state_variance"] = float(np.var(state))
            metrics["state_mean"] = float(np.mean(state))
            metrics["state_entropy"] = self._calculate_entropy(state)
            metrics["dynamical_richness"] = self._calculate_richness(state)
        else:
            metrics.update({
                "state_variance": 0.0, "state_mean": 0.0,
                "state_entropy": 0.0, "dynamical_richness": 0.0
            })
        
        # Memory and echo state analysis
        metrics["memory_capacity"] = self._estimate_memory_capacity(reservoir)
        metrics["echo_state_property"] = self._check_echo_state_property(reservoir)
        
        return metrics
    
    def monitor_cognitive_performance(self, cognitive_mesh) -> PerformanceMetric:
        """Monitor overall cognitive mesh performance."""
        timestamp = time.time()
        
        # Calculate performance based on mesh health and efficiency
        try:
            if hasattr(cognitive_mesh, 'get_mesh_status'):
                status = cognitive_mesh.get_mesh_status()
                performance = status.get('health', 0.5)
            else:
                performance = 0.5  # Default baseline
        except:
            performance = 0.5
        
        metric = PerformanceMetric(
            name="cognitive_mesh_performance",
            value=performance,
            timestamp=timestamp,
            context={"type": "mesh_monitoring"}
        )
        
        self.metrics_history.append(metric)
        return metric
    
    def detect_performance_patterns(self) -> Dict[str, Any]:
        """Detect patterns in performance history for meta-analysis."""
        if len(self.metrics_history) < 10:
            return {"insufficient_data": True}
        
        values = [m.value for m in self.metrics_history]
        
        patterns = {
            "trend": self._calculate_trend(values),
            "volatility": np.std(values),
            "mean_performance": np.mean(values),
            "recent_performance": np.mean(values[-10:]),
            "improvement_rate": self._calculate_improvement_rate(values),
            "stability_score": self._calculate_stability(values)
        }
        
        return patterns
    
    def _calculate_entropy(self, state: np.ndarray) -> float:
        """Calculate state entropy for complexity analysis."""
        try:
            # Normalize state to probabilities
            state_norm = np.abs(state) / (np.sum(np.abs(state)) + 1e-10)
            # Calculate entropy
            entropy = -np.sum(state_norm * np.log(state_norm + 1e-10))
            return float(entropy)
        except:
            return 0.0
    
    def _calculate_richness(self, state: np.ndarray) -> float:
        """Calculate dynamical richness of reservoir states."""
        try:
            # Measure the effective dimensionality
            _, s, _ = np.linalg.svd(state.reshape(-1, 1))
            effective_rank = np.sum(s > 0.01 * s[0]) / len(s)
            return float(effective_rank)
        except:
            return 0.0
    
    def _estimate_memory_capacity(self, reservoir) -> float:
        """Estimate memory capacity of the reservoir."""
        # Simplified memory capacity estimation
        try:
            if hasattr(reservoir, 'W'):
                eigenvals = np.linalg.eigvals(reservoir.W)
                # Memory capacity related to spectral properties
                capacity = np.sum(np.abs(eigenvals) > 0.1) / len(eigenvals)
                return float(capacity)
        except:
            pass
        return 0.5  # Default estimate
    
    def _check_echo_state_property(self, reservoir) -> float:
        """Check if reservoir satisfies echo state property."""
        try:
            if hasattr(reservoir, 'W'):
                spectral_radius = np.max(np.abs(np.linalg.eigvals(reservoir.W)))
                return 1.0 if spectral_radius < 1.0 else 0.0
        except:
            pass
        return 0.5  # Uncertain
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in performance values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """Calculate rate of improvement over time."""
        if len(values) < 10:
            return 0.0
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        if first_half > 0:
            improvement = (second_half - first_half) / first_half
        else:
            improvement = 0.0
        
        return float(improvement)
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability score based on variance."""
        if len(values) < 2:
            return 1.0
        
        # Stability is inverse of normalized variance
        variance = np.var(values)
        mean_val = np.mean(values)
        if mean_val > 0:
            normalized_variance = variance / (mean_val ** 2)
            stability = 1.0 / (1.0 + normalized_variance)
        else:
            stability = 0.5
        
        return float(stability)


class MetaAttentionAllocator:
    """
    Meta-cognitive attention allocation for self-improvement.
    
    Manages attention resources for meta-cognitive processes including
    self-analysis, performance monitoring, and improvement planning.
    """
    
    def __init__(self, total_attention: float = 1.0):
        self.total_attention = total_attention
        self.attention_allocation = {
            "self_analysis": 0.3,
            "performance_monitoring": 0.25,
            "improvement_planning": 0.2,
            "adaptation": 0.15,
            "exploration": 0.1
        }
        self.attention_history = deque(maxlen=50)
        
    def allocate_meta_attention(self, current_state: MetaCognitiveState,
                              performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Dynamically allocate attention for meta-cognitive processes."""
        
        # Analyze current needs
        needs = self._assess_attention_needs(current_state, performance_metrics)
        
        # Adjust allocation based on needs
        new_allocation = self._optimize_allocation(needs)
        
        # Update allocation with temporal smoothing
        alpha = 0.3  # Smoothing factor
        for key in self.attention_allocation:
            self.attention_allocation[key] = (
                alpha * new_allocation.get(key, 0.0) + 
                (1 - alpha) * self.attention_allocation[key]
            )
        
        # Normalize to ensure sum equals total attention
        total = sum(self.attention_allocation.values())
        if total > 0:
            for key in self.attention_allocation:
                self.attention_allocation[key] *= self.total_attention / total
        
        # Record allocation history
        self.attention_history.append(self.attention_allocation.copy())
        
        return self.attention_allocation.copy()
    
    def _assess_attention_needs(self, state: MetaCognitiveState, 
                              metrics: Dict[str, float]) -> Dict[str, float]:
        """Assess attention needs based on current state and performance."""
        needs = {}
        
        # High cognitive load requires more self-analysis
        needs["self_analysis"] = min(1.0, state.cognitive_load * 0.5)
        
        # Poor performance requires more monitoring
        performance = metrics.get("mean_performance", 0.5)
        needs["performance_monitoring"] = max(0.1, 1.0 - performance)
        
        # High improvement potential requires more planning
        needs["improvement_planning"] = state.improvement_potential * 0.8
        
        # Unstable performance requires more adaptation
        stability = metrics.get("stability_score", 0.5)
        needs["adaptation"] = max(0.1, 1.0 - stability)
        
        # Low awareness requires more exploration
        needs["exploration"] = max(0.05, 1.0 - state.self_awareness_level)
        
        return needs
    
    def _optimize_allocation(self, needs: Dict[str, float]) -> Dict[str, float]:
        """Optimize attention allocation based on assessed needs."""
        # Start with needs as base allocation
        allocation = needs.copy()
        
        # Apply constraints and balancing
        total_needs = sum(allocation.values())
        if total_needs > 0:
            # Normalize to available attention
            for key in allocation:
                allocation[key] /= total_needs
                allocation[key] *= self.total_attention
        
        # Ensure minimum allocations
        minimums = {
            "self_analysis": 0.1,
            "performance_monitoring": 0.1,
            "improvement_planning": 0.05,
            "adaptation": 0.05,
            "exploration": 0.02
        }
        
        for key, minimum in minimums.items():
            allocation[key] = max(allocation.get(key, 0), minimum)
        
        return allocation


class SelfAnalysisModule:
    """
    Feedback-driven self-analysis module.
    
    Provides comprehensive self-analysis capabilities including performance
    reflection, weakness identification, and improvement opportunity discovery.
    """
    
    def __init__(self):
        self.analysis_history = deque(maxlen=100)
        self.identified_weaknesses = {}
        self.improvement_opportunities = {}
        self.reflection_depth = 0.5
        
    def perform_self_analysis(self, introspection_data: Dict[str, float],
                            performance_history: List[PerformanceMetric]) -> Dict[str, Any]:
        """Perform comprehensive self-analysis."""
        
        analysis_result = {
            "timestamp": time.time(),
            "performance_assessment": self._assess_performance(performance_history),
            "weakness_identification": self._identify_weaknesses(introspection_data),
            "opportunity_discovery": self._discover_opportunities(introspection_data),
            "self_reflection": self._perform_reflection(introspection_data),
            "meta_insights": self._generate_meta_insights()
        }
        
        self.analysis_history.append(analysis_result)
        return analysis_result
    
    def _assess_performance(self, history: List[PerformanceMetric]) -> Dict[str, float]:
        """Assess current performance levels and trends."""
        if not history:
            return {"insufficient_data": True}
        
        values = [m.value for m in history]
        
        assessment = {
            "current_performance": values[-1] if values else 0.0,
            "average_performance": np.mean(values),
            "performance_trend": self._calculate_trend(values),
            "performance_stability": 1.0 - np.std(values),
            "recent_improvement": (np.mean(values[-10:]) - np.mean(values[:-10])) if len(values) > 20 else 0.0
        }
        
        return assessment
    
    def _identify_weaknesses(self, data: Dict[str, float]) -> Dict[str, float]:
        """Identify system weaknesses based on introspection data."""
        weaknesses = {}
        
        # Spectral radius issues
        spectral_radius = data.get("spectral_radius", 1.0)
        if spectral_radius > 1.0:
            weaknesses["spectral_instability"] = spectral_radius - 1.0
        
        # Low dynamical richness
        richness = data.get("dynamical_richness", 0.5)
        if richness < 0.3:
            weaknesses["low_complexity"] = 0.3 - richness
        
        # Poor memory capacity
        memory = data.get("memory_capacity", 0.5)
        if memory < 0.4:
            weaknesses["insufficient_memory"] = 0.4 - memory
        
        # High variance (instability)
        variance = data.get("state_variance", 0.0)
        if variance > 1.0:
            weaknesses["high_variance"] = variance - 1.0
        
        # Update tracked weaknesses
        for weakness, severity in weaknesses.items():
            if weakness not in self.identified_weaknesses:
                self.identified_weaknesses[weakness] = []
            self.identified_weaknesses[weakness].append({
                "severity": severity,
                "timestamp": time.time()
            })
        
        return weaknesses
    
    def _discover_opportunities(self, data: Dict[str, float]) -> Dict[str, float]:
        """Discover improvement opportunities."""
        opportunities = {}
        
        # Optimization opportunities
        spectral_radius = data.get("spectral_radius", 1.0)
        if 0.8 <= spectral_radius <= 0.95:
            opportunities["spectral_optimization"] = 0.95 - spectral_radius
        
        # Complexity enhancement
        richness = data.get("dynamical_richness", 0.5)
        if richness > 0.3:
            opportunities["complexity_enhancement"] = min(0.3, 1.0 - richness)
        
        # Memory expansion
        memory = data.get("memory_capacity", 0.5)
        if memory > 0.4:
            opportunities["memory_expansion"] = min(0.3, 1.0 - memory)
        
        # Stability improvement
        stability = data.get("spectral_stability", 0.5)
        if stability > 0.7:
            opportunities["stability_enhancement"] = min(0.2, 1.0 - stability)
        
        return opportunities
    
    def _perform_reflection(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Perform meta-cognitive reflection on current state."""
        reflection = {
            "self_awareness_level": self._calculate_self_awareness(data),
            "cognitive_coherence": self._assess_coherence(data),
            "adaptation_readiness": self._assess_adaptation_readiness(),
            "learning_efficiency": self._assess_learning_efficiency(),
            "reflection_depth": self.reflection_depth
        }
        
        # Adjust reflection depth based on insights
        if reflection["self_awareness_level"] > 0.8:
            self.reflection_depth = min(1.0, self.reflection_depth + 0.1)
        elif reflection["self_awareness_level"] < 0.3:
            self.reflection_depth = max(0.1, self.reflection_depth - 0.05)
        
        return reflection
    
    def _generate_meta_insights(self) -> Dict[str, Any]:
        """Generate meta-level insights about the analysis process itself."""
        insights = {
            "analysis_quality": self._assess_analysis_quality(),
            "insight_generation_rate": len(self.analysis_history) / max(1, len(self.analysis_history)),
            "meta_learning_progress": self._assess_meta_learning(),
            "recursive_improvement_potential": self._assess_recursive_potential()
        }
        
        return insights
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def _calculate_self_awareness(self, data: Dict[str, float]) -> float:
        """Calculate self-awareness level based on introspection data."""
        # Self-awareness based on data richness and coherence
        data_richness = len([v for v in data.values() if v != 0.0]) / max(1, len(data))
        data_coherence = 1.0 - np.std(list(data.values()))
        return (data_richness + data_coherence) / 2.0
    
    def _assess_coherence(self, data: Dict[str, float]) -> float:
        """Assess cognitive coherence from introspection data."""
        values = list(data.values())
        if not values:
            return 0.5
        
        # Coherence based on value consistency and relationships
        variance = np.var(values)
        coherence = 1.0 / (1.0 + variance)
        return float(coherence)
    
    def _assess_adaptation_readiness(self) -> float:
        """Assess readiness for adaptation based on analysis history."""
        if len(self.analysis_history) < 3:
            return 0.5
        
        # Readiness based on trend in weakness identification
        recent_analyses = list(self.analysis_history)[-3:]
        weakness_trends = []
        
        for analysis in recent_analyses:
            weaknesses = analysis.get("weakness_identification", {})
            weakness_trends.append(len(weaknesses))
        
        if len(weakness_trends) >= 2:
            trend = weakness_trends[-1] - weakness_trends[0]
            readiness = max(0.0, min(1.0, 0.5 - trend * 0.1))
        else:
            readiness = 0.5
        
        return readiness
    
    def _assess_learning_efficiency(self) -> float:
        """Assess meta-learning efficiency."""
        if len(self.analysis_history) < 5:
            return 0.5
        
        # Efficiency based on improvement in analysis quality over time
        recent_quality = self._assess_analysis_quality()
        historical_quality = 0.5  # Baseline
        
        efficiency = min(1.0, recent_quality / max(0.1, historical_quality))
        return efficiency
    
    def _assess_analysis_quality(self) -> float:
        """Assess quality of recent analyses."""
        if not self.analysis_history:
            return 0.5
        
        recent_analysis = self.analysis_history[-1]
        
        # Quality based on completeness and insight depth
        components = ["performance_assessment", "weakness_identification", 
                     "opportunity_discovery", "self_reflection"]
        
        completeness = sum(1 for comp in components 
                          if comp in recent_analysis and recent_analysis[comp]) / len(components)
        
        # Insight depth based on number of identified issues/opportunities
        weaknesses = recent_analysis.get("weakness_identification", {})
        opportunities = recent_analysis.get("opportunity_discovery", {})
        insight_depth = min(1.0, (len(weaknesses) + len(opportunities)) / 10.0)
        
        quality = (completeness + insight_depth) / 2.0
        return quality
    
    def _assess_meta_learning(self) -> float:
        """Assess meta-learning progress."""
        if len(self.analysis_history) < 5:
            return 0.1
        
        # Meta-learning based on improvement in analysis patterns
        recent_analyses = list(self.analysis_history)[-5:]
        quality_trend = []
        
        for analysis in recent_analyses:
            weaknesses = len(analysis.get("weakness_identification", {}))
            opportunities = len(analysis.get("opportunity_discovery", {}))
            quality_trend.append(weaknesses + opportunities)
        
        if len(quality_trend) >= 2:
            progress = (quality_trend[-1] - quality_trend[0]) / max(1, quality_trend[0])
            return max(0.0, min(1.0, progress))
        
        return 0.1
    
    def _assess_recursive_potential(self) -> float:
        """Assess potential for recursive improvement."""
        # Based on identified opportunities and meta-insights
        total_opportunities = sum(len(analysis.get("opportunity_discovery", {})) 
                                for analysis in self.analysis_history)
        
        if len(self.analysis_history) > 0:
            avg_opportunities = total_opportunities / len(self.analysis_history)
            potential = min(1.0, avg_opportunities / 5.0)  # Normalize to max 5 opportunities
        else:
            potential = 0.1
        
        return potential


class MetaCognitiveSystem:
    """
    Main meta-cognitive system coordinating all meta-cognitive processes.
    
    Integrates reservoir introspection, attention allocation, and self-analysis
    for comprehensive recursive self-improvement capabilities.
    """
    
    def __init__(self, window_size: int = 100):
        self.introspector = ReservoirIntrospector(window_size)
        self.attention_allocator = MetaAttentionAllocator()
        self.self_analyzer = SelfAnalysisModule()
        self.current_state = MetaCognitiveState(
            introspection_depth=0.5,
            self_awareness_level=0.3,
            improvement_potential=0.7,
            cognitive_load=0.4,
            attention_allocation={}
        )
        self.feedback_loops = []
        
    def update_meta_cognitive_state(self, reservoir=None, cognitive_mesh=None) -> MetaCognitiveState:
        """Update and return current meta-cognitive state."""
        
        # Gather introspection data
        introspection_data = {}
        if reservoir is not None:
            introspection_data = self.introspector.analyze_reservoir_dynamics(
                reservoir, np.random.randn(10, 5)  # Sample input
            )
        
        # Monitor cognitive performance
        performance_metric = None
        if cognitive_mesh is not None:
            performance_metric = self.introspector.monitor_cognitive_performance(cognitive_mesh)
        
        # Detect performance patterns
        patterns = self.introspector.detect_performance_patterns()
        
        # Perform self-analysis
        analysis_result = self.self_analyzer.perform_self_analysis(
            introspection_data, list(self.introspector.metrics_history)
        )
        
        # Update meta-cognitive state
        self.current_state.self_awareness_level = analysis_result["self_reflection"]["self_awareness_level"]
        self.current_state.improvement_potential = analysis_result["meta_insights"]["recursive_improvement_potential"]
        self.current_state.cognitive_load = min(1.0, len(introspection_data) / 10.0)
        self.current_state.introspection_depth = analysis_result["self_reflection"]["reflection_depth"]
        
        # Allocate meta-attention
        performance_metrics = patterns if patterns.get("insufficient_data") != True else {"mean_performance": 0.5}
        attention_allocation = self.attention_allocator.allocate_meta_attention(
            self.current_state, performance_metrics
        )
        self.current_state.attention_allocation = attention_allocation
        
        return self.current_state
    
    def generate_feedback_loops(self) -> List[Dict[str, Any]]:
        """Generate feedback loops for recursive improvement."""
        loops = []
        
        # Performance feedback loop
        performance_patterns = self.introspector.detect_performance_patterns()
        if performance_patterns.get("insufficient_data") != True:
            loops.append({
                "type": "performance_feedback",
                "input": performance_patterns,
                "target": "optimization",
                "strength": abs(performance_patterns.get("trend", 0.0)),
                "recommendations": self._generate_performance_recommendations(performance_patterns)
            })
        
        # Self-analysis feedback loop
        if self.self_analyzer.analysis_history:
            recent_analysis = self.self_analyzer.analysis_history[-1]
            loops.append({
                "type": "self_analysis_feedback", 
                "input": recent_analysis,
                "target": "meta_cognition",
                "strength": recent_analysis["meta_insights"]["analysis_quality"],
                "recommendations": self._generate_analysis_recommendations(recent_analysis)
            })
        
        # Attention feedback loop
        if len(self.attention_allocator.attention_history) > 5:
            attention_data = list(self.attention_allocator.attention_history)
            loops.append({
                "type": "attention_feedback",
                "input": attention_data,
                "target": "attention_allocation", 
                "strength": self._calculate_attention_effectiveness(attention_data),
                "recommendations": self._generate_attention_recommendations(attention_data)
            })
        
        self.feedback_loops = loops
        return loops
    
    def _generate_performance_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance patterns."""
        recommendations = []
        
        trend = patterns.get("trend", 0.0)
        if trend < -0.01:
            recommendations.append("Performance declining - increase adaptation attention")
        elif trend > 0.01:
            recommendations.append("Performance improving - maintain current strategy")
        
        volatility = patterns.get("volatility", 0.0)
        if volatility > 0.2:
            recommendations.append("High volatility detected - increase stability focus")
        
        stability = patterns.get("stability_score", 0.5)
        if stability < 0.3:
            recommendations.append("Low stability - consider topology optimization")
        
        return recommendations
    
    def _generate_analysis_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on self-analysis."""
        recommendations = []
        
        weaknesses = analysis.get("weakness_identification", {})
        if "spectral_instability" in weaknesses:
            recommendations.append("Spectral instability detected - evolve spectral radius")
        if "low_complexity" in weaknesses:
            recommendations.append("Low complexity - enhance reservoir richness")
        if "insufficient_memory" in weaknesses:
            recommendations.append("Memory deficiency - expand memory capacity")
        
        opportunities = analysis.get("opportunity_discovery", {})
        if "spectral_optimization" in opportunities:
            recommendations.append("Spectral optimization opportunity available")
        if "complexity_enhancement" in opportunities:
            recommendations.append("Complexity enhancement possible")
        
        return recommendations
    
    def _generate_attention_recommendations(self, attention_data: List[Dict[str, float]]) -> List[str]:
        """Generate recommendations for attention allocation."""
        recommendations = []
        
        # Analyze attention trends
        if len(attention_data) >= 3:
            recent = attention_data[-3:]
            for category in recent[0].keys():
                values = [data[category] for data in recent]
                trend = values[-1] - values[0]
                
                if trend > 0.1:
                    recommendations.append(f"Increasing {category} attention - monitor effectiveness")
                elif trend < -0.1:
                    recommendations.append(f"Decreasing {category} attention - ensure adequacy")
        
        return recommendations
    
    def _calculate_attention_effectiveness(self, attention_data: List[Dict[str, float]]) -> float:
        """Calculate effectiveness of attention allocation."""
        if len(attention_data) < 2:
            return 0.5
        
        # Effectiveness based on stability and appropriateness of allocation
        categories = list(attention_data[0].keys())
        variances = []
        
        for category in categories:
            values = [data[category] for data in attention_data]
            variances.append(np.var(values))
        
        # Lower variance indicates more stable (effective) allocation
        avg_variance = np.mean(variances)
        effectiveness = 1.0 / (1.0 + avg_variance)
        
        return float(effectiveness)