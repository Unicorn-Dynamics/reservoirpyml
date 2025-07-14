"""
Performance Benchmarking and Adaptive Optimization
=================================================

This module implements continuous benchmarking, self-tuning systems,
and multi-objective optimization for cognitive trade-offs.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import time
from collections import deque, defaultdict
import threading
from abc import ABC, abstractmethod

try:
    from ...nodes import Reservoir
    from ...model import Model
    from ..attention import ECANAttentionSystem
    from ..distributed import MeshCoordinator
except ImportError:
    # Fallback for minimal dependencies
    pass


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    metric_name: str
    value: float
    timestamp: float
    context: Dict[str, Any]
    baseline: Optional[float] = None
    improvement: Optional[float] = None


@dataclass
class OptimizationObjective:
    """Multi-objective optimization objective."""
    name: str
    weight: float
    minimize: bool
    target_value: Optional[float] = None
    current_value: float = 0.0


class PerformanceBenchmarker:
    """
    Main performance benchmarking system.
    
    Provides comprehensive benchmarking capabilities for reservoir performance,
    cognitive metrics, and system efficiency measurements.
    """
    
    def __init__(self, baseline_window: int = 50):
        self.baseline_window = baseline_window
        self.benchmark_history = deque(maxlen=1000)
        self.baselines = {}
        self.benchmark_tasks = {}
        self.real_time_metrics = {}
        
    def register_benchmark_task(self, task_name: str, task_function: Callable,
                               task_data: Any = None, frequency: float = 1.0):
        """Register a benchmark task for regular execution."""
        self.benchmark_tasks[task_name] = {
            "function": task_function,
            "data": task_data,
            "frequency": frequency,
            "last_run": 0.0,
            "results": deque(maxlen=100)
        }
    
    def run_benchmark_suite(self, reservoir, cognitive_mesh=None) -> Dict[str, BenchmarkResult]:
        """Run comprehensive benchmark suite."""
        results = {}
        
        # Core reservoir benchmarks
        results.update(self._benchmark_reservoir_performance(reservoir))
        
        # Cognitive benchmarks
        if cognitive_mesh is not None:
            results.update(self._benchmark_cognitive_performance(cognitive_mesh))
        
        # System benchmarks
        results.update(self._benchmark_system_metrics())
        
        # Store results
        for result in results.values():
            self.benchmark_history.append(result)
            self._update_baseline(result)
        
        return results
    
    def _benchmark_reservoir_performance(self, reservoir) -> Dict[str, BenchmarkResult]:
        """Benchmark core reservoir performance metrics."""
        results = {}
        timestamp = time.time()
        
        if not hasattr(reservoir, 'W') or reservoir.W is None:
            return {"error": BenchmarkResult("reservoir_error", 1.0, timestamp, {})}
        
        # Spectral properties
        try:
            eigenvals = np.linalg.eigvals(reservoir.W)
            spectral_radius = np.max(np.abs(eigenvals))
            
            results["spectral_radius"] = BenchmarkResult(
                "spectral_radius", spectral_radius, timestamp,
                {"optimal_range": (0.8, 1.0)}
            )
            
            results["spectral_stability"] = BenchmarkResult(
                "spectral_stability", 1.0 if spectral_radius < 1.0 else 0.0,
                timestamp, {"threshold": 1.0}
            )
        except:
            results["spectral_error"] = BenchmarkResult(
                "spectral_error", 1.0, timestamp, {}
            )
        
        # State dynamics
        if hasattr(reservoir, 'state') and reservoir.state is not None:
            state = reservoir.state
            
            results["state_variance"] = BenchmarkResult(
                "state_variance", float(np.var(state)), timestamp,
                {"optimal_range": (0.1, 1.0)}
            )
            
            results["state_entropy"] = BenchmarkResult(
                "state_entropy", self._calculate_entropy(state), timestamp,
                {"higher_is_better": True}
            )
            
            results["dynamical_richness"] = BenchmarkResult(
                "dynamical_richness", self._calculate_richness(state), timestamp,
                {"optimal_range": (0.3, 0.8)}
            )
        
        # Memory capacity estimation
        results["memory_capacity"] = BenchmarkResult(
            "memory_capacity", self._estimate_memory_capacity(reservoir),
            timestamp, {"higher_is_better": True}
        )
        
        # Echo state property
        results["echo_state_property"] = BenchmarkResult(
            "echo_state_property", self._check_echo_state_property(reservoir),
            timestamp, {"target": 1.0}
        )
        
        return results
    
    def _benchmark_cognitive_performance(self, cognitive_mesh) -> Dict[str, BenchmarkResult]:
        """Benchmark cognitive mesh performance."""
        results = {}
        timestamp = time.time()
        
        try:
            # Mesh health and efficiency
            if hasattr(cognitive_mesh, 'get_mesh_status'):
                status = cognitive_mesh.get_mesh_status()
                
                results["mesh_health"] = BenchmarkResult(
                    "mesh_health", status.get("health", 0.5), timestamp,
                    {"target": 1.0}
                )
                
                results["mesh_efficiency"] = BenchmarkResult(
                    "mesh_efficiency", status.get("efficiency", 0.5), timestamp,
                    {"target": 1.0}
                )
            
            # Agent coordination metrics
            if hasattr(cognitive_mesh, 'orchestrator'):
                orchestrator = cognitive_mesh.orchestrator
                
                if hasattr(orchestrator, 'active_agents'):
                    num_agents = len(orchestrator.active_agents)
                    results["active_agents"] = BenchmarkResult(
                        "active_agents", float(num_agents), timestamp,
                        {"optimal_range": (5, 20)}
                    )
                
                # Task distribution efficiency
                results["task_efficiency"] = BenchmarkResult(
                    "task_efficiency", self._calculate_task_efficiency(orchestrator),
                    timestamp, {"target": 1.0}
                )
            
            # Attention allocation metrics
            if hasattr(cognitive_mesh, 'attention_system'):
                attention_metrics = self._benchmark_attention_system(cognitive_mesh.attention_system)
                results.update(attention_metrics)
        
        except Exception as e:
            results["cognitive_error"] = BenchmarkResult(
                "cognitive_error", 1.0, timestamp, {"error": str(e)}
            )
        
        return results
    
    def _benchmark_system_metrics(self) -> Dict[str, BenchmarkResult]:
        """Benchmark system-level performance metrics."""
        results = {}
        timestamp = time.time()
        
        # Memory usage (simplified)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            results["memory_usage"] = BenchmarkResult(
                "memory_usage", memory_mb, timestamp,
                {"unit": "MB", "lower_is_better": True}
            )
        except:
            results["memory_usage"] = BenchmarkResult(
                "memory_usage", 100.0, timestamp,  # Default estimate
                {"unit": "MB", "estimated": True}
            )
        
        # Processing latency
        start_time = time.perf_counter()
        # Simulate processing
        dummy_data = np.random.randn(100, 10)
        _ = np.dot(dummy_data, dummy_data.T)
        processing_time = (time.perf_counter() - start_time) * 1000  # ms
        
        results["processing_latency"] = BenchmarkResult(
            "processing_latency", processing_time, timestamp,
            {"unit": "ms", "target": "<10ms"}
        )
        
        # System throughput estimate
        throughput = 1000.0 / max(processing_time, 0.001)  # operations per second
        results["system_throughput"] = BenchmarkResult(
            "system_throughput", throughput, timestamp,
            {"unit": "ops/sec", "higher_is_better": True}
        )
        
        return results
    
    def _benchmark_attention_system(self, attention_system) -> Dict[str, BenchmarkResult]:
        """Benchmark attention system performance."""
        results = {}
        timestamp = time.time()
        
        try:
            # Attention allocation efficiency
            if hasattr(attention_system, 'attention_allocation'):
                allocation = attention_system.attention_allocation
                
                # Calculate allocation entropy (diversity)
                values = list(allocation.values())
                if values:
                    entropy = self._calculate_entropy(np.array(values))
                    results["attention_entropy"] = BenchmarkResult(
                        "attention_entropy", entropy, timestamp,
                        {"optimal_range": (0.5, 2.0)}
                    )
                
                # Allocation balance
                balance = 1.0 - np.std(values) if values else 0.5
                results["attention_balance"] = BenchmarkResult(
                    "attention_balance", balance, timestamp,
                    {"target": 1.0}
                )
            
            # Resource utilization
            if hasattr(attention_system, 'resource_allocator'):
                resource_efficiency = self._calculate_resource_efficiency(attention_system.resource_allocator)
                results["resource_efficiency"] = BenchmarkResult(
                    "resource_efficiency", resource_efficiency, timestamp,
                    {"target": 1.0}
                )
        
        except Exception as e:
            results["attention_error"] = BenchmarkResult(
                "attention_error", 1.0, timestamp, {"error": str(e)}
            )
        
        return results
    
    def _update_baseline(self, result: BenchmarkResult):
        """Update baseline values for performance comparison."""
        metric_name = result.metric_name
        
        if metric_name not in self.baselines:
            self.baselines[metric_name] = deque(maxlen=self.baseline_window)
        
        self.baselines[metric_name].append(result.value)
        
        # Calculate baseline and improvement
        if len(self.baselines[metric_name]) >= 5:
            baseline = np.mean(list(self.baselines[metric_name])[:len(self.baselines[metric_name])//2])
            result.baseline = baseline
            result.improvement = result.value - baseline
    
    def get_performance_trends(self, metric_name: str, window: int = 50) -> Dict[str, float]:
        """Get performance trends for a specific metric."""
        relevant_results = [r for r in self.benchmark_history 
                          if r.metric_name == metric_name][-window:]
        
        if len(relevant_results) < 3:
            return {"insufficient_data": True}
        
        values = [r.value for r in relevant_results]
        timestamps = [r.timestamp for r in relevant_results]
        
        # Calculate trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0] if len(values) >= 2 else 0.0
        
        return {
            "trend": slope,
            "mean": np.mean(values),
            "std": np.std(values),
            "latest": values[-1],
            "improvement_rate": (values[-1] - values[0]) / max(abs(values[0]), 0.001),
            "stability": 1.0 - (np.std(values) / max(np.mean(values), 0.001))
        }
    
    # Helper methods
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data."""
        try:
            # Normalize to probabilities
            data_norm = np.abs(data) / (np.sum(np.abs(data)) + 1e-10)
            entropy = -np.sum(data_norm * np.log(data_norm + 1e-10))
            return float(entropy)
        except:
            return 0.0
    
    def _calculate_richness(self, state: np.ndarray) -> float:
        """Calculate dynamical richness."""
        try:
            _, s, _ = np.linalg.svd(state.reshape(-1, 1))
            effective_rank = np.sum(s > 0.01 * s[0]) / len(s)
            return float(effective_rank)
        except:
            return 0.0
    
    def _estimate_memory_capacity(self, reservoir) -> float:
        """Estimate reservoir memory capacity."""
        try:
            if hasattr(reservoir, 'W'):
                eigenvals = np.linalg.eigvals(reservoir.W)
                capacity = np.sum(np.abs(eigenvals) > 0.1) / len(eigenvals)
                return float(capacity)
        except:
            pass
        return 0.5
    
    def _check_echo_state_property(self, reservoir) -> float:
        """Check echo state property."""
        try:
            if hasattr(reservoir, 'W'):
                spectral_radius = np.max(np.abs(np.linalg.eigvals(reservoir.W)))
                return 1.0 if spectral_radius < 1.0 else 0.0
        except:
            pass
        return 0.5
    
    def _calculate_task_efficiency(self, orchestrator) -> float:
        """Calculate task distribution efficiency."""
        # Simplified efficiency calculation
        try:
            if hasattr(orchestrator, 'task_queue') and hasattr(orchestrator, 'completed_tasks'):
                pending = len(orchestrator.task_queue) if orchestrator.task_queue else 0
                completed = len(orchestrator.completed_tasks) if orchestrator.completed_tasks else 1
                efficiency = completed / max(completed + pending, 1)
                return float(efficiency)
        except:
            pass
        return 0.8  # Default reasonable efficiency
    
    def _calculate_resource_efficiency(self, resource_allocator) -> float:
        """Calculate resource allocation efficiency."""
        try:
            if hasattr(resource_allocator, 'total_resources') and hasattr(resource_allocator, 'allocated_resources'):
                total = resource_allocator.total_resources
                allocated = resource_allocator.allocated_resources
                efficiency = allocated / max(total, 1.0)
                return float(min(1.0, efficiency))
        except:
            pass
        return 0.7  # Default efficiency


class ContinuousBenchmarker:
    """
    Real-time continuous benchmarking system.
    
    Provides continuous monitoring and benchmarking of system performance
    with automatic baseline updates and trend detection.
    """
    
    def __init__(self, benchmark_interval: float = 5.0):
        self.benchmark_interval = benchmark_interval
        self.benchmarker = PerformanceBenchmarker()
        self.is_running = False
        self.benchmark_thread = None
        self.continuous_results = deque(maxlen=200)
        self.alert_thresholds = {}
        
    def start_continuous_benchmarking(self, reservoir, cognitive_mesh=None):
        """Start continuous benchmarking in background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.benchmark_thread = threading.Thread(
            target=self._continuous_benchmark_loop,
            args=(reservoir, cognitive_mesh),
            daemon=True
        )
        self.benchmark_thread.start()
    
    def stop_continuous_benchmarking(self):
        """Stop continuous benchmarking."""
        self.is_running = False
        if self.benchmark_thread:
            self.benchmark_thread.join(timeout=1.0)
    
    def _continuous_benchmark_loop(self, reservoir, cognitive_mesh):
        """Main continuous benchmarking loop."""
        last_benchmark_time = 0.0
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_benchmark_time >= self.benchmark_interval:
                try:
                    # Run benchmark suite
                    results = self.benchmarker.run_benchmark_suite(reservoir, cognitive_mesh)
                    
                    # Store results with timestamp
                    result_summary = {
                        "timestamp": current_time,
                        "results": results,
                        "summary": self._summarize_results(results)
                    }
                    
                    self.continuous_results.append(result_summary)
                    
                    # Check for alerts
                    self._check_performance_alerts(results)
                    
                    last_benchmark_time = current_time
                    
                except Exception as e:
                    # Log error but continue
                    error_result = {
                        "timestamp": current_time,
                        "error": str(e),
                        "results": {}
                    }
                    self.continuous_results.append(error_result)
            
            # Sleep briefly to avoid excessive CPU usage
            time.sleep(0.1)
    
    def _summarize_results(self, results: Dict[str, BenchmarkResult]) -> Dict[str, float]:
        """Summarize benchmark results."""
        summary = {}
        
        for name, result in results.items():
            summary[name] = result.value
            
            if result.improvement is not None:
                summary[f"{name}_improvement"] = result.improvement
        
        # Calculate overall performance score
        key_metrics = ["spectral_stability", "echo_state_property", "mesh_health", "task_efficiency"]
        available_metrics = [summary.get(metric, 0.5) for metric in key_metrics if metric in summary]
        
        if available_metrics:
            summary["overall_performance"] = np.mean(available_metrics)
        else:
            summary["overall_performance"] = 0.5
        
        return summary
    
    def _check_performance_alerts(self, results: Dict[str, BenchmarkResult]):
        """Check for performance alerts and degradation."""
        alerts = []
        
        for name, result in results.items():
            # Check against thresholds
            if name in self.alert_thresholds:
                threshold = self.alert_thresholds[name]
                if result.value < threshold:
                    alerts.append(f"Performance alert: {name} = {result.value:.3f} < {threshold}")
            
            # Check for significant degradation
            if result.improvement is not None and result.improvement < -0.1:
                alerts.append(f"Performance degradation: {name} decreased by {abs(result.improvement):.3f}")
        
        # Store alerts (in a real system, these would trigger notifications)
        if alerts:
            self.continuous_results[-1]["alerts"] = alerts
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a specific metric."""
        self.alert_thresholds[metric_name] = threshold
    
    def get_continuous_summary(self, window: int = 20) -> Dict[str, Any]:
        """Get summary of continuous benchmarking results."""
        if len(self.continuous_results) < 2:
            return {"insufficient_data": True}
        
        recent_results = list(self.continuous_results)[-window:]
        
        # Extract performance trends
        performance_values = []
        timestamps = []
        
        for result in recent_results:
            if "summary" in result and "overall_performance" in result["summary"]:
                performance_values.append(result["summary"]["overall_performance"])
                timestamps.append(result["timestamp"])
        
        summary = {
            "window_size": len(recent_results),
            "time_span": timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0,
            "current_performance": performance_values[-1] if performance_values else 0.5,
            "average_performance": np.mean(performance_values) if performance_values else 0.5,
            "performance_trend": self._calculate_trend(performance_values),
            "stability": 1.0 - np.std(performance_values) if len(performance_values) > 1 else 1.0,
            "alerts_count": sum(1 for r in recent_results if "alerts" in r),
            "benchmark_frequency": len(recent_results) / max(summary.get("time_span", 1), 1)
        }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in performance values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)


class SelfTuner:
    """
    Adaptive self-tuning system for kernels and attention mechanisms.
    
    Automatically adjusts system parameters based on performance feedback
    and optimization objectives.
    """
    
    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.tuning_history = deque(maxlen=300)
        self.parameter_ranges = {}
        self.tuning_strategies = {}
        self.performance_targets = {}
        
    def register_tunable_parameter(self, param_name: str, current_value: float,
                                 value_range: Tuple[float, float],
                                 tuning_strategy: str = "gradient"):
        """Register a parameter for automatic tuning."""
        self.parameter_ranges[param_name] = {
            "current": current_value,
            "range": value_range,
            "history": deque(maxlen=50),
            "performance_history": deque(maxlen=50)
        }
        self.tuning_strategies[param_name] = tuning_strategy
    
    def set_performance_target(self, metric_name: str, target_value: float):
        """Set performance target for tuning optimization."""
        self.performance_targets[metric_name] = target_value
    
    def tune_parameters(self, current_performance: Dict[str, float],
                       tunable_objects: Dict[str, Any] = None) -> Dict[str, float]:
        """Perform adaptive parameter tuning based on current performance."""
        tuning_results = {}
        
        for param_name in self.parameter_ranges:
            if param_name not in current_performance:
                continue
            
            current_value = self.parameter_ranges[param_name]["current"]
            performance_value = current_performance[param_name]
            
            # Determine tuning direction
            new_value = self._tune_single_parameter(param_name, current_value, performance_value)
            
            # Apply tuning if objects provided
            if tunable_objects and param_name in tunable_objects:
                self._apply_parameter_change(tunable_objects[param_name], param_name, new_value)
            
            # Update tracking
            self.parameter_ranges[param_name]["current"] = new_value
            self.parameter_ranges[param_name]["history"].append(new_value)
            self.parameter_ranges[param_name]["performance_history"].append(performance_value)
            
            tuning_results[param_name] = {
                "old_value": current_value,
                "new_value": new_value,
                "change": new_value - current_value,
                "performance": performance_value
            }
        
        # Record tuning session
        self.tuning_history.append({
            "timestamp": time.time(),
            "tuning_results": tuning_results,
            "performance": current_performance
        })
        
        return tuning_results
    
    def _tune_single_parameter(self, param_name: str, current_value: float,
                             performance_value: float) -> float:
        """Tune a single parameter based on performance."""
        strategy = self.tuning_strategies.get(param_name, "gradient")
        param_info = self.parameter_ranges[param_name]
        value_range = param_info["range"]
        
        if strategy == "gradient":
            return self._gradient_tuning(param_name, current_value, performance_value, value_range)
        elif strategy == "hill_climbing":
            return self._hill_climbing_tuning(param_name, current_value, performance_value, value_range)
        elif strategy == "adaptive":
            return self._adaptive_tuning(param_name, current_value, performance_value, value_range)
        else:
            return current_value
    
    def _gradient_tuning(self, param_name: str, current_value: float,
                        performance_value: float, value_range: Tuple[float, float]) -> float:
        """Gradient-based parameter tuning."""
        param_info = self.parameter_ranges[param_name]
        
        # Calculate gradient if we have history
        if len(param_info["history"]) >= 2 and len(param_info["performance_history"]) >= 2:
            value_history = list(param_info["history"])
            perf_history = list(param_info["performance_history"])
            
            # Simple finite difference gradient
            dvalue = value_history[-1] - value_history[-2]
            dperf = perf_history[-1] - perf_history[-2]
            
            if abs(dvalue) > 1e-6:
                gradient = dperf / dvalue
                
                # Move in direction of positive gradient
                step_size = self.adaptation_rate * (value_range[1] - value_range[0])
                change = step_size * np.sign(gradient)
                new_value = current_value + change
            else:
                # Random exploration
                exploration = np.random.uniform(-0.05, 0.05) * (value_range[1] - value_range[0])
                new_value = current_value + exploration
        else:
            # Random exploration for initial steps
            exploration = np.random.uniform(-0.1, 0.1) * (value_range[1] - value_range[0])
            new_value = current_value + exploration
        
        # Clip to valid range
        return np.clip(new_value, value_range[0], value_range[1])
    
    def _hill_climbing_tuning(self, param_name: str, current_value: float,
                            performance_value: float, value_range: Tuple[float, float]) -> float:
        """Hill climbing parameter tuning."""
        param_info = self.parameter_ranges[param_name]
        
        # Check if performance improved
        if len(param_info["performance_history"]) >= 2:
            prev_performance = param_info["performance_history"][-2]
            
            if performance_value > prev_performance:
                # Performance improved, continue in same direction
                if len(param_info["history"]) >= 2:
                    prev_value = param_info["history"][-2]
                    direction = current_value - prev_value
                    step_size = self.adaptation_rate * (value_range[1] - value_range[0])
                    new_value = current_value + step_size * np.sign(direction)
                else:
                    new_value = current_value
            else:
                # Performance degraded, try opposite direction
                step_size = self.adaptation_rate * (value_range[1] - value_range[0])
                direction = np.random.choice([-1, 1])
                new_value = current_value + step_size * direction
        else:
            # Initial random step
            step_size = self.adaptation_rate * (value_range[1] - value_range[0])
            direction = np.random.choice([-1, 1])
            new_value = current_value + step_size * direction
        
        return np.clip(new_value, value_range[0], value_range[1])
    
    def _adaptive_tuning(self, param_name: str, current_value: float,
                        performance_value: float, value_range: Tuple[float, float]) -> float:
        """Adaptive tuning that adjusts step size based on performance variance."""
        param_info = self.parameter_ranges[param_name]
        
        # Calculate performance variance for step size adaptation
        if len(param_info["performance_history"]) >= 5:
            perf_history = list(param_info["performance_history"])
            perf_variance = np.var(perf_history[-5:])
            
            # Adapt step size based on variance
            if perf_variance > 0.01:  # High variance - reduce step size
                step_size = self.adaptation_rate * 0.5
            else:  # Low variance - increase step size for exploration
                step_size = self.adaptation_rate * 1.5
        else:
            step_size = self.adaptation_rate
        
        # Check performance target
        target_performance = self.performance_targets.get(param_name, 0.8)
        if performance_value < target_performance:
            # Need improvement - more aggressive tuning
            step_size *= 2.0
        
        # Apply tuning
        step = step_size * (value_range[1] - value_range[0])
        direction = np.random.choice([-1, 1])
        new_value = current_value + step * direction
        
        return np.clip(new_value, value_range[0], value_range[1])
    
    def _apply_parameter_change(self, target_object: Any, param_name: str, new_value: float):
        """Apply parameter change to target object."""
        try:
            if hasattr(target_object, param_name):
                setattr(target_object, param_name, new_value)
            elif hasattr(target_object, f"set_{param_name}"):
                getattr(target_object, f"set_{param_name}")(new_value)
            elif isinstance(target_object, dict):
                target_object[param_name] = new_value
        except Exception as e:
            # Parameter change failed - continue without error
            pass
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get summary of tuning performance and trends."""
        if not self.tuning_history:
            return {"no_tuning_data": True}
        
        summary = {
            "total_tuning_sessions": len(self.tuning_history),
            "parameters_tuned": list(self.parameter_ranges.keys()),
            "parameter_trends": {},
            "performance_improvement": {}
        }
        
        # Calculate parameter trends
        for param_name, param_info in self.parameter_ranges.items():
            history = list(param_info["history"])
            if len(history) >= 3:
                trend = self._calculate_trend(history)
                summary["parameter_trends"][param_name] = {
                    "trend": trend,
                    "current_value": param_info["current"],
                    "value_range": param_info["range"],
                    "stability": 1.0 - np.std(history[-10:]) if len(history) >= 10 else 0.5
                }
        
        # Calculate performance improvements
        if len(self.tuning_history) >= 2:
            initial_performance = self.tuning_history[0]["performance"]
            latest_performance = self.tuning_history[-1]["performance"]
            
            for metric in initial_performance:
                if metric in latest_performance:
                    initial = initial_performance[metric]
                    latest = latest_performance[metric]
                    improvement = (latest - initial) / max(abs(initial), 0.001)
                    summary["performance_improvement"][metric] = improvement
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in parameter values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for cognitive trade-offs.
    
    Handles optimization problems with multiple competing objectives
    such as performance vs. stability, speed vs. accuracy, etc.
    """
    
    def __init__(self):
        self.objectives = []
        self.pareto_front = []
        self.optimization_history = deque(maxlen=200)
        self.trade_off_weights = {}
        
    def add_objective(self, name: str, weight: float = 1.0, minimize: bool = False,
                     target_value: Optional[float] = None):
        """Add optimization objective."""
        objective = OptimizationObjective(
            name=name,
            weight=weight,
            minimize=minimize,
            target_value=target_value
        )
        self.objectives.append(objective)
        
    def set_trade_off_weights(self, weights: Dict[str, float]):
        """Set trade-off weights between objectives."""
        self.trade_off_weights = weights
        
        # Update objective weights
        for objective in self.objectives:
            if objective.name in weights:
                objective.weight = weights[objective.name]
    
    def evaluate_solution(self, solution_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate a solution against all objectives."""
        objective_values = {}
        weighted_sum = 0.0
        
        for objective in self.objectives:
            if objective.name in solution_metrics:
                value = solution_metrics[objective.name]
                objective.current_value = value
                
                # Normalize value based on objective type
                if objective.minimize:
                    # For minimization, lower values are better
                    if objective.target_value is not None:
                        normalized_value = max(0, 1.0 - (value / objective.target_value))
                    else:
                        normalized_value = 1.0 / (1.0 + value)
                else:
                    # For maximization, higher values are better
                    if objective.target_value is not None:
                        normalized_value = min(1.0, value / objective.target_value)
                    else:
                        normalized_value = min(1.0, value)
                
                objective_values[objective.name] = {
                    "raw_value": value,
                    "normalized_value": normalized_value,
                    "weight": objective.weight,
                    "contribution": normalized_value * objective.weight
                }
                
                weighted_sum += normalized_value * objective.weight
        
        # Calculate overall fitness
        total_weight = sum(obj.weight for obj in self.objectives)
        overall_fitness = weighted_sum / max(total_weight, 1.0)
        
        evaluation = {
            "overall_fitness": overall_fitness,
            "objective_values": objective_values,
            "trade_offs": self._analyze_trade_offs(objective_values),
            "pareto_dominated": self._check_pareto_dominance(solution_metrics)
        }
        
        return evaluation
    
    def optimize_multi_objective(self, solution_generator: Callable,
                               num_solutions: int = 100) -> Dict[str, Any]:
        """Perform multi-objective optimization."""
        solutions = []
        
        # Generate and evaluate solutions
        for i in range(num_solutions):
            try:
                solution_metrics = solution_generator()
                evaluation = self.evaluate_solution(solution_metrics)
                
                solution = {
                    "id": i,
                    "metrics": solution_metrics,
                    "evaluation": evaluation,
                    "timestamp": time.time()
                }
                
                solutions.append(solution)
                
            except Exception as e:
                # Skip failed solutions
                continue
        
        # Update Pareto front
        self._update_pareto_front(solutions)
        
        # Find best solutions
        best_overall = max(solutions, key=lambda x: x["evaluation"]["overall_fitness"])
        best_per_objective = self._find_best_per_objective(solutions)
        
        optimization_result = {
            "num_solutions_evaluated": len(solutions),
            "best_overall_solution": best_overall,
            "best_per_objective": best_per_objective,
            "pareto_front_size": len(self.pareto_front),
            "trade_off_analysis": self._analyze_optimization_trade_offs(solutions),
            "convergence_metrics": self._calculate_convergence_metrics(solutions)
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def _analyze_trade_offs(self, objective_values: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze trade-offs between objectives."""
        trade_offs = {}
        
        objective_names = list(objective_values.keys())
        
        for i, obj1 in enumerate(objective_names):
            for obj2 in objective_names[i+1:]:
                val1 = objective_values[obj1]["normalized_value"]
                val2 = objective_values[obj2]["normalized_value"]
                
                # Calculate trade-off ratio
                if val2 > 0:
                    trade_off_ratio = val1 / val2
                else:
                    trade_off_ratio = float('inf') if val1 > 0 else 1.0
                
                trade_offs[f"{obj1}_vs_{obj2}"] = {
                    "ratio": trade_off_ratio,
                    "difference": val1 - val2,
                    "balance": 1.0 - abs(val1 - val2)  # Higher is more balanced
                }
        
        return trade_offs
    
    def _check_pareto_dominance(self, solution_metrics: Dict[str, float]) -> bool:
        """Check if solution is Pareto dominated."""
        for pareto_solution in self.pareto_front:
            if self._dominates(pareto_solution["metrics"], solution_metrics):
                return True
        return False
    
    def _dominates(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> bool:
        """Check if solution1 dominates solution2."""
        dominates = False
        
        for objective in self.objectives:
            if objective.name not in solution1 or objective.name not in solution2:
                continue
            
            val1 = solution1[objective.name]
            val2 = solution2[objective.name]
            
            if objective.minimize:
                if val1 > val2:  # Worse in this objective
                    return False
                elif val1 < val2:  # Better in this objective
                    dominates = True
            else:
                if val1 < val2:  # Worse in this objective
                    return False
                elif val1 > val2:  # Better in this objective
                    dominates = True
        
        return dominates
    
    def _update_pareto_front(self, new_solutions: List[Dict[str, Any]]):
        """Update Pareto front with new solutions."""
        all_solutions = self.pareto_front + new_solutions
        
        # Find non-dominated solutions
        pareto_front = []
        
        for solution in all_solutions:
            is_dominated = False
            
            for other_solution in all_solutions:
                if solution != other_solution:
                    if self._dominates(other_solution["metrics"], solution["metrics"]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(solution)
        
        self.pareto_front = pareto_front
    
    def _find_best_per_objective(self, solutions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Find best solution for each individual objective."""
        best_per_objective = {}
        
        for objective in self.objectives:
            if objective.minimize:
                best_solution = min(solutions, 
                                  key=lambda x: x["metrics"].get(objective.name, float('inf')))
            else:
                best_solution = max(solutions,
                                  key=lambda x: x["metrics"].get(objective.name, 0.0))
            
            best_per_objective[objective.name] = best_solution
        
        return best_per_objective
    
    def _analyze_optimization_trade_offs(self, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trade-offs across all solutions."""
        if len(solutions) < 2:
            return {"insufficient_solutions": True}
        
        # Calculate correlations between objectives
        correlations = {}
        objective_names = [obj.name for obj in self.objectives]
        
        for i, obj1 in enumerate(objective_names):
            for obj2 in objective_names[i+1:]:
                values1 = [s["metrics"].get(obj1, 0) for s in solutions]
                values2 = [s["metrics"].get(obj2, 0) for s in solutions]
                
                if len(values1) >= 2 and len(values2) >= 2:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    correlations[f"{obj1}_vs_{obj2}"] = correlation
        
        # Analyze Pareto efficiency
        pareto_efficiency = len(self.pareto_front) / len(solutions)
        
        return {
            "objective_correlations": correlations,
            "pareto_efficiency": pareto_efficiency,
            "trade_off_strength": np.mean([abs(corr) for corr in correlations.values()]) if correlations else 0.0
        }
    
    def _calculate_convergence_metrics(self, solutions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate convergence metrics for optimization."""
        if len(solutions) < 10:
            return {"insufficient_solutions": True}
        
        # Calculate fitness variance (lower = more converged)
        fitness_values = [s["evaluation"]["overall_fitness"] for s in solutions]
        fitness_variance = np.var(fitness_values)
        
        # Calculate diversity (higher = more diverse)
        diversity = self._calculate_solution_diversity(solutions)
        
        return {
            "fitness_variance": fitness_variance,
            "solution_diversity": diversity,
            "convergence_score": 1.0 / (1.0 + fitness_variance)  # Higher = more converged
        }
    
    def _calculate_solution_diversity(self, solutions: List[Dict[str, Any]]) -> float:
        """Calculate diversity of solutions."""
        if len(solutions) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        distances = []
        
        for i, sol1 in enumerate(solutions):
            for sol2 in solutions[i+1:]:
                distance = 0.0
                count = 0
                
                for objective in self.objectives:
                    if objective.name in sol1["metrics"] and objective.name in sol2["metrics"]:
                        val1 = sol1["metrics"][objective.name]
                        val2 = sol2["metrics"][objective.name]
                        distance += (val1 - val2) ** 2
                        count += 1
                
                if count > 0:
                    distances.append(np.sqrt(distance / count))
        
        return np.mean(distances) if distances else 0.0


class FitnessEvaluator:
    """
    Fitness evaluation system for evolutionary optimization.
    
    Provides comprehensive fitness evaluation combining performance metrics,
    stability measures, and optimization objectives.
    """
    
    def __init__(self):
        self.fitness_components = {}
        self.evaluation_history = deque(maxlen=500)
        self.baseline_performance = {}
        
    def register_fitness_component(self, name: str, weight: float = 1.0,
                                 evaluation_function: Optional[Callable] = None):
        """Register a fitness component for evaluation."""
        self.fitness_components[name] = {
            "weight": weight,
            "evaluation_function": evaluation_function,
            "history": deque(maxlen=100)
        }
    
    def evaluate_reservoir_fitness(self, reservoir, test_data: Optional[np.ndarray] = None) -> float:
        """Evaluate comprehensive fitness of reservoir configuration."""
        fitness_scores = {}
        
        # Core reservoir fitness components
        fitness_scores.update(self._evaluate_spectral_fitness(reservoir))
        fitness_scores.update(self._evaluate_dynamical_fitness(reservoir))
        fitness_scores.update(self._evaluate_stability_fitness(reservoir))
        
        # Performance-based fitness (if test data provided)
        if test_data is not None:
            fitness_scores.update(self._evaluate_performance_fitness(reservoir, test_data))
        
        # Combine fitness scores
        total_fitness = self._combine_fitness_scores(fitness_scores)
        
        # Record evaluation
        evaluation_record = {
            "timestamp": time.time(),
            "fitness_scores": fitness_scores,
            "total_fitness": total_fitness,
            "reservoir_properties": self._extract_reservoir_properties(reservoir)
        }
        
        self.evaluation_history.append(evaluation_record)
        return total_fitness
    
    def _evaluate_spectral_fitness(self, reservoir) -> Dict[str, float]:
        """Evaluate fitness based on spectral properties."""
        scores = {}
        
        try:
            if hasattr(reservoir, 'W') and reservoir.W is not None:
                eigenvals = np.linalg.eigvals(reservoir.W)
                spectral_radius = np.max(np.abs(eigenvals))
                
                # Optimal spectral radius is around 0.9-1.0
                if spectral_radius <= 1.0:
                    spectral_fitness = min(1.0, spectral_radius / 0.9)
                else:
                    spectral_fitness = max(0.1, 1.0 / spectral_radius)
                
                scores["spectral_fitness"] = spectral_fitness
                
                # Spectral distribution fitness
                eigenval_distribution = np.abs(eigenvals)
                eigenval_distribution = eigenval_distribution / np.sum(eigenval_distribution)
                distribution_entropy = -np.sum(eigenval_distribution * np.log(eigenval_distribution + 1e-10))
                scores["spectral_diversity"] = min(1.0, distribution_entropy / 5.0)
            else:
                scores["spectral_fitness"] = 0.1
                scores["spectral_diversity"] = 0.1
        except:
            scores["spectral_fitness"] = 0.1
            scores["spectral_diversity"] = 0.1
        
        return scores
    
    def _evaluate_dynamical_fitness(self, reservoir) -> Dict[str, float]:
        """Evaluate fitness based on dynamical properties."""
        scores = {}
        
        try:
            if hasattr(reservoir, 'state') and reservoir.state is not None:
                state = reservoir.state
                
                # State variance fitness (moderate variance is good)
                state_variance = np.var(state)
                if 0.1 <= state_variance <= 1.0:
                    variance_fitness = 1.0
                elif state_variance < 0.1:
                    variance_fitness = state_variance / 0.1
                else:
                    variance_fitness = 1.0 / (1.0 + (state_variance - 1.0))
                
                scores["variance_fitness"] = variance_fitness
                
                # Dynamical richness fitness
                try:
                    _, s, _ = np.linalg.svd(state.reshape(-1, 1))
                    effective_rank = np.sum(s > 0.01 * s[0]) / len(s)
                    richness_fitness = min(1.0, effective_rank / 0.7)
                    scores["richness_fitness"] = richness_fitness
                except:
                    scores["richness_fitness"] = 0.5
                
                # State entropy fitness
                state_norm = np.abs(state) / (np.sum(np.abs(state)) + 1e-10)
                entropy = -np.sum(state_norm * np.log(state_norm + 1e-10))
                entropy_fitness = min(1.0, entropy / 5.0)
                scores["entropy_fitness"] = entropy_fitness
            else:
                scores["variance_fitness"] = 0.3
                scores["richness_fitness"] = 0.3
                scores["entropy_fitness"] = 0.3
        except:
            scores["variance_fitness"] = 0.1
            scores["richness_fitness"] = 0.1
            scores["entropy_fitness"] = 0.1
        
        return scores
    
    def _evaluate_stability_fitness(self, reservoir) -> Dict[str, float]:
        """Evaluate fitness based on stability properties."""
        scores = {}
        
        try:
            if hasattr(reservoir, 'W') and reservoir.W is not None:
                # Echo state property fitness
                eigenvals = np.linalg.eigvals(reservoir.W)
                spectral_radius = np.max(np.abs(eigenvals))
                echo_state_fitness = 1.0 if spectral_radius < 1.0 else 0.1
                scores["echo_state_fitness"] = echo_state_fitness
                
                # Connection sparsity fitness (moderate sparsity is good)
                total_connections = reservoir.W.size
                nonzero_connections = np.count_nonzero(reservoir.W)
                sparsity = 1.0 - (nonzero_connections / total_connections)
                
                if 0.3 <= sparsity <= 0.8:
                    sparsity_fitness = 1.0
                else:
                    distance_from_optimal = min(abs(sparsity - 0.3), abs(sparsity - 0.8))
                    sparsity_fitness = max(0.1, 1.0 - distance_from_optimal * 2.0)
                
                scores["sparsity_fitness"] = sparsity_fitness
            else:
                scores["echo_state_fitness"] = 0.1
                scores["sparsity_fitness"] = 0.1
        except:
            scores["echo_state_fitness"] = 0.1
            scores["sparsity_fitness"] = 0.1
        
        return scores
    
    def _evaluate_performance_fitness(self, reservoir, test_data: np.ndarray) -> Dict[str, float]:
        """Evaluate fitness based on actual performance on test data."""
        scores = {}
        
        try:
            # Simple performance test: memory capacity approximation
            input_seq = test_data[:50] if len(test_data) > 50 else test_data
            
            # Simulate reservoir computation (simplified)
            states = []
            current_state = np.random.randn(reservoir.W.shape[0]) * 0.1 if hasattr(reservoir, 'W') else np.array([0.1])
            
            for inp in input_seq:
                if hasattr(reservoir, 'W'):
                    # Simplified reservoir update
                    new_state = np.tanh(reservoir.W @ current_state + inp * 0.1)
                    states.append(new_state)
                    current_state = new_state
                else:
                    states.append(current_state)
            
            if states:
                # Performance metrics
                state_matrix = np.array(states)
                
                # Memory capacity (correlation with delayed inputs)
                memory_capacity = 0.0
                for delay in range(1, min(10, len(input_seq))):
                    if delay < len(states):
                        delayed_input = input_seq[:-delay] if delay > 0 else input_seq
                        state_projection = np.mean(state_matrix[delay:], axis=1)
                        
                        if len(delayed_input) == len(state_projection):
                            correlation = np.corrcoef(delayed_input, state_projection)[0, 1]
                            memory_capacity += max(0, correlation ** 2)
                
                memory_fitness = min(1.0, memory_capacity / 5.0)
                scores["memory_fitness"] = memory_fitness
                
                # Separation property (different inputs produce different states)
                if len(states) > 1:
                    state_distances = []
                    for i in range(len(states) - 1):
                        distance = np.linalg.norm(states[i] - states[i+1])
                        state_distances.append(distance)
                    
                    separation = np.mean(state_distances)
                    separation_fitness = min(1.0, separation / 2.0)
                    scores["separation_fitness"] = separation_fitness
                else:
                    scores["separation_fitness"] = 0.5
            else:
                scores["memory_fitness"] = 0.1
                scores["separation_fitness"] = 0.1
        
        except Exception as e:
            scores["memory_fitness"] = 0.1
            scores["separation_fitness"] = 0.1
        
        return scores
    
    def _combine_fitness_scores(self, fitness_scores: Dict[str, float]) -> float:
        """Combine individual fitness scores into total fitness."""
        if not fitness_scores:
            return 0.1
        
        # Default weights for fitness components
        default_weights = {
            "spectral_fitness": 0.2,
            "spectral_diversity": 0.1,
            "variance_fitness": 0.15,
            "richness_fitness": 0.15,
            "entropy_fitness": 0.1,
            "echo_state_fitness": 0.15,
            "sparsity_fitness": 0.1,
            "memory_fitness": 0.25,
            "separation_fitness": 0.2
        }
        
        # Use registered weights if available
        weights = {}
        for component_name in fitness_scores:
            if component_name in self.fitness_components:
                weights[component_name] = self.fitness_components[component_name]["weight"]
            else:
                weights[component_name] = default_weights.get(component_name, 1.0)
        
        # Calculate weighted sum
        weighted_sum = sum(fitness_scores[name] * weights[name] for name in fitness_scores)
        total_weight = sum(weights.values())
        
        total_fitness = weighted_sum / max(total_weight, 1.0)
        
        # Record component histories
        for name, score in fitness_scores.items():
            if name in self.fitness_components:
                self.fitness_components[name]["history"].append(score)
        
        return float(min(1.0, max(0.0, total_fitness)))
    
    def _extract_reservoir_properties(self, reservoir) -> Dict[str, Any]:
        """Extract key properties of reservoir for analysis."""
        properties = {}
        
        if hasattr(reservoir, 'W') and reservoir.W is not None:
            properties["matrix_shape"] = reservoir.W.shape
            properties["nonzero_connections"] = int(np.count_nonzero(reservoir.W))
            properties["sparsity"] = 1.0 - (properties["nonzero_connections"] / reservoir.W.size)
            
            try:
                eigenvals = np.linalg.eigvals(reservoir.W)
                properties["spectral_radius"] = float(np.max(np.abs(eigenvals)))
                properties["num_eigenvalues"] = len(eigenvals)
            except:
                properties["spectral_radius"] = 0.0
                properties["num_eigenvalues"] = 0
        
        if hasattr(reservoir, 'state') and reservoir.state is not None:
            properties["state_size"] = len(reservoir.state)
            properties["state_mean"] = float(np.mean(reservoir.state))
            properties["state_std"] = float(np.std(reservoir.state))
        
        return properties
    
    def get_fitness_trends(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """Get fitness trends for analysis."""
        if component_name and component_name in self.fitness_components:
            history = list(self.fitness_components[component_name]["history"])
            
            if len(history) < 3:
                return {"insufficient_data": True}
            
            return {
                "component": component_name,
                "current_fitness": history[-1],
                "mean_fitness": np.mean(history),
                "fitness_trend": self._calculate_trend(history),
                "fitness_stability": 1.0 - np.std(history),
                "improvement_rate": (history[-1] - history[0]) / max(abs(history[0]), 0.001)
            }
        else:
            # Overall fitness trends
            if len(self.evaluation_history) < 3:
                return {"insufficient_data": True}
            
            total_fitness_history = [eval_record["total_fitness"] for eval_record in self.evaluation_history]
            
            return {
                "overall_fitness_trend": self._calculate_trend(total_fitness_history),
                "current_fitness": total_fitness_history[-1],
                "mean_fitness": np.mean(total_fitness_history),
                "best_fitness": max(total_fitness_history),
                "fitness_improvement": (total_fitness_history[-1] - total_fitness_history[0]) / max(abs(total_fitness_history[0]), 0.001)
            }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in fitness values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)