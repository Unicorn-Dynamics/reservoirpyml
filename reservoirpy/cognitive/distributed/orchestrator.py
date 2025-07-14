"""
Cognitive Agent Orchestrator

Handles multi-agent task distribution, load balancing, and fault tolerance
for the distributed cognitive mesh.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .mesh_coordinator import MeshCoordinator, ReservoirNode


class TaskPriority(Enum):
    """Task priority levels for cognitive processing."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AgentStatus(Enum):
    """Agent status in the cognitive mesh."""
    IDLE = "idle"
    PROCESSING = "processing" 
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class CognitiveTask:
    """Represents a cognitive task to be distributed across agents."""
    
    task_id: str
    task_type: str
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    required_capabilities: Set[str] = field(default_factory=set)
    target_agents: List[str] = field(default_factory=list)
    max_execution_time: float = 30.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_nodes: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.started_at is None:
            return False
        return time.time() - self.started_at > self.max_execution_time
    
    @property
    def execution_time(self) -> float:
        """Get task execution time."""
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at


@dataclass  
class CognitiveAgent:
    """Represents a cognitive agent in the mesh."""
    
    agent_id: str
    agent_type: str
    embodiment: str = "none"
    capabilities: Set[str] = field(default_factory=set)
    status: AgentStatus = AgentStatus.IDLE
    assigned_node: Optional[str] = None
    current_task: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    task_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def assign_task(self, task_id: str):
        """Assign a task to this agent."""
        self.current_task = task_id
        self.status = AgentStatus.PROCESSING
        self.update_activity()
    
    def complete_task(self, task_id: str):
        """Mark task as completed."""
        if self.current_task == task_id:
            self.task_history.append(task_id)
            self.current_task = None
            self.status = AgentStatus.IDLE
            self.update_activity()


class TaskDistributor:
    """
    Distributes cognitive tasks across available agents and nodes.
    
    Implements load balancing, priority scheduling, and fault tolerance.
    """
    
    def __init__(self, mesh_coordinator: MeshCoordinator):
        self.mesh_coordinator = mesh_coordinator
        self.task_queue: List[CognitiveTask] = []
        self.active_tasks: Dict[str, CognitiveTask] = {}
        self.completed_tasks: Dict[str, CognitiveTask] = {}
        self.task_lock = threading.Lock()
        
        # Load balancing configuration
        self.max_tasks_per_node = 5
        self.load_balancing_strategy = "round_robin"  # round_robin, least_loaded, capability_based
        
    def submit_task(self, task: CognitiveTask) -> bool:
        """Submit a task for distribution."""
        with self.task_lock:
            # Check for duplicate task
            if task.task_id in self.active_tasks or task.task_id in self.completed_tasks:
                return False
            
            # Add to queue based on priority
            self._insert_task_by_priority(task)
            return True
    
    def get_next_task(self, agent_id: str, node_id: str) -> Optional[CognitiveTask]:
        """Get next task for an agent/node."""
        with self.task_lock:
            # Check node capacity
            node_task_count = sum(
                1 for task in self.active_tasks.values()
                if node_id in task.assigned_nodes
            )
            
            if node_task_count >= self.max_tasks_per_node:
                return None
            
            # Find suitable task
            for i, task in enumerate(self.task_queue):
                if self._can_assign_task(task, agent_id, node_id):
                    # Remove from queue and assign
                    task = self.task_queue.pop(i)
                    task.started_at = time.time()
                    task.assigned_nodes.append(node_id)
                    self.active_tasks[task.task_id] = task
                    return task
            
            return None
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark task as completed."""
        with self.task_lock:
            if task_id not in self.active_tasks:
                return False
            
            task = self.active_tasks.pop(task_id)
            task.completed_at = time.time()
            task.result = result
            self.completed_tasks[task_id] = task
            
            return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed."""
        with self.task_lock:
            if task_id not in self.active_tasks:
                return False
            
            task = self.active_tasks.pop(task_id)
            task.completed_at = time.time()
            task.error = error
            self.completed_tasks[task_id] = task
            
            return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        with self.task_lock:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": "active",
                    "progress": self._calculate_task_progress(task),
                    "assigned_nodes": task.assigned_nodes,
                    "execution_time": task.execution_time,
                    "priority": task.priority.name
                }
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": "completed" if task.error is None else "failed",
                    "result": task.result,
                    "error": task.error,
                    "execution_time": task.execution_time,
                    "priority": task.priority.name
                }
            
            # Check queued tasks
            for task in self.task_queue:
                if task.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "status": "queued",
                        "queue_position": self.task_queue.index(task),
                        "priority": task.priority.name
                    }
            
            return None
    
    def get_distribution_statistics(self) -> Dict[str, Any]:
        """Get task distribution statistics."""
        with self.task_lock:
            # Calculate load per node
            node_loads = {}
            for task in self.active_tasks.values():
                for node_id in task.assigned_nodes:
                    node_loads[node_id] = node_loads.get(node_id, 0) + 1
            
            # Priority distribution
            priority_counts = {priority.name: 0 for priority in TaskPriority}
            for task in self.task_queue:
                priority_counts[task.priority.name] += 1
            
            return {
                "queued_tasks": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "node_loads": node_loads,
                "priority_distribution": priority_counts,
                "average_execution_time": self._calculate_average_execution_time()
            }
    
    def cleanup_expired_tasks(self) -> List[str]:
        """Clean up expired tasks and return their IDs."""
        with self.task_lock:
            expired_task_ids = []
            
            for task_id, task in list(self.active_tasks.items()):
                if task.is_expired:
                    expired_task_ids.append(task_id)
                    task.error = "Task expired"
                    task.completed_at = time.time()
                    self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
            
            return expired_task_ids
    
    def _insert_task_by_priority(self, task: CognitiveTask):
        """Insert task into queue based on priority."""
        # Find insertion position based on priority
        insert_pos = len(self.task_queue)
        for i, existing_task in enumerate(self.task_queue):
            if task.priority.value > existing_task.priority.value:
                insert_pos = i
                break
        
        self.task_queue.insert(insert_pos, task)
    
    def _can_assign_task(self, task: CognitiveTask, agent_id: str, node_id: str) -> bool:
        """Check if task can be assigned to agent/node."""
        # Check if task targets specific agents
        if task.target_agents and agent_id not in task.target_agents:
            return False
        
        # Check node capabilities (simplified)
        node = self.mesh_coordinator.nodes.get(node_id)
        if node and task.required_capabilities:
            if not task.required_capabilities.issubset(node.capabilities):
                return False
        
        return True
    
    def _calculate_task_progress(self, task: CognitiveTask) -> float:
        """Calculate task progress (0-1)."""
        if task.started_at is None:
            return 0.0
        
        elapsed = time.time() - task.started_at
        progress = min(1.0, elapsed / task.max_execution_time)
        return progress
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average task execution time."""
        completed_tasks = [
            task for task in self.completed_tasks.values()
            if task.completed_at is not None and task.started_at is not None
        ]
        
        if not completed_tasks:
            return 0.0
        
        total_time = sum(task.execution_time for task in completed_tasks)
        return total_time / len(completed_tasks)


class AgentOrchestrator:
    """
    Orchestrates cognitive agents across the distributed mesh.
    
    Handles agent lifecycle, load balancing, fault tolerance, and
    coordination with the mesh infrastructure.
    """
    
    def __init__(self, mesh_coordinator: MeshCoordinator):
        self.mesh_coordinator = mesh_coordinator
        self.task_distributor = TaskDistributor(mesh_coordinator)
        
        # Agent management
        self.agents: Dict[str, CognitiveAgent] = {}
        self.agent_lock = threading.Lock()
        
        # Fault tolerance
        self.health_check_interval = 10.0  # seconds
        self.max_inactive_time = 60.0  # seconds
        self.auto_recovery = True
        
        # Performance monitoring
        self.orchestration_metrics = {
            "total_agents_spawned": 0,
            "total_tasks_distributed": 0,
            "total_failures": 0,
            "average_response_time": 0.0
        }
        
        # Start background monitoring
        self._start_monitoring()
    
    def spawn_agent(self, agent_type: str, embodiment: str = "none", 
                   capabilities: Set[str] = None, config: Dict[str, Any] = None) -> str:
        """Spawn a new cognitive agent."""
        with self.agent_lock:
            # Generate agent ID
            agent_id = f"agent_{int(time.time() * 1000)}_{len(self.agents)}"
            
            # Create agent
            agent = CognitiveAgent(
                agent_id=agent_id,
                agent_type=agent_type,
                embodiment=embodiment,
                capabilities=capabilities or set()
            )
            
            # Assign to optimal node
            optimal_node = self._select_optimal_node_for_agent(agent)
            agent.assigned_node = optimal_node
            
            # Register agent
            self.agents[agent_id] = agent
            self.orchestration_metrics["total_agents_spawned"] += 1
            
            return agent_id
    
    def terminate_agent(self, agent_id: str) -> bool:
        """Terminate a cognitive agent."""
        with self.agent_lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # Cancel current task if any
            if agent.current_task:
                self.task_distributor.fail_task(
                    agent.current_task, 
                    f"Agent {agent_id} terminated"
                )
            
            # Mark as terminated
            agent.status = AgentStatus.TERMINATED
            
            # Remove from active agents after a delay
            # (keeping for historical purposes)
            
            return True
    
    def submit_task(self, task_type: str, data: Dict[str, Any], 
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   target_agents: List[str] = None,
                   required_capabilities: Set[str] = None,
                   max_execution_time: float = 30.0) -> str:
        """Submit a task for distributed processing."""
        # Generate task ID
        task_id = f"task_{int(time.time() * 1000)}_{self.orchestration_metrics['total_tasks_distributed']}"
        
        # Create task
        task = CognitiveTask(
            task_id=task_id,
            task_type=task_type,
            data=data,
            priority=priority,
            target_agents=target_agents or [],
            required_capabilities=required_capabilities or set(),
            max_execution_time=max_execution_time
        )
        
        # Submit to distributor
        if self.task_distributor.submit_task(task):
            self.orchestration_metrics["total_tasks_distributed"] += 1
            return task_id
        else:
            raise ValueError(f"Failed to submit task {task_id}")
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        with self.agent_lock:
            if agent_id not in self.agents:
                return None
            
            agent = self.agents[agent_id]
            return {
                "agent_id": agent_id,
                "agent_type": agent.agent_type,
                "embodiment": agent.embodiment,
                "status": agent.status.value,
                "assigned_node": agent.assigned_node,
                "current_task": agent.current_task,
                "capabilities": list(agent.capabilities),
                "task_count": len(agent.task_history),
                "last_activity": agent.last_activity,
                "performance_metrics": agent.performance_metrics
            }
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get overall orchestration status."""
        with self.agent_lock:
            active_agents = [
                agent for agent in self.agents.values()
                if agent.status != AgentStatus.TERMINATED
            ]
            
            status_counts = {}
            for status in AgentStatus:
                status_counts[status.value] = len([
                    agent for agent in active_agents 
                    if agent.status == status
                ])
            
            return {
                "total_agents": len(self.agents),
                "active_agents": len(active_agents),
                "agent_status_distribution": status_counts,
                "task_distribution": self.task_distributor.get_distribution_statistics(),
                "mesh_status": self.mesh_coordinator.get_status(),
                "performance_metrics": self.orchestration_metrics
            }
    
    def process_agent_work(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Process work for a specific agent."""
        with self.agent_lock:
            if agent_id not in self.agents:
                return None
            
            agent = self.agents[agent_id]
            
            # Skip if agent is not idle
            if agent.status != AgentStatus.IDLE:
                return {"status": "busy"}
            
            # Get next task
            task = self.task_distributor.get_next_task(agent_id, agent.assigned_node)
            
            if task is None:
                return {"status": "no_tasks"}
            
            # Assign task to agent
            agent.assign_task(task.task_id)
            
            return {
                "status": "task_assigned",
                "task_id": task.task_id,
                "task_type": task.task_type,
                "task_data": task.data
            }
    
    def complete_agent_task(self, agent_id: str, task_id: str, 
                          result: Dict[str, Any]) -> bool:
        """Complete a task for an agent."""
        with self.agent_lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # Verify agent is working on this task
            if agent.current_task != task_id:
                return False
            
            # Complete task
            success = self.task_distributor.complete_task(task_id, result)
            if success:
                agent.complete_task(task_id)
                
                # Update performance metrics
                agent.performance_metrics["tasks_completed"] = (
                    agent.performance_metrics.get("tasks_completed", 0) + 1
                )
            
            return success
    
    def _select_optimal_node_for_agent(self, agent: CognitiveAgent) -> str:
        """Select optimal node for agent assignment."""
        # Simple load-based selection
        nodes = self.mesh_coordinator.nodes
        if not nodes:
            return "default"
        
        # Find node with lowest load
        min_load_node = min(
            nodes.items(),
            key=lambda x: x[1].load
        )
        
        return min_load_node[0]
    
    def _start_monitoring(self):
        """Start background monitoring for fault tolerance."""
        def monitor_loop():
            while True:
                try:
                    self._health_check()
                    self._cleanup_expired_tasks()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    print(f"Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _health_check(self):
        """Perform health check on agents and nodes."""
        current_time = time.time()
        
        with self.agent_lock:
            for agent in self.agents.values():
                # Check for inactive agents
                if (agent.status == AgentStatus.PROCESSING and 
                    current_time - agent.last_activity > self.max_inactive_time):
                    
                    # Mark as error and recover
                    agent.status = AgentStatus.ERROR
                    self.orchestration_metrics["total_failures"] += 1
                    
                    # Cancel current task
                    if agent.current_task:
                        self.task_distributor.fail_task(
                            agent.current_task,
                            "Agent health check timeout"
                        )
                        agent.current_task = None
                    
                    # Auto-recovery
                    if self.auto_recovery:
                        agent.status = AgentStatus.IDLE
    
    def _cleanup_expired_tasks(self):
        """Clean up expired tasks."""
        expired_tasks = self.task_distributor.cleanup_expired_tasks()
        
        # Update agents with expired tasks
        with self.agent_lock:
            for agent in self.agents.values():
                if agent.current_task in expired_tasks:
                    agent.status = AgentStatus.IDLE
                    agent.current_task = None