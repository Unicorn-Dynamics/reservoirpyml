"""
Resource Allocation and Economic Attention Markets
=================================================

Implements economic attention markets, resource scheduling, and attention banking
for dynamic cognitive resource allocation in ECAN systems.
"""

import numpy as np
import heapq
import time
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import PriorityQueue, Queue
import uuid

from .ecan import AttentionValue, ECANAttentionSystem


class ResourceType(Enum):
    """Types of cognitive resources that can be allocated."""
    COMPUTATION = "computation"
    MEMORY = "memory" 
    ATTENTION = "attention"
    COMMUNICATION = "communication"
    STORAGE = "storage"


@dataclass
class ResourceRequest:
    """
    Request for cognitive resources with economic bidding.
    
    Parameters
    ----------
    request_id : str
        Unique identifier for the request
    resource_type : ResourceType
        Type of resource being requested
    amount : float
        Amount of resource requested
    max_bid : float
        Maximum attention value willing to pay
    priority : float
        Priority level (higher = more urgent)
    deadline : float, optional
        Deadline timestamp for the request
    callback : Callable, optional
        Callback function when request is fulfilled
    """
    request_id: str
    resource_type: ResourceType
    amount: float
    max_bid: float
    priority: float
    deadline: Optional[float] = None
    callback: Optional[Callable] = None
    request_time: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """Priority queue ordering by priority and bid."""
        return (self.priority, self.max_bid) > (other.priority, other.max_bid)


@dataclass 
class ResourceAllocation:
    """
    Allocation of resources to a request.
    
    Parameters
    ----------
    allocation_id : str
        Unique identifier for the allocation
    request : ResourceRequest
        Original resource request
    allocated_amount : float
        Amount of resource actually allocated
    cost : float
        Actual cost paid for the allocation
    start_time : float
        When allocation began
    duration : float
        How long allocation will last
    """
    allocation_id: str
    request: ResourceRequest
    allocated_amount: float
    cost: float
    start_time: float
    duration: float
    
    @property
    def end_time(self) -> float:
        """Calculate when allocation expires."""
        return self.start_time + self.duration


class AttentionMarket:
    """
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
    """
    
    def __init__(
        self,
        initial_resources: Dict[ResourceType, float],
        base_prices: Optional[Dict[ResourceType, float]] = None,
        price_elasticity: float = 0.5,
        market_efficiency: float = 0.9
    ):
        self.resource_capacity = initial_resources.copy()
        self.available_resources = initial_resources.copy()
        self.base_prices = base_prices or {rt: 1.0 for rt in ResourceType}
        self.current_prices = self.base_prices.copy()
        self.price_elasticity = price_elasticity
        self.market_efficiency = market_efficiency
        
        # Market state
        self.pending_requests: Dict[ResourceType, List[ResourceRequest]] = {
            rt: [] for rt in ResourceType
        }
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        
        # Market history and statistics
        self.price_history: Dict[ResourceType, List[Tuple[float, float]]] = {
            rt: [] for rt in ResourceType
        }
        self.transaction_history: List[ResourceAllocation] = []
        self.market_stats = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'total_revenue': 0.0,
            'average_utilization': 0.0
        }
    
    def submit_request(self, request: ResourceRequest) -> bool:
        """Submit a resource request to the market."""
        self.market_stats['total_requests'] += 1
        
        # Validate request
        if request.amount <= 0 or request.max_bid <= 0:
            return False
        
        # Check if resource type is supported
        if request.resource_type not in self.resource_capacity:
            return False
        
        # Add to pending requests
        heapq.heappush(self.pending_requests[request.resource_type], request)
        
        # Trigger market clearing
        self._clear_market(request.resource_type)
        return True
    
    def _clear_market(self, resource_type: ResourceType):
        """Clear the market for a specific resource type."""
        requests = self.pending_requests[resource_type]
        available = self.available_resources[resource_type]
        
        if not requests or available <= 0:
            return
        
        # Sort requests by economic priority (bid * priority)
        sorted_requests = sorted(requests, key=lambda r: r.max_bid * r.priority, reverse=True)
        
        # Allocate resources to highest bidders
        allocated_requests = []
        remaining_capacity = available
        
        for request in sorted_requests:
            if remaining_capacity <= 0:
                break
            
            # Determine allocation amount and price
            allocation_amount = min(request.amount, remaining_capacity)
            current_price = self._calculate_price(resource_type, allocation_amount)
            total_cost = allocation_amount * current_price
            
            # Check if request can afford the allocation
            if total_cost <= request.max_bid:
                # Create allocation
                allocation = ResourceAllocation(
                    allocation_id=str(uuid.uuid4()),
                    request=request,
                    allocated_amount=allocation_amount,
                    cost=total_cost,
                    start_time=time.time(),
                    duration=allocation_amount * 10.0  # Simple duration calculation
                )
                
                # Record allocation
                self.active_allocations[allocation.allocation_id] = allocation
                self.transaction_history.append(allocation)
                
                # Update resources and statistics
                remaining_capacity -= allocation_amount
                self.available_resources[resource_type] = remaining_capacity
                self.market_stats['successful_allocations'] += 1
                self.market_stats['total_revenue'] += total_cost
                
                # Execute callback if provided
                if request.callback:
                    try:
                        request.callback(allocation)
                    except Exception as e:
                        print(f"Callback error for request {request.request_id}: {e}")
                
                allocated_requests.append(request)
            else:
                self.market_stats['failed_allocations'] += 1
        
        # Remove allocated requests from pending
        self.pending_requests[resource_type] = [
            r for r in requests if r not in allocated_requests
        ]
        
        # Update market prices based on demand
        self._update_prices(resource_type)
    
    def _calculate_price(self, resource_type: ResourceType, amount: float) -> float:
        """Calculate current market price for a resource."""
        base_price = self.base_prices[resource_type]
        capacity = self.resource_capacity[resource_type]
        available = self.available_resources[resource_type]
        
        # Supply-demand pricing
        utilization = 1.0 - (available / capacity) if capacity > 0 else 1.0
        demand_multiplier = 1.0 + (utilization ** self.price_elasticity)
        
        return base_price * demand_multiplier * self.market_efficiency
    
    def _update_prices(self, resource_type: ResourceType):
        """Update market prices based on current demand and supply."""
        new_price = self._calculate_price(resource_type, 1.0)
        self.current_prices[resource_type] = new_price
        
        # Record price history
        current_time = time.time()
        self.price_history[resource_type].append((current_time, new_price))
        
        # Limit history size
        if len(self.price_history[resource_type]) > 1000:
            self.price_history[resource_type] = self.price_history[resource_type][-500:]
    
    def release_allocation(self, allocation_id: str):
        """Release an active resource allocation."""
        if allocation_id in self.active_allocations:
            allocation = self.active_allocations[allocation_id]
            resource_type = allocation.request.resource_type
            
            # Return resources to available pool
            self.available_resources[resource_type] += allocation.allocated_amount
            
            # Remove from active allocations
            del self.active_allocations[allocation_id]
            
            # Clear any pending requests that can now be satisfied
            self._clear_market(resource_type)
    
    def cleanup_expired_allocations(self):
        """Remove expired allocations and free up resources."""
        current_time = time.time()
        expired_allocations = []
        
        for allocation_id, allocation in self.active_allocations.items():
            if current_time >= allocation.end_time:
                expired_allocations.append(allocation_id)
        
        for allocation_id in expired_allocations:
            self.release_allocation(allocation_id)
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status and statistics."""
        self.cleanup_expired_allocations()
        
        # Calculate current utilization
        total_capacity = sum(self.resource_capacity.values())
        total_available = sum(self.available_resources.values())
        current_utilization = 1.0 - (total_available / total_capacity) if total_capacity > 0 else 0.0
        
        return {
            'resource_capacity': self.resource_capacity.copy(),
            'available_resources': self.available_resources.copy(),
            'current_prices': self.current_prices.copy(),
            'active_allocations': len(self.active_allocations),
            'pending_requests': {rt.value: len(reqs) for rt, reqs in self.pending_requests.items()},
            'current_utilization': current_utilization,
            'market_stats': self.market_stats.copy(),
            'total_revenue': self.market_stats['total_revenue']  # Add for convenience
        }


class ResourceScheduler:
    """
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
    """
    
    def __init__(
        self,
        attention_system: ECANAttentionSystem,
        market: AttentionMarket,
        max_concurrent_tasks: int = 10,
        scheduling_quantum: float = 0.1
    ):
        self.attention_system = attention_system
        self.market = market
        self.max_concurrent_tasks = max_concurrent_tasks
        self.scheduling_quantum = scheduling_quantum
        
        # Task management
        self.task_queue = PriorityQueue()
        self.running_tasks: Dict[str, Any] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        
        # Scheduler state
        self.scheduler_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # Performance metrics
        self.scheduling_stats = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'average_wait_time': 0.0,
            'average_execution_time': 0.0,
            'missed_deadlines': 0
        }
    
    def schedule_task(
        self,
        task_id: str,
        task_function: Callable,
        attention_atoms: List[str],
        resource_requirements: Dict[ResourceType, float],
        deadline: Optional[float] = None,
        priority: float = 1.0
    ) -> bool:
        """Schedule a cognitive task with attention-driven priority."""
        
        # Calculate attention-based priority
        attention_priority = 0.0
        for atom_id in attention_atoms:
            av = self.attention_system.get_attention_value(atom_id)
            attention_priority += av.total_importance
        
        # Combine explicit priority with attention priority
        total_priority = priority + attention_priority * 0.1
        
        # Create task descriptor
        task = {
            'task_id': task_id,
            'function': task_function,
            'attention_atoms': attention_atoms,
            'resource_requirements': resource_requirements,
            'deadline': deadline,
            'priority': total_priority,
            'submit_time': time.time(),
            'status': 'queued'
        }
        
        # Submit to scheduler queue
        self.task_queue.put((-total_priority, time.time(), task))
        self.scheduling_stats['tasks_scheduled'] += 1
        
        return True
    
    def start_scheduler(self):
        """Start the real-time task scheduler."""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the task scheduler."""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1.0)
    
    def _scheduler_loop(self):
        """Main scheduler loop for task execution."""
        while self.scheduler_running:
            try:
                # Process pending tasks
                self._process_task_queue()
                
                # Monitor running tasks
                self._monitor_running_tasks()
                
                # Cleanup and maintenance
                self.market.cleanup_expired_allocations()
                
                # Wait for next quantum
                time.sleep(self.scheduling_quantum)
                
            except Exception as e:
                print(f"Scheduler error: {e}")
    
    def _process_task_queue(self):
        """Process tasks from the scheduler queue."""
        while (not self.task_queue.empty() and 
               len(self.running_tasks) < self.max_concurrent_tasks):
            
            try:
                _, _, task = self.task_queue.get_nowait()
                
                # Check deadline
                current_time = time.time()
                if task['deadline'] and current_time > task['deadline']:
                    self.scheduling_stats['missed_deadlines'] += 1
                    continue
                
                # Request resources
                resource_requests = []
                for resource_type, amount in task['resource_requirements'].items():
                    request = ResourceRequest(
                        request_id=f"{task['task_id']}_{resource_type.value}",
                        resource_type=resource_type,
                        amount=amount,
                        max_bid=task['priority'] * amount,
                        priority=task['priority'],
                        deadline=task['deadline']
                    )
                    resource_requests.append(request)
                
                # Submit resource requests
                all_submitted = True
                for request in resource_requests:
                    if not self.market.submit_request(request):
                        all_submitted = False
                        break
                
                if all_submitted:
                    # Start task execution
                    task['status'] = 'running'
                    task['start_time'] = current_time
                    task['resource_requests'] = resource_requests
                    self.running_tasks[task['task_id']] = task
                    
                    # Execute task in separate thread
                    task_thread = threading.Thread(
                        target=self._execute_task,
                        args=(task,)
                    )
                    task_thread.daemon = True
                    task_thread.start()
                
            except Exception as e:
                print(f"Task processing error: {e}")
    
    def _execute_task(self, task: Dict[str, Any]):
        """Execute a scheduled task."""
        try:
            # Stimulate attention atoms
            for atom_id in task['attention_atoms']:
                self.attention_system.stimulate_atom(atom_id, 1.0)
            
            # Execute task function
            start_time = time.time()
            result = task['function']()
            execution_time = time.time() - start_time
            
            # Record completion
            task['status'] = 'completed'
            task['result'] = result
            task['execution_time'] = execution_time
            task['completion_time'] = time.time()
            
            # Update statistics
            self.scheduling_stats['tasks_completed'] += 1
            wait_time = task['start_time'] - task['submit_time']
            self._update_average_stats(wait_time, execution_time)
            
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
        
        finally:
            # Move task to completed list
            if task['task_id'] in self.running_tasks:
                del self.running_tasks[task['task_id']]
                self.completed_tasks.append(task)
                
                # Limit completed task history
                if len(self.completed_tasks) > 1000:
                    self.completed_tasks = self.completed_tasks[-500:]
    
    def _monitor_running_tasks(self):
        """Monitor running tasks for deadline violations and resource cleanup."""
        current_time = time.time()
        
        for task_id, task in list(self.running_tasks.items()):
            # Check for deadline violations
            if task['deadline'] and current_time > task['deadline']:
                task['status'] = 'deadline_missed'
                del self.running_tasks[task_id]
                self.completed_tasks.append(task)
                self.scheduling_stats['missed_deadlines'] += 1
    
    def _update_average_stats(self, wait_time: float, execution_time: float):
        """Update running average statistics."""
        completed = self.scheduling_stats['tasks_completed']
        if completed > 1:
            # Update running averages
            prev_avg_wait = self.scheduling_stats['average_wait_time']
            prev_avg_exec = self.scheduling_stats['average_execution_time']
            
            self.scheduling_stats['average_wait_time'] = (
                (prev_avg_wait * (completed - 1) + wait_time) / completed
            )
            self.scheduling_stats['average_execution_time'] = (
                (prev_avg_exec * (completed - 1) + execution_time) / completed
            )
        else:
            self.scheduling_stats['average_wait_time'] = wait_time
            self.scheduling_stats['average_execution_time'] = execution_time
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status and performance metrics."""
        return {
            'scheduler_running': self.scheduler_running,
            'queued_tasks': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'scheduling_stats': self.scheduling_stats.copy(),
            'market_status': self.market.get_market_status()
        }


class AttentionBank:
    """
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
    """
    
    def __init__(
        self,
        initial_reserves: float = 1000.0,
        base_interest_rate: float = 0.05,
        max_loan_ratio: float = 0.8
    ):
        self.reserves = initial_reserves
        self.base_interest_rate = base_interest_rate
        self.max_loan_ratio = max_loan_ratio
        
        # Loan tracking
        self.active_loans: Dict[str, Dict[str, Any]] = {}
        self.loan_history: List[Dict[str, Any]] = {}
        
        # Banking statistics
        self.banking_stats = {
            'total_loans_issued': 0,
            'total_loans_repaid': 0,
            'total_defaults': 0,
            'total_interest_earned': 0.0,
            'current_loan_portfolio': 0.0
        }
    
    def request_loan(
        self,
        borrower_id: str,
        loan_amount: float,
        collateral_atoms: List[str],
        attention_system: ECANAttentionSystem,
        term_length: float = 100.0
    ) -> Optional[str]:
        """Request an attention loan with atom collateral."""
        
        # Calculate collateral value
        collateral_value = 0.0
        for atom_id in collateral_atoms:
            av = attention_system.get_attention_value(atom_id)
            collateral_value += av.total_importance
        
        # Check loan eligibility
        max_loan = collateral_value * self.max_loan_ratio
        if loan_amount > max_loan:
            return None
        
        if loan_amount > self.reserves:
            return None
        
        # Calculate interest rate based on risk
        risk_factor = loan_amount / collateral_value if collateral_value > 0 else 1.0
        interest_rate = self.base_interest_rate * (1.0 + risk_factor)
        
        # Create loan
        loan_id = str(uuid.uuid4())
        loan = {
            'loan_id': loan_id,
            'borrower_id': borrower_id,
            'principal': loan_amount,
            'interest_rate': interest_rate,
            'collateral_atoms': collateral_atoms,
            'collateral_value': collateral_value,
            'term_length': term_length,
            'issue_time': time.time(),
            'due_time': time.time() + term_length,
            'remaining_balance': loan_amount * (1.0 + interest_rate),
            'status': 'active'
        }
        
        # Record loan and update reserves
        self.active_loans[loan_id] = loan
        self.reserves -= loan_amount
        self.banking_stats['total_loans_issued'] += 1
        self.banking_stats['current_loan_portfolio'] += loan['remaining_balance']
        
        return loan_id
    
    def make_payment(
        self,
        loan_id: str,
        payment_amount: float,
        attention_system: ECANAttentionSystem
    ) -> bool:
        """Make a payment on an active loan."""
        
        if loan_id not in self.active_loans:
            return False
        
        loan = self.active_loans[loan_id]
        if loan['status'] != 'active':
            return False
        
        # Process payment
        payment_applied = min(payment_amount, loan['remaining_balance'])
        loan['remaining_balance'] -= payment_applied
        self.reserves += payment_applied
        
        # Check if loan is fully repaid
        if loan['remaining_balance'] <= 0.01:  # Small tolerance for floating point
            loan['status'] = 'repaid'
            loan['repay_time'] = time.time()
            self.banking_stats['total_loans_repaid'] += 1
            self.banking_stats['current_loan_portfolio'] -= loan['remaining_balance']
            
            # Release collateral (return attention to atoms)
            for atom_id in loan['collateral_atoms']:
                av = attention_system.get_attention_value(atom_id)
                av.update_sti(payment_applied * 0.1)  # Small reward for repayment
        
        return True
    
    def check_defaults(self, attention_system: ECANAttentionSystem):
        """Check for loan defaults and seize collateral if necessary."""
        current_time = time.time()
        defaulted_loans = []
        
        for loan_id, loan in self.active_loans.items():
            if loan['status'] == 'active' and current_time > loan['due_time']:
                # Loan has defaulted
                loan['status'] = 'defaulted'
                loan['default_time'] = current_time
                defaulted_loans.append(loan_id)
                
                # Seize collateral
                seized_value = 0.0
                for atom_id in loan['collateral_atoms']:
                    av = attention_system.get_attention_value(atom_id)
                    seized_amount = min(av.total_importance, loan['remaining_balance'])
                    av.sti = max(0.0, av.sti - seized_amount)
                    seized_value += seized_amount
                
                # Update bank reserves and statistics
                self.reserves += seized_value
                self.banking_stats['total_defaults'] += 1
                self.banking_stats['current_loan_portfolio'] -= loan['remaining_balance']
        
        return defaulted_loans
    
    def get_banking_status(self) -> Dict[str, Any]:
        """Get current banking status and loan portfolio."""
        active_loans_count = sum(1 for loan in self.active_loans.values() 
                                if loan['status'] == 'active')
        
        return {
            'reserves': self.reserves,
            'active_loans': active_loans_count,
            'total_loans': len(self.active_loans),
            'banking_stats': self.banking_stats.copy(),
            'base_interest_rate': self.base_interest_rate,
            'max_loan_ratio': self.max_loan_ratio
        }


class ResourceAllocator:
    """
    Main resource allocation coordinator integrating ECAN attention with economic markets.
    
    Coordinates between attention systems, resource markets, task scheduling,
    and attention banking to provide comprehensive cognitive resource management.
    
    Parameters
    ----------
    attention_system : ECANAttentionSystem
        ECAN attention allocation system
    initial_resources : Dict[ResourceType, float], optional
        Initial resource capacities
    """
    
    def __init__(
        self,
        attention_system: ECANAttentionSystem,
        initial_resources: Optional[Dict[ResourceType, float]] = None
    ):
        self.attention_system = attention_system
        
        # Initialize default resources if not provided
        if initial_resources is None:
            initial_resources = {
                ResourceType.COMPUTATION: 100.0,
                ResourceType.MEMORY: 1000.0,
                ResourceType.ATTENTION: 50.0,
                ResourceType.COMMUNICATION: 20.0,
                ResourceType.STORAGE: 500.0
            }
        
        # Initialize components
        self.market = AttentionMarket(initial_resources)
        self.scheduler = ResourceScheduler(attention_system, self.market)
        self.bank = AttentionBank()
        
        # Start scheduler
        self.scheduler.start_scheduler()
    
    def allocate_resources(
        self,
        task_id: str,
        resource_requirements: Dict[ResourceType, float],
        attention_atoms: List[str],
        priority: float = 1.0,
        deadline: Optional[float] = None
    ) -> bool:
        """
        Allocate resources for a cognitive task.
        
        Integrates attention-based prioritization, economic resource allocation,
        and task scheduling for comprehensive resource management.
        """
        
        # Define task function (placeholder for actual cognitive processing)
        def task_function():
            # Simulate cognitive processing
            time.sleep(0.1)
            return {"status": "completed", "task_id": task_id}
        
        # Schedule task with attention-driven priority
        return self.scheduler.schedule_task(
            task_id=task_id,
            task_function=task_function,
            attention_atoms=attention_atoms,
            resource_requirements=resource_requirements,
            deadline=deadline,
            priority=priority
        )
    
    def run_economic_cycle(self):
        """Run one cycle of economic attention allocation."""
        # Run ECAN economic cycle
        self.attention_system.run_economic_cycle()
        
        # Check for loan defaults
        self.bank.check_defaults(self.attention_system)
        
        # Update market prices based on attention allocation
        for resource_type in ResourceType:
            # Placeholder for attention-driven price updates
            pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status across all components."""
        return {
            'attention_stats': self.attention_system.get_attention_statistics(),
            'market_status': self.market.get_market_status(),
            'scheduler_status': self.scheduler.get_scheduler_status(),
            'banking_status': self.bank.get_banking_status()
        }
    
    def shutdown(self):
        """Shutdown the resource allocation system."""
        self.scheduler.stop_scheduler()