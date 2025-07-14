"""
Cognitive Mesh API Server

Provides REST and WebSocket endpoints for distributed cognitive mesh operations
using ReservoirPy neural-symbolic infrastructure.
"""

import json
import asyncio
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Any, Optional, Callable
import numpy as np

from ..ggml.core import GGMLTensor, GGMLContext
from ..ggml.bridge import NeuralSymbolicBridge  
from ..attention.ecan import ECANAttentionSystem, AttentionValue
from .mesh_coordinator import MeshCoordinator


class CognitiveMeshHandler(BaseHTTPRequestHandler):
    """HTTP request handler for cognitive mesh API endpoints."""
    
    def __init__(self, mesh_api, *args):
        self.mesh_api = mesh_api
        super().__init__(*args)
    
    def do_GET(self):
        """Handle GET requests for cognitive mesh status and queries."""
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            params = parse_qs(parsed_url.query)
            
            if path == '/mesh/status':
                response = self.mesh_api.get_mesh_status()
            elif path == '/mesh/nodes':
                response = self.mesh_api.get_mesh_nodes()
            elif path == '/reservoir/state':
                node_id = params.get('node_id', [None])[0]
                response = self.mesh_api.get_reservoir_state(node_id)
            elif path == '/attention/allocation':
                response = self.mesh_api.get_attention_allocation()
            elif path == '/agents':
                response = self.mesh_api.get_active_agents()
            else:
                self._send_error(404, "Endpoint not found")
                return
                
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests for cognitive operations."""
        try:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else '{}'
            request_data = json.loads(body)
            
            if path == '/mesh/process':
                response = self.mesh_api.process_cognitive_input(request_data)
            elif path == '/reservoir/update':
                response = self.mesh_api.update_reservoir_state(request_data)
            elif path == '/attention/allocate':
                response = self.mesh_api.allocate_attention(request_data)
            elif path == '/agents/spawn':
                response = self.mesh_api.spawn_agent(request_data)
            elif path == '/agents/terminate':
                response = self.mesh_api.terminate_agent(request_data)
            elif path == '/tasks/distribute':
                response = self.mesh_api.distribute_task(request_data)
            else:
                self._send_error(404, "Endpoint not found")
                return
                
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def _send_json_response(self, data: Dict[str, Any]):
        """Send JSON response with proper headers."""
        response_data = json.dumps(data, default=self._json_serializer)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response_data.encode('utf-8'))
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        error_response = json.dumps({"error": message, "code": code})
        self.wfile.write(error_response.encode('utf-8'))
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for NumPy arrays and custom objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP server for handling concurrent requests."""
    daemon_threads = True


class CognitiveMeshAPI:
    """
    Main API class for the distributed cognitive mesh.
    
    Provides REST endpoints for:
    - Mesh status and node coordination
    - Reservoir state management and synchronization
    - Attention allocation and focus management
    - Agent orchestration and task distribution
    - Embodiment interface coordination
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.is_running = False
        
        # Initialize cognitive components
        self.context = GGMLContext()
        from ..hypergraph import AtomSpace
        self.atom_space = AtomSpace()
        self.bridge = NeuralSymbolicBridge(self.context, self.atom_space)
        self.attention_allocator = ECANAttentionSystem(self.atom_space)
        self.mesh_coordinator = MeshCoordinator()
        
        # Track active agents and tasks
        self.active_agents = {}
        self.active_tasks = {}
        self.agent_counter = 0
        self.task_counter = 0
        
        # WebSocket connections (simplified tracking)
        self.websocket_connections = {}
        
    def start_server(self):
        """Start the cognitive mesh API server."""
        if self.is_running:
            return {"status": "already_running", "host": self.host, "port": self.port}
        
        def handler(*args):
            return CognitiveMeshHandler(self, *args)
        
        self.server = ThreadingHTTPServer((self.host, self.port), handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.is_running = True
        
        return {
            "status": "started",
            "host": self.host,
            "port": self.port,
            "endpoints": self._get_available_endpoints()
        }
    
    def stop_server(self):
        """Stop the cognitive mesh API server."""
        if not self.is_running:
            return {"status": "not_running"}
        
        self.server.shutdown()
        self.server_thread.join()
        self.is_running = False
        
        return {"status": "stopped"}
    
    def get_mesh_status(self) -> Dict[str, Any]:
        """Get overall mesh status and statistics."""
        return {
            "status": "active" if self.is_running else "inactive",
            "timestamp": time.time(),
            "mesh_coordinator": self.mesh_coordinator.get_status(),
            "attention_allocator": {
                "active_nodes": len(self.attention_allocator.attention_values),
                "total_sti": sum(av.sti for av in self.attention_allocator.attention_values.values()),
                "allocation_strategy": "ecan"
            },
            "active_agents": len(self.active_agents),
            "active_tasks": len(self.active_tasks),
            "context_stats": {
                "memory_usage": getattr(self.context, 'memory_usage', 0.0),
                "tensor_count": len(getattr(self.context, 'tensors', {})),
                "total_operations": getattr(self.context, 'total_operations', 0)
            }
        }
    
    def get_mesh_nodes(self) -> Dict[str, Any]:
        """Get information about all mesh nodes."""
        return {
            "nodes": self.mesh_coordinator.get_all_nodes(),
            "topology": self.mesh_coordinator.get_topology(),
            "synchronization_status": self.mesh_coordinator.get_sync_status()
        }
    
    def get_reservoir_state(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Get reservoir state for specific node or all nodes."""
        if node_id:
            return self.mesh_coordinator.get_node_state(node_id)
        else:
            return {
                "all_nodes": self.mesh_coordinator.get_all_states(),
                "global_state": self.mesh_coordinator.get_global_state()
            }
    
    def get_attention_allocation(self) -> Dict[str, Any]:
        """Get current attention allocation across the mesh."""
        return {
            "allocation_map": {k: v.sti for k, v in self.attention_allocator.attention_values.items()},
            "focus_nodes": self.attention_allocator.get_attentional_focus_nodes(top_k=5),
            "spreading_activation": {"status": "active"},
            "rent_collection": {"status": "active"}
        }
    
    def get_active_agents(self) -> Dict[str, Any]:
        """Get information about active cognitive agents."""
        return {
            "agents": {
                agent_id: {
                    "type": agent_info["type"],
                    "status": agent_info["status"],
                    "created_at": agent_info["created_at"],
                    "last_activity": agent_info.get("last_activity", "never"),
                    "embodiment": agent_info.get("embodiment", "none")
                }
                for agent_id, agent_info in self.active_agents.items()
            },
            "total_agents": len(self.active_agents)
        }
    
    def process_cognitive_input(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process cognitive input through the mesh."""
        try:
            # Extract input data
            input_data = request_data.get("input", [])
            agent_id = request_data.get("agent_id", "default")
            processing_mode = request_data.get("mode", "neural_symbolic")
            
            # Convert to tensor
            if isinstance(input_data, list):
                input_array = np.array(input_data, dtype=np.float32)
            else:
                input_array = np.array([[input_data]], dtype=np.float32)
            
            # Create GGML tensor
            input_tensor = GGMLTensor(
                data=input_array,
                name=f"mesh_input_{agent_id}",
                tensor_type="neural"
            )
            
            # Process through neural-symbolic bridge
            if processing_mode == "neural_symbolic":
                symbolic_tensor = self.bridge.neural_to_symbolic(input_tensor)
                output_tensor = self.bridge.symbolic_to_neural(symbolic_tensor)
            else:
                output_tensor = input_tensor
            
            # Update attention allocation
            self.attention_allocator.allocate_attention(output_tensor.name, 1.0)
            
            # Coordinate with mesh
            mesh_result = self.mesh_coordinator.process_distributed_input(
                output_tensor, agent_id
            )
            
            return {
                "status": "processed",
                "agent_id": agent_id,
                "input_shape": list(input_tensor.data.shape),
                "output_shape": list(output_tensor.data.shape),
                "output_data": output_tensor.data.tolist(),
                "processing_mode": processing_mode,
                "mesh_coordination": mesh_result,
                "attention_allocated": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_reservoir_state(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update reservoir state across the mesh."""
        try:
            node_id = request_data.get("node_id", "default")
            state_data = request_data.get("state", [])
            
            # Convert state data to numpy array
            if isinstance(state_data, list):
                state_array = np.array(state_data, dtype=np.float32)
            else:
                state_array = np.array([[state_data]], dtype=np.float32)
            
            # Update mesh coordinator
            update_result = self.mesh_coordinator.update_node_state(node_id, state_array)
            
            return {
                "status": "updated",
                "node_id": node_id,
                "state_shape": list(state_array.shape),
                "mesh_sync": update_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def allocate_attention(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate attention to specific nodes or concepts."""
        try:
            target = request_data.get("target", "")
            sti_amount = request_data.get("sti", 1.0)
            
            # Allocate attention
            self.attention_allocator.allocate_attention(target, sti_amount)
            
            return {
                "status": "allocated",
                "target": target,
                "sti_amount": sti_amount,
                "new_total_sti": self.attention_allocator.attention_values.get(target, AttentionValue()).sti
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e)
            }
    
    def spawn_agent(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn a new cognitive agent."""
        try:
            agent_type = request_data.get("type", "generic")
            embodiment = request_data.get("embodiment", "none")
            config = request_data.get("config", {})
            
            # Generate new agent ID
            self.agent_counter += 1
            agent_id = f"agent_{self.agent_counter}"
            
            # Create agent record
            agent_info = {
                "type": agent_type,
                "embodiment": embodiment,
                "config": config,
                "status": "active",
                "created_at": time.time(),
                "last_activity": time.time()
            }
            
            self.active_agents[agent_id] = agent_info
            
            return {
                "status": "spawned",
                "agent_id": agent_id,
                "agent_type": agent_type,
                "embodiment": embodiment
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def terminate_agent(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Terminate a cognitive agent."""
        try:
            agent_id = request_data.get("agent_id", "")
            
            if agent_id not in self.active_agents:
                return {
                    "status": "error",
                    "error": f"Agent {agent_id} not found"
                }
            
            # Remove agent
            del self.active_agents[agent_id]
            
            return {
                "status": "terminated",
                "agent_id": agent_id
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def distribute_task(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute a task across the cognitive mesh."""
        try:
            task_data = request_data.get("task", {})
            priority = request_data.get("priority", 1.0)
            target_agents = request_data.get("target_agents", [])
            
            # Generate new task ID
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"
            
            # Create task record
            task_info = {
                "data": task_data,
                "priority": priority,
                "target_agents": target_agents,
                "status": "distributed",
                "created_at": time.time()
            }
            
            self.active_tasks[task_id] = task_info
            
            # Use mesh coordinator to distribute
            distribution_result = self.mesh_coordinator.distribute_task(
                task_id, task_data, target_agents
            )
            
            return {
                "status": "distributed",
                "task_id": task_id,
                "priority": priority,
                "target_agents": target_agents,
                "distribution_result": distribution_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_available_endpoints(self) -> List[str]:
        """Get list of available API endpoints."""
        return [
            "GET /mesh/status",
            "GET /mesh/nodes", 
            "GET /reservoir/state",
            "GET /attention/allocation",
            "GET /agents",
            "POST /mesh/process",
            "POST /reservoir/update",
            "POST /attention/allocate",
            "POST /agents/spawn",
            "POST /agents/terminate",
            "POST /tasks/distribute"
        ]