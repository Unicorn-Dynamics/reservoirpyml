"""
Unity3D Interface for Cognitive Mesh

Provides JSON/HTTP-based communication interface for Unity3D cognitive agents.
Unity3D agents can communicate with the cognitive mesh via REST API calls.
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import numpy as np

from ..api_server import CognitiveMeshAPI


@dataclass
class Unity3DMessage:
    """Standard message format for Unity3D communication."""
    
    message_type: str  # command, query, response, notification
    agent_id: str
    timestamp: float
    data: Dict[str, Any]
    message_id: Optional[str] = None
    response_to: Optional[str] = None


@dataclass
class Unity3DAgent:
    """Represents a Unity3D cognitive agent."""
    
    agent_id: str
    scene_name: str
    transform: Dict[str, Any]  # position, rotation, scale
    components: List[str]  # cognitive components
    status: str = "active"
    last_update: float = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = time.time()


class Unity3DInterface:
    """
    Interface for Unity3D cognitive agents.
    
    Provides standardized communication protocols for Unity3D agents
    to interact with the distributed cognitive mesh via HTTP/JSON.
    """
    
    def __init__(self, mesh_api: CognitiveMeshAPI):
        self.mesh_api = mesh_api
        self.unity_agents: Dict[str, Unity3DAgent] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.agent_lock = threading.Lock()
        
        # Message tracking
        self.message_counter = 0
        self.pending_responses: Dict[str, Unity3DMessage] = {}
        
        # Setup default message handlers
        self._setup_message_handlers()
    
    def _setup_message_handlers(self):
        """Setup default message handlers for Unity3D communication."""
        self.message_handlers.update({
            "register_agent": self._handle_register_agent,
            "unregister_agent": self._handle_unregister_agent,
            "update_transform": self._handle_update_transform,
            "cognitive_process": self._handle_cognitive_process,
            "query_state": self._handle_query_state,
            "allocate_attention": self._handle_allocate_attention,
            "spawn_cognitive_agent": self._handle_spawn_cognitive_agent,
            "get_mesh_status": self._handle_get_mesh_status
        })
    
    def register_unity_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a Unity3D agent with the cognitive mesh."""
        try:
            agent_id = agent_data.get("agent_id", f"unity_agent_{int(time.time())}")
            scene_name = agent_data.get("scene_name", "DefaultScene")
            transform = agent_data.get("transform", {
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "scale": {"x": 1, "y": 1, "z": 1}
            })
            components = agent_data.get("components", [])
            
            # Create Unity3D agent record
            unity_agent = Unity3DAgent(
                agent_id=agent_id,
                scene_name=scene_name,
                transform=transform,
                components=components
            )
            
            with self.agent_lock:
                self.unity_agents[agent_id] = unity_agent
            
            # Register with mesh API
            mesh_response = self.mesh_api.spawn_agent({
                "type": "unity3d",
                "embodiment": "unity3d",
                "config": {
                    "scene_name": scene_name,
                    "transform": transform,
                    "components": components
                }
            })
            
            return {
                "status": "registered",
                "agent_id": agent_id,
                "mesh_agent_id": mesh_response.get("agent_id"),
                "scene_name": scene_name,
                "cognitive_capabilities": [
                    "spatial_reasoning",
                    "object_recognition", 
                    "scene_understanding",
                    "behavioral_control"
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def unregister_unity_agent(self, agent_id: str) -> Dict[str, Any]:
        """Unregister a Unity3D agent."""
        try:
            with self.agent_lock:
                if agent_id not in self.unity_agents:
                    return {
                        "status": "error",
                        "error": f"Agent {agent_id} not found"
                    }
                
                # Remove from Unity agents
                del self.unity_agents[agent_id]
            
            # Terminate in mesh
            mesh_response = self.mesh_api.terminate_agent({"agent_id": agent_id})
            
            return {
                "status": "unregistered",
                "agent_id": agent_id,
                "mesh_response": mesh_response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_unity_cognitive_input(self, agent_id: str, 
                                    cognitive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process cognitive input from Unity3D agent."""
        try:
            # Validate agent
            with self.agent_lock:
                if agent_id not in self.unity_agents:
                    return {
                        "status": "error",
                        "error": f"Unity agent {agent_id} not registered"
                    }
                
                # Update agent activity
                self.unity_agents[agent_id].last_update = time.time()
            
            # Process through mesh API
            mesh_request = {
                "input": cognitive_data.get("sensory_input", []),
                "agent_id": agent_id,
                "mode": "neural_symbolic"
            }
            
            mesh_response = self.mesh_api.process_cognitive_input(mesh_request)
            
            # Convert to Unity3D format
            unity_response = self._convert_to_unity_format(mesh_response, agent_id)
            
            return unity_response
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_unity_transform(self, agent_id: str, 
                             transform_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update Unity3D agent transform."""
        try:
            with self.agent_lock:
                if agent_id not in self.unity_agents:
                    return {
                        "status": "error",
                        "error": f"Agent {agent_id} not found"
                    }
                
                # Update transform
                agent = self.unity_agents[agent_id]
                agent.transform.update(transform_data)
                agent.last_update = time.time()
            
            return {
                "status": "updated",
                "agent_id": agent_id,
                "transform": transform_data
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e)
            }
    
    def get_unity_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of Unity3D agent."""
        try:
            with self.agent_lock:
                if agent_id not in self.unity_agents:
                    return {
                        "status": "error",
                        "error": f"Agent {agent_id} not found"
                    }
                
                agent = self.unity_agents[agent_id]
                
                return {
                    "status": "active",
                    "agent_id": agent_id,
                    "scene_name": agent.scene_name,
                    "transform": agent.transform,
                    "components": agent.components,
                    "last_update": agent.last_update,
                    "mesh_status": self.mesh_api.get_active_agents()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_unity_scene_agents(self, scene_name: str) -> Dict[str, Any]:
        """Get all Unity3D agents in a specific scene."""
        try:
            with self.agent_lock:
                scene_agents = {
                    agent_id: {
                        "agent_id": agent_id,
                        "transform": agent.transform,
                        "components": agent.components,
                        "status": agent.status,
                        "last_update": agent.last_update
                    }
                    for agent_id, agent in self.unity_agents.items()
                    if agent.scene_name == scene_name
                }
            
            return {
                "status": "success",
                "scene_name": scene_name,
                "agent_count": len(scene_agents),
                "agents": scene_agents
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def handle_unity_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message from Unity3D."""
        try:
            # Parse message
            message = Unity3DMessage(
                message_type=message_data.get("message_type", "command"),
                agent_id=message_data.get("agent_id", ""),
                timestamp=message_data.get("timestamp", time.time()),
                data=message_data.get("data", {}),
                message_id=message_data.get("message_id"),
                response_to=message_data.get("response_to")
            )
            
            # Get handler
            command = message.data.get("command", "unknown")
            handler = self.message_handlers.get(command)
            
            if handler:
                response = handler(message)
            else:
                response = {
                    "status": "error",
                    "error": f"Unknown command: {command}"
                }
            
            # Create response message
            self.message_counter += 1
            response_message = {
                "message_type": "response",
                "agent_id": message.agent_id,
                "timestamp": time.time(),
                "data": response,
                "message_id": f"msg_{self.message_counter}",
                "response_to": message.message_id
            }
            
            return response_message
            
        except Exception as e:
            return {
                "message_type": "response",
                "agent_id": message_data.get("agent_id", "unknown"),
                "timestamp": time.time(),
                "data": {
                    "status": "error",
                    "error": str(e)
                },
                "message_id": f"error_{int(time.time())}"
            }
    
    def _convert_to_unity_format(self, mesh_response: Dict[str, Any], 
                                agent_id: str) -> Dict[str, Any]:
        """Convert mesh response to Unity3D format."""
        return {
            "status": "processed",
            "agent_id": agent_id,
            "cognitive_output": {
                "motor_commands": self._extract_motor_commands(mesh_response),
                "attention_focus": self._extract_attention_focus(mesh_response),
                "behavioral_state": self._extract_behavioral_state(mesh_response),
                "spatial_reasoning": self._extract_spatial_reasoning(mesh_response)
            },
            "mesh_coordination": mesh_response.get("mesh_coordination", {}),
            "processing_time": time.time(),
            "unity_compatible": True
        }
    
    def _extract_motor_commands(self, mesh_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract motor commands for Unity3D from mesh response."""
        output_data = mesh_response.get("output_data", [])
        
        if isinstance(output_data, list) and len(output_data) > 0:
            # Convert neural output to motor commands
            if isinstance(output_data[0], list) and len(output_data[0]) >= 6:
                return {
                    "movement": {
                        "forward": float(output_data[0][0]) if len(output_data[0]) > 0 else 0.0,
                        "right": float(output_data[0][1]) if len(output_data[0]) > 1 else 0.0,
                        "up": float(output_data[0][2]) if len(output_data[0]) > 2 else 0.0
                    },
                    "rotation": {
                        "yaw": float(output_data[0][3]) if len(output_data[0]) > 3 else 0.0,
                        "pitch": float(output_data[0][4]) if len(output_data[0]) > 4 else 0.0,
                        "roll": float(output_data[0][5]) if len(output_data[0]) > 5 else 0.0
                    }
                }
        
        return {
            "movement": {"forward": 0.0, "right": 0.0, "up": 0.0},
            "rotation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        }
    
    def _extract_attention_focus(self, mesh_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attention focus for Unity3D visualization."""
        attention_data = mesh_response.get("mesh_coordination", {})
        
        return {
            "focus_point": {"x": 0.0, "y": 0.0, "z": 1.0},  # Default forward
            "attention_strength": 1.0,
            "focus_objects": [],
            "spatial_priority": "forward"
        }
    
    def _extract_behavioral_state(self, mesh_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral state for Unity3D agents."""
        return {
            "state": "exploring",
            "confidence": 0.8,
            "energy_level": 1.0,
            "social_tendency": 0.5
        }
    
    def _extract_spatial_reasoning(self, mesh_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract spatial reasoning for Unity3D navigation."""
        return {
            "obstacle_detected": False,
            "path_planning": {"waypoints": []},
            "spatial_memory": {"landmarks": []},
            "navigation_confidence": 0.8
        }
    
    # Message handlers
    def _handle_register_agent(self, message: Unity3DMessage) -> Dict[str, Any]:
        """Handle agent registration."""
        return self.register_unity_agent(message.data)
    
    def _handle_unregister_agent(self, message: Unity3DMessage) -> Dict[str, Any]:
        """Handle agent unregistration."""
        return self.unregister_unity_agent(message.agent_id)
    
    def _handle_update_transform(self, message: Unity3DMessage) -> Dict[str, Any]:
        """Handle transform update."""
        return self.update_unity_transform(message.agent_id, message.data.get("transform", {}))
    
    def _handle_cognitive_process(self, message: Unity3DMessage) -> Dict[str, Any]:
        """Handle cognitive processing request."""
        return self.process_unity_cognitive_input(message.agent_id, message.data)
    
    def _handle_query_state(self, message: Unity3DMessage) -> Dict[str, Any]:
        """Handle state query."""
        return self.get_unity_agent_status(message.agent_id)
    
    def _handle_allocate_attention(self, message: Unity3DMessage) -> Dict[str, Any]:
        """Handle attention allocation."""
        return self.mesh_api.allocate_attention(message.data)
    
    def _handle_spawn_cognitive_agent(self, message: Unity3DMessage) -> Dict[str, Any]:
        """Handle cognitive agent spawning."""
        spawn_data = message.data.copy()
        spawn_data.update({"type": "unity3d", "embodiment": "unity3d"})
        return self.mesh_api.spawn_agent(spawn_data)
    
    def _handle_get_mesh_status(self, message: Unity3DMessage) -> Dict[str, Any]:
        """Handle mesh status query."""
        return self.mesh_api.get_mesh_status()