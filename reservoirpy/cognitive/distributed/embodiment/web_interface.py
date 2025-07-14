"""
Web Interface for Cognitive Mesh

Provides WebSocket and HTTP interfaces for web-based cognitive agents.
Enables real-time bi-directional communication with browser-based agents.
"""

import json
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
import numpy as np

from ..api_server import CognitiveMeshAPI


@dataclass
class WebSocketMessage:
    """Standard WebSocket message format."""
    
    type: str           # connect, disconnect, cognitive_input, motor_output, status
    agent_id: str
    timestamp: float
    data: Dict[str, Any]
    session_id: Optional[str] = None


@dataclass
class WebAgent:
    """Represents a web-based cognitive agent."""
    
    agent_id: str
    session_id: str
    agent_type: str = "web_agent"
    capabilities: Set[str] = None
    connection_time: float = None
    last_activity: float = None
    status: str = "connected"
    user_agent: str = ""
    ip_address: str = ""
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = {"visual_processing", "user_interaction", "data_visualization"}
        if self.connection_time is None:
            self.connection_time = time.time()
        if self.last_activity is None:
            self.last_activity = time.time()


class WebSocketConnection:
    """Simulated WebSocket connection for web agents."""
    
    def __init__(self, agent_id: str, session_id: str):
        self.agent_id = agent_id
        self.session_id = session_id
        self.connected = True
        self.message_queue: List[Dict[str, Any]] = []
        self.last_ping = time.time()
    
    def send_message(self, message: Dict[str, Any]):
        """Send message to web client (simulated)."""
        if self.connected:
            self.message_queue.append({
                "timestamp": time.time(),
                "message": message
            })
    
    def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from web client (simulated)."""
        if self.message_queue:
            return self.message_queue.pop(0)["message"]
        return None
    
    def close(self):
        """Close WebSocket connection."""
        self.connected = False


class WebInterface:
    """
    Interface for web-based cognitive agents.
    
    Provides WebSocket and HTTP endpoints for browser-based agents
    to interact with the distributed cognitive mesh in real-time.
    """
    
    def __init__(self, mesh_api: CognitiveMeshAPI):
        self.mesh_api = mesh_api
        self.web_agents: Dict[str, WebAgent] = {}
        self.websocket_connections: Dict[str, WebSocketConnection] = {}
        self.agent_lock = threading.Lock()
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.broadcast_channels: Dict[str, Set[str]] = {}  # channel -> agent_ids
        
        # Real-time features
        self.real_time_enabled = True
        self.max_message_rate = 60  # messages per second
        self.connection_timeout = 30.0  # seconds
        
        # Performance tracking
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "average_latency": 0.0
        }
        
        # Setup message handlers
        self._setup_message_handlers()
        
        # Start connection monitoring
        self._start_connection_monitoring()
    
    def _setup_message_handlers(self):
        """Setup WebSocket message handlers."""
        self.message_handlers.update({
            "connect": self._handle_connect,
            "disconnect": self._handle_disconnect,
            "cognitive_input": self._handle_cognitive_input,
            "visual_input": self._handle_visual_input,
            "user_interaction": self._handle_user_interaction,
            "status_query": self._handle_status_query,
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            "ping": self._handle_ping
        })
    
    def connect_web_agent(self, connection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Connect a new web-based agent."""
        try:
            agent_id = connection_data.get("agent_id", f"web_agent_{int(time.time())}")
            session_id = connection_data.get("session_id", f"session_{int(time.time())}")
            agent_type = connection_data.get("agent_type", "web_agent")
            capabilities = set(connection_data.get("capabilities", []))
            user_agent = connection_data.get("user_agent", "")
            ip_address = connection_data.get("ip_address", "127.0.0.1")
            
            # Create web agent
            web_agent = WebAgent(
                agent_id=agent_id,
                session_id=session_id,
                agent_type=agent_type,
                capabilities=capabilities,
                user_agent=user_agent,
                ip_address=ip_address
            )
            
            # Create WebSocket connection
            ws_connection = WebSocketConnection(agent_id, session_id)
            
            with self.agent_lock:
                self.web_agents[agent_id] = web_agent
                self.websocket_connections[session_id] = ws_connection
                self.metrics["total_connections"] += 1
                self.metrics["active_connections"] += 1
            
            # Register with mesh API
            mesh_response = self.mesh_api.spawn_agent({
                "type": "web_agent",
                "embodiment": "web",
                "config": {
                    "agent_type": agent_type,
                    "capabilities": list(capabilities),
                    "session_id": session_id
                }
            })
            
            # Send welcome message
            welcome_message = {
                "type": "connection_established",
                "agent_id": agent_id,
                "session_id": session_id,
                "mesh_agent_id": mesh_response.get("agent_id"),
                "capabilities": list(capabilities),
                "available_channels": list(self.broadcast_channels.keys()),
                "mesh_status": self.mesh_api.get_mesh_status()
            }
            
            ws_connection.send_message(welcome_message)
            
            return {
                "status": "connected",
                "agent_id": agent_id,
                "session_id": session_id,
                "mesh_agent_id": mesh_response.get("agent_id")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def disconnect_web_agent(self, agent_id: str, session_id: str) -> Dict[str, Any]:
        """Disconnect a web-based agent."""
        try:
            with self.agent_lock:
                # Remove from agents
                if agent_id in self.web_agents:
                    del self.web_agents[agent_id]
                
                # Close WebSocket connection
                if session_id in self.websocket_connections:
                    self.websocket_connections[session_id].close()
                    del self.websocket_connections[session_id]
                    self.metrics["active_connections"] -= 1
                
                # Remove from broadcast channels
                for channel_agents in self.broadcast_channels.values():
                    channel_agents.discard(agent_id)
            
            # Terminate in mesh
            mesh_response = self.mesh_api.terminate_agent({"agent_id": agent_id})
            
            return {
                "status": "disconnected",
                "agent_id": agent_id,
                "session_id": session_id,
                "mesh_response": mesh_response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_web_cognitive_input(self, agent_id: str, 
                                  cognitive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process cognitive input from web agent."""
        try:
            # Validate agent
            with self.agent_lock:
                if agent_id not in self.web_agents:
                    return {
                        "status": "error",
                        "error": f"Web agent {agent_id} not connected"
                    }
                
                # Update activity
                self.web_agents[agent_id].last_activity = time.time()
                self.metrics["messages_received"] += 1
            
            # Convert web input to mesh format
            mesh_input = self._convert_web_input_to_mesh(cognitive_data)
            
            # Process through mesh API
            mesh_request = {
                "input": mesh_input,
                "agent_id": agent_id,
                "mode": "neural_symbolic"
            }
            
            start_time = time.time()
            mesh_response = self.mesh_api.process_cognitive_input(mesh_request)
            processing_time = time.time() - start_time
            
            # Update latency metrics
            self._update_latency_metrics(processing_time)
            
            # Convert response to web format
            web_response = self._convert_mesh_output_to_web(mesh_response, agent_id)
            
            # Send response via WebSocket
            if agent_id in self.web_agents:
                session_id = self.web_agents[agent_id].session_id
                if session_id in self.websocket_connections:
                    self.websocket_connections[session_id].send_message(web_response)
                    self.metrics["messages_sent"] += 1
            
            return {
                "status": "processed",
                "agent_id": agent_id,
                "processing_time": processing_time,
                "response_sent": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def send_visual_feedback(self, agent_id: str, 
                           visual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send visual feedback to web agent."""
        try:
            with self.agent_lock:
                if agent_id not in self.web_agents:
                    return {
                        "status": "error",
                        "error": f"Agent {agent_id} not found"
                    }
                
                session_id = self.web_agents[agent_id].session_id
                
                if session_id not in self.websocket_connections:
                    return {
                        "status": "error",
                        "error": f"WebSocket connection not found"
                    }
            
            # Create visual message
            visual_message = {
                "type": "visual_feedback",
                "agent_id": agent_id,
                "timestamp": time.time(),
                "data": {
                    "visualization_type": visual_data.get("type", "cognitive_state"),
                    "content": visual_data.get("content", {}),
                    "rendering_hints": visual_data.get("hints", {}),
                    "interactive": visual_data.get("interactive", False)
                }
            }
            
            # Send message
            self.websocket_connections[session_id].send_message(visual_message)
            self.metrics["messages_sent"] += 1
            
            return {
                "status": "sent",
                "agent_id": agent_id,
                "message_type": "visual_feedback"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def broadcast_to_channel(self, channel: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast message to all agents in a channel."""
        try:
            if channel not in self.broadcast_channels:
                return {
                    "status": "error",
                    "error": f"Channel {channel} does not exist"
                }
            
            agents_in_channel = self.broadcast_channels[channel].copy()
            sent_count = 0
            
            broadcast_message = {
                "type": "broadcast",
                "channel": channel,
                "timestamp": time.time(),
                "data": message
            }
            
            with self.agent_lock:
                for agent_id in agents_in_channel:
                    if agent_id in self.web_agents:
                        session_id = self.web_agents[agent_id].session_id
                        if session_id in self.websocket_connections:
                            self.websocket_connections[session_id].send_message(broadcast_message)
                            sent_count += 1
            
            self.metrics["messages_sent"] += sent_count
            
            return {
                "status": "broadcast",
                "channel": channel,
                "recipients": sent_count,
                "total_in_channel": len(agents_in_channel)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_web_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of web agent."""
        try:
            with self.agent_lock:
                if agent_id not in self.web_agents:
                    return {
                        "status": "error",
                        "error": f"Agent {agent_id} not found"
                    }
                
                agent = self.web_agents[agent_id]
                session_id = agent.session_id
                
                # Check connection status
                connection_active = (
                    session_id in self.websocket_connections and
                    self.websocket_connections[session_id].connected
                )
                
                return {
                    "status": "active" if connection_active else "disconnected",
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "agent_type": agent.agent_type,
                    "capabilities": list(agent.capabilities),
                    "connection_time": agent.connection_time,
                    "last_activity": agent.last_activity,
                    "user_agent": agent.user_agent,
                    "ip_address": agent.ip_address,
                    "connection_active": connection_active,
                    "message_queue_size": len(self.websocket_connections[session_id].message_queue) if connection_active else 0
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_web_interface_metrics(self) -> Dict[str, Any]:
        """Get web interface performance metrics."""
        try:
            with self.agent_lock:
                active_agents = len([
                    agent for agent in self.web_agents.values()
                    if agent.status == "connected"
                ])
                
                return {
                    "status": "active",
                    "metrics": self.metrics.copy(),
                    "active_agents": active_agents,
                    "total_agents": len(self.web_agents),
                    "active_connections": len(self.websocket_connections),
                    "broadcast_channels": {
                        channel: len(agents)
                        for channel, agents in self.broadcast_channels.items()
                    },
                    "real_time_enabled": self.real_time_enabled
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def handle_websocket_message(self, session_id: str, 
                                message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming WebSocket message."""
        try:
            # Parse message
            message = WebSocketMessage(
                type=message_data.get("type", "unknown"),
                agent_id=message_data.get("agent_id", ""),
                timestamp=message_data.get("timestamp", time.time()),
                data=message_data.get("data", {}),
                session_id=session_id
            )
            
            # Get handler
            handler = self.message_handlers.get(message.type)
            
            if handler:
                response = handler(message)
            else:
                response = {
                    "status": "error",
                    "error": f"Unknown message type: {message.type}"
                }
            
            return response
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _convert_web_input_to_mesh(self, cognitive_data: Dict[str, Any]) -> List[float]:
        """Convert web cognitive input to mesh format."""
        input_vector = []
        
        # Process different types of web input
        if "visual_input" in cognitive_data:
            visual_data = cognitive_data["visual_input"]
            if isinstance(visual_data, list):
                input_vector.extend([float(v) for v in visual_data])
            elif isinstance(visual_data, dict):
                # Extract numerical features from visual data
                for key, value in visual_data.items():
                    if isinstance(value, (int, float)):
                        input_vector.append(float(value))
        
        if "user_input" in cognitive_data:
            user_data = cognitive_data["user_input"]
            if isinstance(user_data, dict):
                # Convert user interactions to numerical features
                for key, value in user_data.items():
                    if isinstance(value, (int, float)):
                        input_vector.append(float(value))
                    elif isinstance(value, bool):
                        input_vector.append(1.0 if value else 0.0)
        
        if "sensor_data" in cognitive_data:
            sensor_data = cognitive_data["sensor_data"]
            if isinstance(sensor_data, list):
                input_vector.extend([float(v) for v in sensor_data])
        
        # Ensure minimum input size
        if len(input_vector) < 8:
            input_vector.extend([0.0] * (8 - len(input_vector)))
        
        return input_vector
    
    def _convert_mesh_output_to_web(self, mesh_response: Dict[str, Any], 
                                  agent_id: str) -> Dict[str, Any]:
        """Convert mesh response to web-compatible format."""
        output_data = mesh_response.get("output_data", [])
        
        return {
            "type": "cognitive_output",
            "agent_id": agent_id,
            "timestamp": time.time(),
            "data": {
                "visual_output": self._extract_visual_output(output_data),
                "interaction_suggestions": self._extract_interaction_suggestions(output_data),
                "cognitive_state": self._extract_cognitive_state(mesh_response),
                "attention_focus": self._extract_attention_focus(mesh_response),
                "recommendations": self._extract_recommendations(output_data)
            },
            "mesh_coordination": mesh_response.get("mesh_coordination", {}),
            "processing_metadata": {
                "input_shape": mesh_response.get("input_shape", []),
                "output_shape": mesh_response.get("output_shape", []),
                "processing_mode": mesh_response.get("processing_mode", "neural_symbolic")
            }
        }
    
    def _extract_visual_output(self, output_data: List[Any]) -> Dict[str, Any]:
        """Extract visual output for web display."""
        if isinstance(output_data, list) and len(output_data) > 0:
            if isinstance(output_data[0], list):
                data = output_data[0]
            else:
                data = output_data
            
            return {
                "visualization_type": "neural_activity",
                "data_points": data[:10] if len(data) >= 10 else data,
                "color_intensity": [abs(float(d)) for d in data[:5]] if len(data) >= 5 else [0.5],
                "animation_speed": 1.0
            }
        
        return {
            "visualization_type": "static",
            "data_points": [0.0],
            "color_intensity": [0.0],
            "animation_speed": 0.0
        }
    
    def _extract_interaction_suggestions(self, output_data: List[Any]) -> List[Dict[str, Any]]:
        """Extract interaction suggestions for web UI."""
        suggestions = []
        
        if isinstance(output_data, list) and len(output_data) > 0:
            # Generate suggestions based on output patterns
            suggestions.append({
                "type": "explore",
                "description": "Explore cognitive patterns",
                "confidence": 0.8
            })
            
            suggestions.append({
                "type": "adjust",
                "description": "Adjust input parameters",
                "confidence": 0.6
            })
        
        return suggestions
    
    def _extract_cognitive_state(self, mesh_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cognitive state for web display."""
        return {
            "processing_status": mesh_response.get("status", "unknown"),
            "confidence": 0.8,
            "complexity": 0.6,
            "stability": 0.9,
            "learning_rate": 0.1
        }
    
    def _extract_attention_focus(self, mesh_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attention focus for web visualization."""
        return {
            "primary_focus": "input_processing",
            "secondary_focus": "pattern_recognition",
            "focus_strength": 0.7,
            "focus_distribution": [0.4, 0.3, 0.2, 0.1]
        }
    
    def _extract_recommendations(self, output_data: List[Any]) -> List[str]:
        """Extract recommendations for the user."""
        return [
            "Continue current interaction pattern",
            "Try varying the input complexity",
            "Observe attention allocation patterns"
        ]
    
    def _update_latency_metrics(self, processing_time: float):
        """Update latency metrics."""
        current_avg = self.metrics["average_latency"]
        message_count = self.metrics["messages_received"]
        
        if message_count > 1:
            # Rolling average
            alpha = 0.1  # Smoothing factor
            self.metrics["average_latency"] = (
                current_avg * (1 - alpha) + processing_time * alpha
            )
        else:
            self.metrics["average_latency"] = processing_time
    
    def _start_connection_monitoring(self):
        """Start background connection monitoring."""
        def monitor_loop():
            while True:
                try:
                    self._check_connection_timeouts()
                    time.sleep(10.0)  # Check every 10 seconds
                except Exception as e:
                    print(f"Connection monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _check_connection_timeouts(self):
        """Check for timed out connections."""
        current_time = time.time()
        expired_agents = []
        
        with self.agent_lock:
            for agent_id, agent in self.web_agents.items():
                if current_time - agent.last_activity > self.connection_timeout:
                    expired_agents.append(agent_id)
        
        # Disconnect expired agents
        for agent_id in expired_agents:
            if agent_id in self.web_agents:
                session_id = self.web_agents[agent_id].session_id
                self.disconnect_web_agent(agent_id, session_id)
    
    # Message handlers
    def _handle_connect(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle connection message."""
        return self.connect_web_agent(message.data)
    
    def _handle_disconnect(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle disconnection message."""
        return self.disconnect_web_agent(message.agent_id, message.session_id)
    
    def _handle_cognitive_input(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle cognitive input message."""
        return self.process_web_cognitive_input(message.agent_id, message.data)
    
    def _handle_visual_input(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle visual input message."""
        return self.process_web_cognitive_input(message.agent_id, {
            "visual_input": message.data
        })
    
    def _handle_user_interaction(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle user interaction message."""
        return self.process_web_cognitive_input(message.agent_id, {
            "user_input": message.data
        })
    
    def _handle_status_query(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle status query."""
        return self.get_web_agent_status(message.agent_id)
    
    def _handle_subscribe(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle channel subscription."""
        channel = message.data.get("channel", "")
        if channel not in self.broadcast_channels:
            self.broadcast_channels[channel] = set()
        
        self.broadcast_channels[channel].add(message.agent_id)
        
        return {
            "status": "subscribed",
            "channel": channel,
            "agent_id": message.agent_id
        }
    
    def _handle_unsubscribe(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle channel unsubscription."""
        channel = message.data.get("channel", "")
        if channel in self.broadcast_channels:
            self.broadcast_channels[channel].discard(message.agent_id)
        
        return {
            "status": "unsubscribed",
            "channel": channel,
            "agent_id": message.agent_id
        }
    
    def _handle_ping(self, message: WebSocketMessage) -> Dict[str, Any]:
        """Handle ping message."""
        if message.session_id in self.websocket_connections:
            self.websocket_connections[message.session_id].last_ping = time.time()
        
        return {
            "type": "pong",
            "timestamp": time.time()
        }