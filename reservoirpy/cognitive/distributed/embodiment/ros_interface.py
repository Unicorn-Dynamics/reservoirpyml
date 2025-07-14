"""
ROS Interface for Cognitive Mesh

Provides ROS-compatible communication interface for robotics platforms.
Uses standard message formats and service patterns.
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from ..api_server import CognitiveMeshAPI


@dataclass
class ROSMessage:
    """Standard ROS message format for cognitive mesh communication."""
    
    header: Dict[str, Any]  # timestamp, frame_id, seq
    message_type: str       # cognitive_input, motor_output, status_query
    agent_id: str
    data: Dict[str, Any]


@dataclass  
class ROSCognitiveInput:
    """ROS message for cognitive input data."""
    
    header: Dict[str, Any]
    sensor_data: Dict[str, List[float]]  # sensor readings
    perception_data: Dict[str, Any]      # processed perceptions
    goal_state: Dict[str, Any]           # current goals
    context: Dict[str, Any]              # environmental context


@dataclass
class ROSMotorOutput:
    """ROS message for motor control output."""
    
    header: Dict[str, Any]
    linear_velocity: Dict[str, float]    # x, y, z
    angular_velocity: Dict[str, float]   # x, y, z
    joint_commands: List[float]          # joint positions/velocities
    gripper_commands: Dict[str, float]   # gripper controls
    safety_constraints: Dict[str, Any]   # safety limits


@dataclass
class ROSRobot:
    """Represents a ROS-based robotic agent."""
    
    agent_id: str
    robot_type: str          # mobile_robot, manipulator, humanoid
    namespace: str           # ROS namespace
    capabilities: List[str]  # navigation, manipulation, vision
    joint_states: Dict[str, float]
    sensor_states: Dict[str, Any]
    status: str = "active"
    last_update: float = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = time.time()


class ROSInterface:
    """
    Interface for ROS-based cognitive robots.
    
    Provides standardized communication protocols for ROS robots
    to interact with the distributed cognitive mesh.
    """
    
    def __init__(self, mesh_api: CognitiveMeshAPI):
        self.mesh_api = mesh_api
        self.ros_robots: Dict[str, ROSRobot] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.robot_lock = threading.Lock()
        
        # ROS-style message tracking
        self.sequence_counter = 0
        self.published_messages: List[ROSMessage] = []
        
        # Topic simulation (in real ROS, these would be actual topics)
        self.topics = {
            "/cognitive_input": [],
            "/motor_output": [],
            "/robot_status": [],
            "/mesh_status": [],
            "/attention_allocation": []
        }
        
        # Service simulation
        self.services = {
            "register_robot": self._service_register_robot,
            "process_cognitive_input": self._service_process_cognitive,
            "get_robot_status": self._service_get_robot_status,
            "allocate_attention": self._service_allocate_attention,
            "emergency_stop": self._service_emergency_stop
        }
        
        # Setup message handlers
        self._setup_message_handlers()
    
    def _setup_message_handlers(self):
        """Setup ROS message handlers."""
        self.message_handlers.update({
            "cognitive_input": self._handle_cognitive_input,
            "status_query": self._handle_status_query,
            "attention_request": self._handle_attention_request,
            "joint_state": self._handle_joint_state,
            "sensor_data": self._handle_sensor_data,
            "goal_update": self._handle_goal_update
        })
    
    def register_ros_robot(self, robot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a ROS robot with the cognitive mesh."""
        try:
            agent_id = robot_data.get("agent_id", f"ros_robot_{int(time.time())}")
            robot_type = robot_data.get("robot_type", "mobile_robot")
            namespace = robot_data.get("namespace", f"/{agent_id}")
            capabilities = robot_data.get("capabilities", ["navigation"])
            
            # Create ROS robot record
            ros_robot = ROSRobot(
                agent_id=agent_id,
                robot_type=robot_type,
                namespace=namespace,
                capabilities=capabilities,
                joint_states={},
                sensor_states={}
            )
            
            with self.robot_lock:
                self.ros_robots[agent_id] = ros_robot
            
            # Register with mesh API
            mesh_response = self.mesh_api.spawn_agent({
                "type": "ros_robot",
                "embodiment": "robotic",
                "config": {
                    "robot_type": robot_type,
                    "namespace": namespace,
                    "capabilities": capabilities
                }
            })
            
            # Publish registration message
            self._publish_message("/robot_status", {
                "agent_id": agent_id,
                "status": "registered",
                "robot_type": robot_type,
                "capabilities": capabilities,
                "mesh_agent_id": mesh_response.get("agent_id")
            })
            
            return {
                "status": "registered",
                "agent_id": agent_id,
                "robot_type": robot_type,
                "namespace": namespace,
                "mesh_agent_id": mesh_response.get("agent_id"),
                "available_topics": list(self.topics.keys()),
                "available_services": list(self.services.keys())
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def unregister_ros_robot(self, agent_id: str) -> Dict[str, Any]:
        """Unregister a ROS robot."""
        try:
            with self.robot_lock:
                if agent_id not in self.ros_robots:
                    return {
                        "status": "error",
                        "error": f"Robot {agent_id} not found"
                    }
                
                # Remove robot
                del self.ros_robots[agent_id]
            
            # Terminate in mesh
            mesh_response = self.mesh_api.terminate_agent({"agent_id": agent_id})
            
            # Publish unregistration
            self._publish_message("/robot_status", {
                "agent_id": agent_id,
                "status": "unregistered"
            })
            
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
    
    def process_ros_cognitive_input(self, agent_id: str, 
                                  cognitive_input: ROSCognitiveInput) -> Dict[str, Any]:
        """Process cognitive input from ROS robot."""
        try:
            # Validate robot
            with self.robot_lock:
                if agent_id not in self.ros_robots:
                    return {
                        "status": "error",
                        "error": f"ROS robot {agent_id} not registered"
                    }
                
                # Update robot activity
                self.ros_robots[agent_id].last_update = time.time()
            
            # Convert ROS cognitive input to mesh format
            mesh_input = self._convert_ros_input_to_mesh(cognitive_input)
            
            # Process through mesh API
            mesh_request = {
                "input": mesh_input,
                "agent_id": agent_id,
                "mode": "neural_symbolic"
            }
            
            mesh_response = self.mesh_api.process_cognitive_input(mesh_request)
            
            # Convert response to ROS motor output
            motor_output = self._convert_mesh_output_to_ros(mesh_response, agent_id)
            
            # Publish motor output
            self._publish_message("/motor_output", motor_output)
            
            return {
                "status": "processed",
                "agent_id": agent_id,
                "motor_output": motor_output,
                "processing_time": time.time() - cognitive_input.header.get("timestamp", time.time())
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_robot_joint_states(self, agent_id: str, 
                                joint_states: Dict[str, float]) -> Dict[str, Any]:
        """Update robot joint states."""
        try:
            with self.robot_lock:
                if agent_id not in self.ros_robots:
                    return {
                        "status": "error",
                        "error": f"Robot {agent_id} not found"
                    }
                
                # Update joint states
                robot = self.ros_robots[agent_id]
                robot.joint_states.update(joint_states)
                robot.last_update = time.time()
            
            # Publish joint state update
            self._publish_message("/robot_status", {
                "agent_id": agent_id,
                "joint_states": joint_states,
                "timestamp": time.time()
            })
            
            return {
                "status": "updated",
                "agent_id": agent_id,
                "joint_states": joint_states
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_robot_sensor_states(self, agent_id: str, 
                                 sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update robot sensor states."""
        try:
            with self.robot_lock:
                if agent_id not in self.ros_robots:
                    return {
                        "status": "error",
                        "error": f"Robot {agent_id} not found"
                    }
                
                # Update sensor states
                robot = self.ros_robots[agent_id]
                robot.sensor_states.update(sensor_data)
                robot.last_update = time.time()
            
            return {
                "status": "updated",
                "agent_id": agent_id,
                "sensor_count": len(sensor_data)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_ros_robot_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of ROS robot."""
        try:
            with self.robot_lock:
                if agent_id not in self.ros_robots:
                    return {
                        "status": "error",
                        "error": f"Robot {agent_id} not found"
                    }
                
                robot = self.ros_robots[agent_id]
                
                return {
                    "status": "active",
                    "agent_id": agent_id,
                    "robot_type": robot.robot_type,
                    "namespace": robot.namespace,
                    "capabilities": robot.capabilities,
                    "joint_states": robot.joint_states,
                    "sensor_states": robot.sensor_states,
                    "last_update": robot.last_update,
                    "mesh_status": self.mesh_api.get_active_agents()
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def emergency_stop_robot(self, agent_id: str) -> Dict[str, Any]:
        """Emergency stop for ROS robot."""
        try:
            with self.robot_lock:
                if agent_id not in self.ros_robots:
                    return {
                        "status": "error",
                        "error": f"Robot {agent_id} not found"
                    }
            
            # Create emergency stop motor output
            emergency_output = ROSMotorOutput(
                header=self._create_ros_header(f"/{agent_id}/base_link"),
                linear_velocity={"x": 0.0, "y": 0.0, "z": 0.0},
                angular_velocity={"x": 0.0, "y": 0.0, "z": 0.0},
                joint_commands=[],  # Will be filled with zero velocities
                gripper_commands={"position": 0.0},
                safety_constraints={"emergency_stop": True}
            )
            
            # Publish emergency stop
            self._publish_message("/motor_output", asdict(emergency_output))
            
            return {
                "status": "emergency_stopped",
                "agent_id": agent_id,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def call_ros_service(self, service_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call a ROS-style service."""
        try:
            if service_name not in self.services:
                return {
                    "status": "error",
                    "error": f"Service {service_name} not available"
                }
            
            # Call service handler
            service_handler = self.services[service_name]
            response = service_handler(request_data)
            
            return {
                "status": "success",
                "service": service_name,
                "response": response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_ros_topic_data(self, topic_name: str, max_messages: int = 10) -> Dict[str, Any]:
        """Get recent messages from a ROS topic."""
        try:
            if topic_name not in self.topics:
                return {
                    "status": "error",
                    "error": f"Topic {topic_name} not available"
                }
            
            # Get recent messages
            messages = self.topics[topic_name][-max_messages:]
            
            return {
                "status": "success",
                "topic": topic_name,
                "message_count": len(messages),
                "messages": messages
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _convert_ros_input_to_mesh(self, cognitive_input: ROSCognitiveInput) -> List[float]:
        """Convert ROS cognitive input to mesh-compatible format."""
        input_vector = []
        
        # Process sensor data
        for sensor_type, readings in cognitive_input.sensor_data.items():
            if isinstance(readings, list):
                input_vector.extend(readings)
            else:
                input_vector.append(float(readings))
        
        # Add perception data
        perception_data = cognitive_input.perception_data
        if isinstance(perception_data, dict):
            for key, value in perception_data.items():
                if isinstance(value, (int, float)):
                    input_vector.append(float(value))
                elif isinstance(value, list):
                    input_vector.extend([float(v) for v in value if isinstance(v, (int, float))])
        
        # Ensure minimum input size
        if len(input_vector) < 10:
            input_vector.extend([0.0] * (10 - len(input_vector)))
        
        return input_vector
    
    def _convert_mesh_output_to_ros(self, mesh_response: Dict[str, Any], 
                                  agent_id: str) -> Dict[str, Any]:
        """Convert mesh response to ROS motor output format."""
        output_data = mesh_response.get("output_data", [])
        
        # Create ROS header
        header = self._create_ros_header(f"/{agent_id}/base_link")
        
        # Extract motor commands
        if isinstance(output_data, list) and len(output_data) > 0:
            if isinstance(output_data[0], list):
                data = output_data[0]
            else:
                data = output_data
            
            # Convert to motor commands
            linear_vel = {
                "x": float(data[0]) if len(data) > 0 else 0.0,
                "y": float(data[1]) if len(data) > 1 else 0.0,
                "z": float(data[2]) if len(data) > 2 else 0.0
            }
            
            angular_vel = {
                "x": float(data[3]) if len(data) > 3 else 0.0,
                "y": float(data[4]) if len(data) > 4 else 0.0,
                "z": float(data[5]) if len(data) > 5 else 0.0
            }
            
            # Joint commands (if available)
            joint_commands = [float(d) for d in data[6:]] if len(data) > 6 else []
            
        else:
            linear_vel = {"x": 0.0, "y": 0.0, "z": 0.0}
            angular_vel = {"x": 0.0, "y": 0.0, "z": 0.0}
            joint_commands = []
        
        motor_output = {
            "header": header,
            "linear_velocity": linear_vel,
            "angular_velocity": angular_vel,
            "joint_commands": joint_commands,
            "gripper_commands": {"position": 0.0},
            "safety_constraints": {"max_velocity": 1.0}
        }
        
        return motor_output
    
    def _create_ros_header(self, frame_id: str = "base_link") -> Dict[str, Any]:
        """Create a ROS-style header."""
        self.sequence_counter += 1
        return {
            "seq": self.sequence_counter,
            "stamp": {
                "secs": int(time.time()),
                "nsecs": int((time.time() % 1) * 1e9)
            },
            "frame_id": frame_id
        }
    
    def _publish_message(self, topic: str, data: Dict[str, Any]):
        """Publish message to a ROS topic (simulated)."""
        if topic in self.topics:
            message = {
                "header": self._create_ros_header(),
                "data": data,
                "timestamp": time.time()
            }
            self.topics[topic].append(message)
            
            # Keep only recent messages
            if len(self.topics[topic]) > 100:
                self.topics[topic] = self.topics[topic][-50:]
    
    # Message handlers
    def _handle_cognitive_input(self, message: ROSMessage) -> Dict[str, Any]:
        """Handle cognitive input message."""
        cognitive_input = ROSCognitiveInput(**message.data)
        return self.process_ros_cognitive_input(message.agent_id, cognitive_input)
    
    def _handle_status_query(self, message: ROSMessage) -> Dict[str, Any]:
        """Handle status query."""
        return self.get_ros_robot_status(message.agent_id)
    
    def _handle_attention_request(self, message: ROSMessage) -> Dict[str, Any]:
        """Handle attention allocation request."""
        return self.mesh_api.allocate_attention(message.data)
    
    def _handle_joint_state(self, message: ROSMessage) -> Dict[str, Any]:
        """Handle joint state update."""
        return self.update_robot_joint_states(message.agent_id, message.data.get("joint_states", {}))
    
    def _handle_sensor_data(self, message: ROSMessage) -> Dict[str, Any]:
        """Handle sensor data update."""
        return self.update_robot_sensor_states(message.agent_id, message.data.get("sensor_data", {}))
    
    def _handle_goal_update(self, message: ROSMessage) -> Dict[str, Any]:
        """Handle goal update."""
        # For now, just acknowledge
        return {"status": "goal_updated", "agent_id": message.agent_id}
    
    # Service handlers
    def _service_register_robot(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Service handler for robot registration."""
        return self.register_ros_robot(request)
    
    def _service_process_cognitive(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Service handler for cognitive processing."""
        agent_id = request.get("agent_id", "")
        cognitive_input = ROSCognitiveInput(**request.get("cognitive_input", {}))
        return self.process_ros_cognitive_input(agent_id, cognitive_input)
    
    def _service_get_robot_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Service handler for robot status."""
        agent_id = request.get("agent_id", "")
        return self.get_ros_robot_status(agent_id)
    
    def _service_allocate_attention(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Service handler for attention allocation."""
        return self.mesh_api.allocate_attention(request)
    
    def _service_emergency_stop(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Service handler for emergency stop."""
        agent_id = request.get("agent_id", "")
        return self.emergency_stop_robot(agent_id)