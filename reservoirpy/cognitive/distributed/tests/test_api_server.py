"""
Basic tests for the distributed cognitive mesh API server.
"""

import time
import threading
import numpy as np
from typing import Dict, Any

from ..api_server import CognitiveMeshAPI


def test_api_server_basic():
    """Test basic API server functionality."""
    print("Testing Cognitive Mesh API Server...")
    
    # Create API server
    api = CognitiveMeshAPI(host="localhost", port=8081)
    
    # Test server startup
    start_result = api.start_server()
    print(f"Server start: {start_result['status']}")
    assert start_result["status"] == "started"
    
    # Test mesh status
    status = api.get_mesh_status()
    print(f"Mesh status: {status['status']}")
    assert status["status"] == "active"
    
    # Test cognitive processing
    test_input = {
        "input": [1.0, 2.0, 3.0, 4.0, 5.0],
        "agent_id": "test_agent",
        "mode": "neural_symbolic"
    }
    
    process_result = api.process_cognitive_input(test_input)
    print(f"Processing result: {process_result['status']}")
    assert process_result["status"] == "processed"
    assert "output_data" in process_result
    
    # Test agent spawning
    spawn_result = api.spawn_agent({
        "type": "test_agent",
        "embodiment": "virtual",
        "config": {"test": True}
    })
    print(f"Agent spawn: {spawn_result['status']}")
    assert spawn_result["status"] == "spawned"
    
    agent_id = spawn_result["agent_id"]
    
    # Test agent status
    agents = api.get_active_agents()
    print(f"Active agents: {agents['total_agents']}")
    assert agents["total_agents"] > 0
    assert agent_id in agents["agents"]
    
    # Test task distribution
    task_result = api.distribute_task({
        "task": {"type": "test", "data": [1, 2, 3]},
        "priority": 1.0,
        "target_agents": [agent_id]
    })
    print(f"Task distribution: {task_result['status']}")
    assert task_result["status"] == "distributed"
    
    # Test attention allocation
    attention_result = api.allocate_attention({
        "target": "test_concept",
        "sti": 2.0
    })
    print(f"Attention allocation: {attention_result['status']}")
    assert attention_result["status"] == "allocated"
    
    # Test agent termination
    terminate_result = api.terminate_agent({"agent_id": agent_id})
    print(f"Agent termination: {terminate_result['status']}")
    assert terminate_result["status"] == "terminated"
    
    # Test server shutdown
    stop_result = api.stop_server()
    print(f"Server stop: {stop_result['status']}")
    assert stop_result["status"] == "stopped"
    
    print("✅ API Server tests passed!")
    return True


def test_api_server_concurrent():
    """Test concurrent API server operations."""
    print("Testing concurrent API operations...")
    
    api = CognitiveMeshAPI(host="localhost", port=8082)
    api.start_server()
    
    # Test concurrent processing
    def process_input(thread_id):
        for i in range(5):
            result = api.process_cognitive_input({
                "input": [float(thread_id), float(i), 1.0, 2.0, 3.0],
                "agent_id": f"thread_{thread_id}",
                "mode": "neural_symbolic"
            })
            assert result["status"] == "processed"
    
    # Run concurrent threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=process_input, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Check final state
    status = api.get_mesh_status()
    assert status["status"] == "active"
    
    api.stop_server()
    print("✅ Concurrent API tests passed!")
    return True


def test_api_server_error_handling():
    """Test API server error handling."""
    print("Testing API error handling...")
    
    api = CognitiveMeshAPI(host="localhost", port=8083)
    api.start_server()
    
    # Test invalid input
    invalid_result = api.process_cognitive_input({
        "input": "invalid_input",  # Should be list
        "agent_id": "test_agent"
    })
    print(f"Invalid input handling: {invalid_result['status']}")
    assert invalid_result["status"] == "error"
    
    # Test missing agent
    missing_agent_result = api.terminate_agent({"agent_id": "nonexistent"})
    print(f"Missing agent handling: {missing_agent_result['status']}")
    assert missing_agent_result["status"] == "error"
    
    # Test invalid attention target
    attention_result = api.allocate_attention({
        "target": "",  # Empty target
        "sti": "invalid"  # Should be numeric
    })
    print(f"Invalid attention handling: {attention_result['status']}")
    assert attention_result["status"] == "error"
    
    api.stop_server()
    print("✅ Error handling tests passed!")
    return True


if __name__ == "__main__":
    test_api_server_basic()
    test_api_server_concurrent()
    test_api_server_error_handling()
    print("🎉 All API server tests completed successfully!")