"""
Embodiment interfaces for the distributed cognitive mesh.

Provides interfaces for Unity3D, ROS, and web-based cognitive agents.
"""

from .unity3d_interface import Unity3DInterface
from .ros_interface import ROSInterface  
from .web_interface import WebInterface

__all__ = [
    'Unity3DInterface',
    'ROSInterface',
    'WebInterface'
]