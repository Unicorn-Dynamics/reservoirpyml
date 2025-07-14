"""
GGML-backed ReservoirPy Nodes
============================

ReservoirPy nodes with GGML backend for neural-symbolic computation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .core import GGMLTensor, GGMLContext, TensorType
from .bridge import NeuralSymbolicBridge
from .kernels import hypergraph_conv, symbolic_activation, attention_weighted_op

from ..hypergraph import AtomSpace
from ..attention.ecan import ECANAttentionSystem
from ..attention.attention_reservoir import AttentionReservoir

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import Node


class GGMLReservoir(Node):
    """
    GGML-backed reservoir computing node with neural-symbolic capabilities.
    
    Extends ReservoirPy's Node architecture with GGML tensor operations
    and symbolic reasoning capabilities.
    
    Parameters
    ----------
    units : int
        Number of reservoir units
    ggml_context : GGMLContext
        GGML computation context
    atom_space : AtomSpace, optional
        AtomSpace for symbolic representation
    symbolic_enhancement : bool
        Whether to apply symbolic enhancement
    lr : float
        Leak rate for reservoir dynamics
    sr : float
        Spectral radius for reservoir initialization
    input_scaling : float
        Input scaling factor
    **kwargs
        Additional parameters for reservoir initialization
    """
    
    def __init__(
        self,
        units: int = 100,
        ggml_context: Optional[GGMLContext] = None,
        atom_space: Optional[AtomSpace] = None,
        symbolic_enhancement: bool = True,
        lr: float = 0.3,
        sr: float = 1.25,
        input_scaling: float = 1.0,
        **kwargs
    ):
        # Initialize parent Node class
        super().__init__(
            hypers={"units": units, "lr": lr, "sr": sr, "input_scaling": input_scaling},
            **kwargs
        )
        
        self.units = units
        self.lr = lr
        self.sr = sr
        self.input_scaling = input_scaling
        self.symbolic_enhancement = symbolic_enhancement
        
        # GGML and symbolic components
        self.ggml_context = ggml_context or GGMLContext()
        self.atom_space = atom_space or AtomSpace()
        self.bridge = NeuralSymbolicBridge(self.ggml_context, self.atom_space)
        
        # Internal reservoir for base computation
        self._reservoir = None
        self._reservoir_initialized = False
        
        # Symbolic state tracking
        self.symbolic_state = None
        self.computation_history = []
        
    def _initialize_reservoir(self, input_dim: int):
        """Initialize the internal reservoir."""
        if not self._reservoir_initialized:
            self._reservoir = Reservoir(
                units=self.units,
                lr=self.lr,
                sr=self.sr,
                input_scaling=self.input_scaling,
                input_dim=input_dim
            )
            self._reservoir_initialized = True
            
    def initialize(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """Initialize the GGML reservoir node."""
        # Handle input dimension detection
        if x is not None:
            input_dim = x.shape[-1] if x.ndim > 1 else 1
        else:
            input_dim = getattr(self, 'input_dim', None)
            
        if input_dim is None:
            raise RuntimeError(
                f"Impossible to initialize node {self.name}: input_dim is unknown and no input data x was given."
            )
            
        self._initialize_reservoir(input_dim)
        if self._reservoir:
            self._reservoir.initialize(x, y)
            
        # Initialize GGML state
        if x is not None:
            initial_state = np.zeros((1, self.units))
            self.symbolic_state = self.ggml_context.create_tensor(
                data=initial_state,
                tensor_type=TensorType.NEURAL,
                name=f"ggml_reservoir_{self.name}_state"
            )
            
    def call(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward pass through GGML reservoir.
        
        Parameters
        ----------
        x : np.ndarray
            Input data
            
        Returns
        -------
        np.ndarray
            Reservoir output with optional symbolic enhancement
        """
        # Ensure reservoir is initialized
        if not self._reservoir_initialized:
            self._initialize_reservoir(x.shape[-1] if x.ndim > 1 else 1)
            self._reservoir.initialize(x)
            
        # Base reservoir computation
        reservoir_output = self._reservoir(x)
        
        if not self.symbolic_enhancement:
            return reservoir_output
            
        # Convert to GGML tensor for symbolic processing
        input_tensor = self.ggml_context.create_tensor(
            data=reservoir_output,
            tensor_type=TensorType.NEURAL,
            name=f"reservoir_output_{self.name}"
        )
        
        # Apply symbolic enhancement
        symbolic_tensor = input_tensor.to_symbolic(self.atom_space)
        
        # Apply symbolic activation for cognitive processing
        enhanced_tensor = symbolic_activation(
            symbolic_tensor,
            activation_type="cognitive_tanh"
        )
        
        # Apply hypergraph convolution for structural reasoning
        if enhanced_tensor.shape[0] > 1:  # Only if multiple units
            # Create simple edge structure (sequential connections)
            num_nodes = enhanced_tensor.shape[0]
            edge_indices = self.ggml_context.create_tensor(
                data=np.array([[i, (i+1) % num_nodes] for i in range(num_nodes)]),
                tensor_type=TensorType.SYMBOLIC,
                name="sequential_edges"
            )
            
            final_tensor = hypergraph_conv(
                enhanced_tensor, edge_indices,
                hidden_dim=min(64, enhanced_tensor.shape[-1]),
                num_heads=4
            )
        else:
            final_tensor = enhanced_tensor
            
        # Update symbolic state
        self.symbolic_state = final_tensor
        
        # Record computation step
        self.computation_history.append({
            "step": len(self.computation_history),
            "input_shape": x.shape,
            "output_shape": final_tensor.shape,
            "symbolic_context": final_tensor.symbolic_annotation.semantic_context,
            "enhancement_applied": True
        })
        
        return final_tensor.data
        
    def get_symbolic_state(self) -> Optional[GGMLTensor]:
        """Get current symbolic state of the reservoir."""
        return self.symbolic_state
        
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        return {
            "total_computations": len(self.computation_history),
            "symbolic_enhancement": self.symbolic_enhancement,
            "bridge_stats": self.bridge.get_bridge_statistics(),
            "ggml_context_stats": self.ggml_context.get_computation_stats(),
            "atom_space_nodes": len(self.atom_space.nodes),
            "computation_history": self.computation_history[-5:]  # Last 5 steps
        }


class SymbolicReadout(Node):
    """
    Symbolic reasoning readout layer with GGML backend.
    
    Implements readout operations with symbolic interpretation capabilities,
    enabling explicit reasoning about reservoir dynamics.
    
    Parameters
    ----------
    output_dim : int
        Output dimension
    ggml_context : GGMLContext, optional
        GGML computation context
    symbolic_interpretation : bool
        Whether to provide symbolic interpretation
    ridge : float
        Ridge regression regularization parameter
    **kwargs
        Additional parameters
    """
    
    def __init__(
        self,
        output_dim: int = 1,
        ggml_context: Optional[GGMLContext] = None,
        symbolic_interpretation: bool = True,
        ridge: float = 1e-6,
        **kwargs
    ):
        super().__init__(
            hypers={"output_dim": output_dim, "ridge": ridge},
            **kwargs
        )
        
        self._output_dim = output_dim
        self._ridge = ridge
        self.symbolic_interpretation = symbolic_interpretation
        
        # GGML components
        self.ggml_context = ggml_context or GGMLContext()
        
        # Internal readout for base computation
        self._readout = None
        self._readout_initialized = False
        
        # Symbolic interpretation tracking
        self.interpretation_history = []
        
    @property
    def output_dim(self):
        """Get output dimension."""
        return self._output_dim
        
    @property 
    def ridge(self):
        """Get ridge parameter."""
        return self._ridge
        
    def _initialize_readout(self, input_dim: int):
        """Initialize the internal readout."""
        if not self._readout_initialized:
            self._readout = Ridge(
                output_dim=self.output_dim,
                ridge=self.ridge
            )
            self._readout_initialized = True
            
    def initialize(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """Initialize the symbolic readout node."""
        if x is not None:
            input_dim = x.shape[-1] if x.ndim > 1 else 1
            self._initialize_readout(input_dim)
            if self._readout:
                self._readout.initialize(x, y)
                
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "SymbolicReadout":
        """
        Fit the symbolic readout with training data.
        
        Parameters
        ----------
        x : np.ndarray
            Input training data
        y : np.ndarray
            Target training data
            
        Returns
        -------
        SymbolicReadout
            Fitted readout node
        """
        # Ensure readout is initialized
        if not self._readout_initialized:
            self._initialize_readout(x.shape[-1] if x.ndim > 1 else 1)
            self._readout.initialize(x, y)
            
        # Fit base readout
        self._readout.fit(x, y, **kwargs)
        
        if self.symbolic_interpretation:
            # Analyze training data for symbolic patterns
            input_tensor = self.ggml_context.create_tensor(
                data=x,
                tensor_type=TensorType.NEURAL,
                name="training_input"
            )
            
            target_tensor = self.ggml_context.create_tensor(
                data=y,
                tensor_type=TensorType.NEURAL,
                name="training_target"
            )
            
            # Record symbolic interpretation of training
            self.interpretation_history.append({
                "phase": "training",
                "input_statistics": {
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "shape": x.shape
                },
                "target_statistics": {
                    "mean": float(np.mean(y)),
                    "std": float(np.std(y)),
                    "shape": y.shape
                },
                "symbolic_context": "readout_training_phase"
            })
            
        return self
        
    def call(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Forward pass through symbolic readout.
        
        Parameters
        ----------
        x : np.ndarray
            Input data
            
        Returns
        -------
        np.ndarray
            Readout output with symbolic interpretation
        """
        # Ensure readout is initialized
        if not self._readout_initialized:
            self._initialize_readout(x.shape[-1] if x.ndim > 1 else 1)
            if self._readout and hasattr(self._readout, 'Wout') and self._readout.Wout is None:
                raise RuntimeError(f"Readout {self.name} must be fitted before calling.")
                
        # Base readout computation
        output = self._readout(x)
        
        if not self.symbolic_interpretation:
            return output
            
        # Create symbolic interpretation
        input_tensor = self.ggml_context.create_tensor(
            data=x,
            tensor_type=TensorType.NEURAL,
            name=f"readout_input_{self.name}"
        )
        
        output_tensor = self.ggml_context.create_tensor(
            data=output,
            tensor_type=TensorType.NEURAL,
            name=f"readout_output_{self.name}"
        )
        
        # Apply symbolic activation to output for interpretation
        symbolic_output = symbolic_activation(
            output_tensor,
            activation_type="attention_softmax" if self.output_dim > 1 else "cognitive_tanh"
        )
        
        # Record symbolic interpretation
        interpretation = {
            "step": len(self.interpretation_history),
            "input_norm": float(np.linalg.norm(x)),
            "output_norm": float(np.linalg.norm(output)),
            "output_distribution": {
                "mean": float(np.mean(output)),
                "std": float(np.std(output)),
                "min": float(np.min(output)),
                "max": float(np.max(output))
            },
            "symbolic_context": symbolic_output.symbolic_annotation.semantic_context,
            "confidence_level": self._compute_confidence(output)
        }
        
        self.interpretation_history.append(interpretation)
        
        return output
        
    def _compute_confidence(self, output: np.ndarray) -> float:
        """Compute confidence level for symbolic interpretation."""
        if self.output_dim == 1:
            # For regression, confidence based on output magnitude
            return float(min(1.0, np.abs(np.mean(output))))
        else:
            # For classification, confidence based on max probability
            if output.ndim > 1:
                softmax_output = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
                return float(np.max(softmax_output))
            else:
                return float(np.max(np.abs(output)))
                
    def get_symbolic_interpretation(self) -> Dict[str, Any]:
        """Get symbolic interpretation of readout behavior."""
        if not self.interpretation_history:
            return {"symbolic_interpretation": False, "history": []}
            
        recent_interpretations = self.interpretation_history[-10:]  # Last 10 steps
        
        # Compute aggregate statistics
        confidence_scores = [interp.get("confidence_level", 0.0) for interp in recent_interpretations]
        output_norms = [interp.get("output_norm", 0.0) for interp in recent_interpretations]
        
        return {
            "symbolic_interpretation": True,
            "total_interpretations": len(self.interpretation_history),
            "recent_interpretations": recent_interpretations,
            "aggregate_stats": {
                "mean_confidence": float(np.mean(confidence_scores)) if confidence_scores else 0.0,
                "std_confidence": float(np.std(confidence_scores)) if confidence_scores else 0.0,
                "mean_output_norm": float(np.mean(output_norms)) if output_norms else 0.0,
                "interpretation_consistency": self._compute_interpretation_consistency()
            },
            "symbolic_patterns": self._detect_symbolic_patterns()
        }
        
    def _compute_interpretation_consistency(self) -> float:
        """Compute consistency of symbolic interpretations."""
        if len(self.interpretation_history) < 2:
            return 1.0
            
        recent = self.interpretation_history[-5:]
        confidence_values = [interp.get("confidence_level", 0.0) for interp in recent]
        
        # Consistency based on variance in confidence
        if len(confidence_values) > 1:
            consistency = 1.0 / (1.0 + np.var(confidence_values))
        else:
            consistency = 1.0
            
        return float(consistency)
        
    def _detect_symbolic_patterns(self) -> Dict[str, Any]:
        """Detect patterns in symbolic interpretations."""
        if len(self.interpretation_history) < 3:
            return {"patterns_detected": False}
            
        recent = self.interpretation_history[-10:]
        
        # Detect trends in confidence
        confidence_trend = [interp.get("confidence_level", 0.0) for interp in recent]
        if len(confidence_trend) > 1:
            trend_slope = np.polyfit(range(len(confidence_trend)), confidence_trend, 1)[0]
        else:
            trend_slope = 0.0
            
        # Detect output stability
        output_norms = [interp.get("output_norm", 0.0) for interp in recent]
        output_stability = 1.0 / (1.0 + np.std(output_norms)) if len(output_norms) > 1 else 1.0
        
        return {
            "patterns_detected": True,
            "confidence_trend": {
                "slope": float(trend_slope),
                "direction": "increasing" if trend_slope > 0.01 else "decreasing" if trend_slope < -0.01 else "stable"
            },
            "output_stability": float(output_stability),
            "stability_level": "high" if output_stability > 0.8 else "medium" if output_stability > 0.5 else "low"
        }