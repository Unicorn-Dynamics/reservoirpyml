"""
Cognitive Visualization Tools
===========================

Visualization tools for hypergraph dynamics and cognitive state representation.
Provides clear visual representation of ReservoirPy → Hypergraph translations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from io import BytesIO
import base64

from .hypergraph import AtomSpace, HypergraphNode, HypergraphLink
from .tensor_fragment import TensorFragment, TensorSignature


class CognitiveVisualizer:
    """
    Visualization tools for hypergraph dynamics and cognitive states.
    
    Provides multiple visualization modes including network graphs,
    tensor fragment heatmaps, and cognitive flow diagrams.
    
    Parameters
    ----------
    style : str, default='cognitive'
        Visualization style ('cognitive', 'technical', 'minimal')
    figsize : tuple, default=(12, 8)
        Default figure size for plots
    """
    
    def __init__(self, style: str = 'cognitive', figsize: Tuple[int, int] = (12, 8)):
        self.style = style
        self.figsize = figsize
        
        # Color schemes for different styles
        self._color_schemes = {
            'cognitive': {
                'reservoir': '#FF6B6B',      # Coral red
                'readout': '#4ECDC4',        # Teal
                'activation': '#45B7D1',     # Blue
                'input': '#96CEB4',          # Light green  
                'output': '#FFEAA7',         # Light yellow
                'operator': '#DDA0DD',       # Plum
                'tensor_fragment': '#FFB347', # Peach
                'connection': '#34495e',      # Dark blue-gray
                'feedback': '#e74c3c',       # Red
                'state_flow': '#9b59b6'      # Purple
            },
            'technical': {
                'reservoir': '#2c3e50',
                'readout': '#34495e', 
                'activation': '#7f8c8d',
                'input': '#95a5a6',
                'output': '#bdc3c7',
                'operator': '#ecf0f1',
                'tensor_fragment': '#3498db',
                'connection': '#2c3e50',
                'feedback': '#e74c3c',
                'state_flow': '#9b59b6'
            },
            'minimal': {
                'reservoir': '#333333',
                'readout': '#666666',
                'activation': '#999999',
                'input': '#cccccc',
                'output': '#eeeeee',
                'operator': '#bbbbbb',
                'tensor_fragment': '#777777',
                'connection': '#333333',
                'feedback': '#666666',
                'state_flow': '#999999'
            }
        }
    
    def visualize_atomspace(
        self, 
        atomspace: AtomSpace,
        layout: str = 'spring',
        show_properties: bool = False,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize an AtomSpace as a network graph.
        
        Parameters
        ----------
        atomspace : AtomSpace
            AtomSpace to visualize
        layout : str, default='spring'
            Network layout algorithm ('spring', 'circular', 'hierarchical')
        show_properties : bool, default=False
            Whether to show node properties as labels
        save_path : str, optional
            Path to save the visualization
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        node_colors = []
        node_labels = {}
        
        for node in atomspace.nodes:
            G.add_node(node.name)
            
            # Get color based on node type
            color = self._color_schemes[self.style].get(node.node_type, '#cccccc')
            node_colors.append(color)
            
            # Create label
            if show_properties:
                props_str = self._format_properties_brief(node.properties)
                node_labels[node.name] = f"{node.name}\n{props_str}"
            else:
                node_labels[node.name] = node.name
        
        # Add edges
        edge_colors = []
        edge_styles = []
        
        for link in atomspace.links:
            if len(link.nodes) >= 2:
                # For hypergraph links with more than 2 nodes, create pairwise connections
                for i in range(len(link.nodes)):
                    for j in range(i + 1, len(link.nodes)):
                        G.add_edge(link.nodes[i].name, link.nodes[j].name)
                        
                        # Edge styling based on link type
                        color = self._color_schemes[self.style].get(link.link_type, '#cccccc')
                        edge_colors.append(color)
                        
                        if link.link_type == 'feedback':
                            edge_styles.append('dashed')
                        else:
                            edge_styles.append('solid')
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'hierarchical':
            pos = self._hierarchical_layout(G, atomspace)
        else:
            pos = nx.spring_layout(G)
        
        # Draw the graph
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, 
            node_size=1000, alpha=0.8, ax=ax
        )
        
        # Draw edges with different styles
        for i, (edge, color, style) in enumerate(zip(G.edges(), edge_colors, edge_styles)):
            nx.draw_networkx_edges(
                G, pos, [edge], edge_color=[color],
                style=style, width=2, alpha=0.7, ax=ax
            )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=ax)
        
        # Add title and legend
        ax.set_title(f"AtomSpace: {atomspace.name}\n"
                    f"Nodes: {len(atomspace.nodes)}, Links: {len(atomspace.links)}",
                    fontsize=14, pad=20)
        
        self._add_legend(ax, atomspace)
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_tensor_fragment(
        self, 
        fragment: TensorFragment,
        slice_dim: int = 0,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize a tensor fragment as heatmaps and statistics.
        
        Parameters
        ----------
        fragment : TensorFragment
            Tensor fragment to visualize
        slice_dim : int, default=0
            Dimension along which to slice for visualization
        save_path : str, optional
            Path to save the visualization
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Tensor signature info
        ax_info = fig.add_subplot(gs[0, :2])
        self._plot_signature_info(ax_info, fragment.signature)
        
        # Tensor shape visualization
        ax_shape = fig.add_subplot(gs[0, 2:])
        self._plot_tensor_shape(ax_shape, fragment)
        
        # Heatmap of tensor slices
        if fragment.data is not None and fragment.data.size > 0:
            # Reshape data for visualization
            data = fragment.data
            if len(data.shape) > 2:
                # Take slices along the specified dimension
                for i in range(min(4, data.shape[slice_dim])):
                    ax_heat = fig.add_subplot(gs[1 + i // 2, (i % 2) * 2:(i % 2 + 1) * 2])
                    
                    if len(data.shape) == 5:
                        # 5D tensor: show 2D slice
                        slice_data = np.take(data, i, axis=slice_dim)
                        slice_2d = slice_data.reshape(slice_data.shape[0], -1)
                    else:
                        slice_2d = data
                    
                    im = ax_heat.imshow(slice_2d, cmap='viridis', aspect='auto')
                    ax_heat.set_title(f'Slice {i} (dim {slice_dim})')
                    plt.colorbar(im, ax=ax_heat, shrink=0.8)
        
        # Statistics
        ax_stats = fig.add_subplot(gs[2, :])
        self._plot_tensor_statistics(ax_stats, fragment)
        
        fig.suptitle(f'Tensor Fragment Visualization\n'
                    f'Shape: {fragment.signature.get_tensor_shape()}', 
                    fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_cognitive_flow(
        self, 
        atomspace: AtomSpace,
        highlight_path: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize cognitive flow through the hypergraph.
        
        Parameters
        ----------
        atomspace : AtomSpace
            AtomSpace containing the cognitive architecture
        highlight_path : List[str], optional
            List of node names to highlight as a processing path
        save_path : str, optional
            Path to save the visualization
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create layered layout based on node types
        layers = self._create_cognitive_layers(atomspace)
        
        # Position nodes in layers
        positions = {}
        layer_heights = np.linspace(0.1, 0.9, len(layers))
        
        for layer_idx, (layer_name, nodes) in enumerate(layers.items()):
            y = layer_heights[layer_idx]
            x_positions = np.linspace(0.1, 0.9, len(nodes))
            
            for node_idx, node in enumerate(nodes):
                positions[node.name] = (x_positions[node_idx], y)
        
        # Draw nodes
        for node in atomspace.nodes:
            if node.name in positions:
                x, y = positions[node.name]
                
                # Node styling
                color = self._color_schemes[self.style].get(node.node_type, '#cccccc')
                
                # Highlight if in path
                if highlight_path and node.name in highlight_path:
                    edge_color = '#FFD700'  # Gold
                    linewidth = 3
                else:
                    edge_color = 'black'
                    linewidth = 1
                
                # Draw node
                circle = plt.Circle((x, y), 0.03, color=color, 
                                  ec=edge_color, linewidth=linewidth, alpha=0.8)
                ax.add_patch(circle)
                
                # Add label
                ax.text(x, y - 0.05, node.name, ha='center', va='top', 
                       fontsize=8, weight='bold' if highlight_path and node.name in highlight_path else 'normal')
        
        # Draw connections
        for link in atomspace.links:
            if len(link.nodes) >= 2:
                for i in range(len(link.nodes) - 1):
                    node1_name = link.nodes[i].name
                    node2_name = link.nodes[i + 1].name
                    
                    if node1_name in positions and node2_name in positions:
                        x1, y1 = positions[node1_name]
                        x2, y2 = positions[node2_name]
                        
                        # Connection styling
                        color = self._color_schemes[self.style].get(link.link_type, '#cccccc')
                        
                        if link.link_type == 'feedback':
                            linestyle = '--'
                            alpha = 0.6
                        else:
                            linestyle = '-'
                            alpha = 0.8
                        
                        # Highlight if part of path
                        if (highlight_path and 
                            node1_name in highlight_path and 
                            node2_name in highlight_path):
                            color = '#FFD700'
                            linewidth = 3
                        else:
                            linewidth = 2
                        
                        ax.plot([x1, x2], [y1, y2], color=color, 
                               linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                        
                        # Add arrow for direction
                        if link.link_type == 'connection':
                            self._add_arrow(ax, x1, y1, x2, y2, color)
        
        # Add layer labels
        for layer_idx, layer_name in enumerate(layers.keys()):
            y = layer_heights[layer_idx]
            ax.text(0.02, y, layer_name.replace('_', ' ').title(), 
                   rotation=90, va='center', ha='right', fontsize=10, weight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.set_title(f'Cognitive Flow Diagram: {atomspace.name}', fontsize=16, pad=20)
        
        # Add legend
        self._add_flow_legend(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparison_plot(
        self, 
        original_data: np.ndarray,
        reconstructed_data: np.ndarray,
        title: str = "Round-trip Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comparison plot for round-trip validation.
        
        Parameters
        ----------
        original_data : np.ndarray
            Original data
        reconstructed_data : np.ndarray
            Reconstructed data after round-trip
        title : str, default="Round-trip Comparison"
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Flatten data for comparison
        orig_flat = original_data.flatten()
        recon_flat = reconstructed_data.flatten()
        
        # Align lengths
        min_len = min(len(orig_flat), len(recon_flat))
        orig_flat = orig_flat[:min_len]
        recon_flat = recon_flat[:min_len]
        
        # Time series comparison
        axes[0, 0].plot(orig_flat[:100], label='Original', alpha=0.7)
        axes[0, 0].plot(recon_flat[:100], label='Reconstructed', alpha=0.7)
        axes[0, 0].set_title('Time Series (first 100 points)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(orig_flat, recon_flat, alpha=0.5, s=1)
        min_val = min(np.min(orig_flat), np.min(recon_flat))
        max_val = max(np.max(orig_flat), np.max(recon_flat))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Original')
        axes[0, 1].set_ylabel('Reconstructed')
        axes[0, 1].set_title('Correlation Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error histogram
        error = orig_flat - recon_flat
        axes[1, 0].hist(error, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics
        mse = np.mean(error ** 2)
        correlation = np.corrcoef(orig_flat, recon_flat)[0, 1]
        max_error = np.max(np.abs(error))
        
        stats_text = f"""Statistics:
MSE: {mse:.6f}
Correlation: {correlation:.4f}
Max Error: {max_error:.6f}
Mean Error: {np.mean(error):.6f}
Std Error: {np.std(error):.6f}"""
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Fidelity Metrics')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # Helper methods
    
    def _format_properties_brief(self, properties: Dict[str, Any]) -> str:
        """Format properties for brief display."""
        brief_props = []
        for key, value in properties.items():
            if key in ['class_name', 'input_dim', 'output_dim']:
                brief_props.append(f"{key}: {value}")
        return '\n'.join(brief_props[:3])  # Max 3 properties
    
    def _hierarchical_layout(self, G: nx.Graph, atomspace: AtomSpace) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on node types."""
        layers = self._create_cognitive_layers(atomspace)
        positions = {}
        
        layer_y = np.linspace(0, 1, len(layers))
        
        for layer_idx, (layer_name, nodes) in enumerate(layers.items()):
            y = layer_y[layer_idx]
            x_positions = np.linspace(0, 1, len(nodes))
            
            for node_idx, node in enumerate(nodes):
                positions[node.name] = (x_positions[node_idx], y)
        
        return positions
    
    def _create_cognitive_layers(self, atomspace: AtomSpace) -> Dict[str, List[HypergraphNode]]:
        """Create cognitive processing layers from atomspace."""
        layers = {
            'input_layer': [],
            'processing_layer': [],
            'activation_layer': [],
            'output_layer': [],
            'meta_layer': []
        }
        
        for node in atomspace.nodes:
            if node.node_type == 'input':
                layers['input_layer'].append(node)
            elif node.node_type in ['reservoir', 'readout']:
                layers['processing_layer'].append(node)
            elif node.node_type == 'activation':
                layers['activation_layer'].append(node)
            elif node.node_type == 'output':
                layers['output_layer'].append(node)
            else:
                layers['meta_layer'].append(node)
        
        # Remove empty layers
        return {k: v for k, v in layers.items() if v}
    
    def _add_legend(self, ax: plt.Axes, atomspace: AtomSpace) -> None:
        """Add legend for node and link types."""
        # Get unique types
        node_types = set(node.node_type for node in atomspace.nodes)
        link_types = set(link.link_type for link in atomspace.links)
        
        legend_elements = []
        
        # Node type legend
        for node_type in sorted(node_types):
            color = self._color_schemes[self.style].get(node_type, '#cccccc')
            legend_elements.append(
                patches.Patch(color=color, label=f"{node_type} node")
            )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def _add_flow_legend(self, ax: plt.Axes) -> None:
        """Add legend for cognitive flow diagram."""
        legend_elements = [
            patches.Patch(color=self._color_schemes[self.style]['connection'], 
                         label='Forward connection'),
            patches.Patch(color=self._color_schemes[self.style]['feedback'], 
                         label='Feedback connection'),
            patches.Patch(color='#FFD700', label='Highlighted path')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def _add_arrow(self, ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str) -> None:
        """Add directional arrow to connection."""
        # Calculate arrow position (2/3 along the line)
        arrow_x = x1 + 0.67 * (x2 - x1)
        arrow_y = y1 + 0.67 * (y2 - y1)
        
        # Calculate arrow direction
        dx = x2 - x1
        dy = y2 - y1
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length
        
        # Add arrow
        ax.annotate('', xy=(arrow_x + 0.01*dx, arrow_y + 0.01*dy), 
                   xytext=(arrow_x - 0.01*dx, arrow_y - 0.01*dy),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    def _plot_signature_info(self, ax: plt.Axes, signature: TensorSignature) -> None:
        """Plot tensor signature information."""
        info_text = f"""Tensor Signature:
Modality: {signature.modality}
Depth: {signature.depth}
Context: {signature.context}
Salience: {signature.salience}
Autonomy: {signature.autonomy_index}

Total Dimensions: {signature.get_total_dimensions()}"""
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.axis('off')
        ax.set_title('Signature Info')
    
    def _plot_tensor_shape(self, ax: plt.Axes, fragment: TensorFragment) -> None:
        """Plot tensor shape visualization."""
        shape = fragment.signature.get_tensor_shape()
        
        # Create 3D-like representation
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        
        # Draw tensor dimensions as nested rectangles
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        
        for i, (dim, color) in enumerate(zip(shape, colors)):
            width = 8 - i * 1.2
            height = 6 - i * 1
            x = 1 + i * 0.6
            y = 1 + i * 0.5
            
            rect = FancyBboxPatch((x, y), width, height, 
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, alpha=0.3,
                                 edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # Add dimension label
            ax.text(x + width/2, y + height/2, f"{dim}", 
                   ha='center', va='center', fontsize=12, weight='bold')
        
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Tensor Shape')
    
    def _plot_tensor_statistics(self, ax: plt.Axes, fragment: TensorFragment) -> None:
        """Plot tensor statistics."""
        if fragment.data is None:
            ax.text(0.5, 0.5, 'No tensor data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return
        
        data = fragment.data.flatten()
        
        # Create histogram
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Data Distribution (mean: {np.mean(data):.3f}, std: {np.std(data):.3f})')
        ax.grid(True, alpha=0.3)