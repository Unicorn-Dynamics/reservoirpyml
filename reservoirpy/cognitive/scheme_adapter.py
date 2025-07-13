"""
Scheme Cognitive Grammar Adapter
==============================

Modular Scheme adapters for agentic grammar AtomSpace integration.
Provides S-expression based representation and manipulation of cognitive primitives.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import re
from dataclasses import dataclass

from .hypergraph import AtomSpace, HypergraphNode, HypergraphLink
from .tensor_fragment import TensorFragment, TensorSignature


@dataclass
class SchemeExpression:
    """Represents a Scheme S-expression for cognitive patterns."""
    operator: str
    operands: List[Union[str, int, float, 'SchemeExpression']]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_string(self, indent: int = 0) -> str:
        """Convert to Scheme string representation."""
        if not self.operands:
            return f"({self.operator})"
        
        indent_str = "  " * indent
        operand_strs = []
        
        for operand in self.operands:
            if isinstance(operand, SchemeExpression):
                operand_strs.append(operand.to_string(indent + 1))
            else:
                operand_strs.append(str(operand))
        
        if len(operand_strs) == 1 and not isinstance(self.operands[0], SchemeExpression):
            return f"({self.operator} {operand_strs[0]})"
        
        operands_str = "\n".join(f"{indent_str}  {op}" for op in operand_strs)
        return f"({self.operator}\n{operands_str})"
    
    def __repr__(self) -> str:
        return self.to_string()


class SchemeParser:
    """Parser for Scheme S-expressions."""
    
    def __init__(self):
        self.token_pattern = re.compile(r'\(|\)|[^\s()]+')
    
    def tokenize(self, scheme_str: str) -> List[str]:
        """Tokenize a Scheme string."""
        return self.token_pattern.findall(scheme_str)
    
    def parse(self, scheme_str: str) -> SchemeExpression:
        """Parse a Scheme string into a SchemeExpression."""
        tokens = self.tokenize(scheme_str)
        return self._parse_tokens(tokens)[0]
    
    def _parse_tokens(self, tokens: List[str]) -> List[Union[str, int, float, SchemeExpression]]:
        """Parse tokens into expressions."""
        results = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == '(':
                # Find matching closing paren
                paren_count = 1
                j = i + 1
                
                while j < len(tokens) and paren_count > 0:
                    if tokens[j] == '(':
                        paren_count += 1
                    elif tokens[j] == ')':
                        paren_count -= 1
                    j += 1
                
                # Parse the expression inside parens
                expr_tokens = tokens[i + 1:j - 1]
                if expr_tokens:
                    operator = expr_tokens[0]
                    operands = self._parse_tokens(expr_tokens[1:])
                    results.append(SchemeExpression(operator, operands))
                else:
                    results.append(SchemeExpression("empty", []))
                
                i = j
            elif token == ')':
                # Skip closing parens (handled above)
                i += 1
            else:
                # Parse atomic value
                results.append(self._parse_atom(token))
                i += 1
        
        return results
    
    def _parse_atom(self, token: str) -> Union[str, int, float]:
        """Parse an atomic token."""
        # Try integer
        try:
            return int(token)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(token)
        except ValueError:
            pass
        
        # Return as string
        return token


class SchemeAdapter:
    """
    Scheme-based cognitive grammar adapter for AtomSpace representation.
    
    Provides bidirectional conversion between hypergraph patterns and
    Scheme S-expressions for cognitive grammar manipulation.
    """
    
    def __init__(self):
        self.parser = SchemeParser()
        self._cognitive_primitives = {
            # Node type mappings
            "reservoir": "ReservoirNode",
            "readout": "ReadoutNode", 
            "activation": "ActivationNode",
            "input": "InputNode",
            "output": "OutputNode",
            "operator": "OperatorNode",
            "tensor_fragment": "TensorNode",
            
            # Link type mappings
            "connection": "ConnectionLink",
            "feedback": "FeedbackLink",
            "state_flow": "StateFlowLink"
        }
    
    def atomspace_to_scheme(self, atomspace: AtomSpace) -> SchemeExpression:
        """
        Convert an AtomSpace to Scheme S-expression representation.
        
        Parameters
        ----------
        atomspace : AtomSpace
            Source hypergraph
            
        Returns
        -------
        SchemeExpression
            Scheme representation of the cognitive pattern
        """
        # Create main AtomSpace expression
        nodes_expr = self._nodes_to_scheme(atomspace.nodes)
        links_expr = self._links_to_scheme(atomspace.links)
        
        return SchemeExpression(
            "AtomSpace",
            [
                SchemeExpression("name", [atomspace.name]),
                nodes_expr,
                links_expr
            ],
            metadata={"size": atomspace.size()}
        )
    
    def scheme_to_atomspace(self, scheme_expr: SchemeExpression) -> AtomSpace:
        """
        Convert a Scheme S-expression back to AtomSpace.
        
        Parameters
        ----------
        scheme_expr : SchemeExpression
            Source Scheme representation
            
        Returns
        -------
        AtomSpace
            Reconstructed hypergraph
        """
        if scheme_expr.operator != "AtomSpace":
            raise ValueError("Expected AtomSpace expression")
        
        # Extract name
        name = "default"
        nodes_expr = None
        links_expr = None
        
        for operand in scheme_expr.operands:
            if isinstance(operand, SchemeExpression):
                if operand.operator == "name" and operand.operands:
                    name = str(operand.operands[0])
                elif operand.operator == "nodes":
                    nodes_expr = operand
                elif operand.operator == "links":
                    links_expr = operand
        
        atomspace = AtomSpace(name)
        
        # Convert nodes
        if nodes_expr:
            nodes = self._scheme_to_nodes(nodes_expr)
            for node in nodes:
                atomspace.add_node(node)
        
        # Convert links
        if links_expr:
            links = self._scheme_to_links(links_expr, atomspace)
            for link in links:
                atomspace.add_link(link)
        
        return atomspace
    
    def _nodes_to_scheme(self, nodes: set) -> SchemeExpression:
        """Convert nodes to Scheme representation."""
        node_expressions = []
        
        for node in nodes:
            # Get cognitive primitive name
            cognitive_type = self._cognitive_primitives.get(node.node_type, "UnknownNode")
            
            # Create node expression
            node_expr = SchemeExpression(
                cognitive_type,
                [
                    SchemeExpression("name", [node.name]),
                    SchemeExpression("type", [node.node_type]),
                    self._properties_to_scheme(node.properties),
                    self._tensor_to_scheme(node.tensor_data)
                ]
            )
            node_expressions.append(node_expr)
        
        return SchemeExpression("nodes", node_expressions)
    
    def _links_to_scheme(self, links: set) -> SchemeExpression:
        """Convert links to Scheme representation."""
        link_expressions = []
        
        for link in links:
            # Get cognitive primitive name
            cognitive_type = self._cognitive_primitives.get(link.link_type, "UnknownLink")
            
            # Node references
            node_refs = [SchemeExpression("node-ref", [node.name]) for node in link.nodes]
            
            link_expr = SchemeExpression(
                cognitive_type,
                [
                    SchemeExpression("type", [link.link_type]),
                    SchemeExpression("strength", [link.strength]),
                    SchemeExpression("connects", node_refs),
                    self._properties_to_scheme(link.properties)
                ]
            )
            link_expressions.append(link_expr)
        
        return SchemeExpression("links", link_expressions)
    
    def _properties_to_scheme(self, properties: Dict[str, Any]) -> SchemeExpression:
        """Convert properties dictionary to Scheme representation."""
        prop_expressions = []
        
        for key, value in properties.items():
            if isinstance(value, dict):
                # Nested dictionary
                nested_expr = self._properties_to_scheme(value)
                prop_expressions.append(SchemeExpression(key, [nested_expr]))
            elif isinstance(value, list):
                # List of values
                prop_expressions.append(SchemeExpression(key, value))
            else:
                # Simple value
                prop_expressions.append(SchemeExpression(key, [value]))
        
        return SchemeExpression("properties", prop_expressions)
    
    def _tensor_to_scheme(self, tensor_data: Optional[Any]) -> SchemeExpression:
        """Convert tensor data to Scheme representation."""
        if tensor_data is None:
            return SchemeExpression("tensor", ["nil"])
        
        # Convert numpy array to basic statistics for Scheme representation
        import numpy as np
        if isinstance(tensor_data, np.ndarray):
            stats = {
                "shape": list(tensor_data.shape),
                "mean": float(np.mean(tensor_data)),
                "std": float(np.std(tensor_data)),
                "min": float(np.min(tensor_data)),
                "max": float(np.max(tensor_data))
            }
            return SchemeExpression("tensor", [self._properties_to_scheme(stats)])
        
        return SchemeExpression("tensor", [str(tensor_data)])
    
    def _scheme_to_nodes(self, nodes_expr: SchemeExpression) -> List[HypergraphNode]:
        """Convert Scheme nodes expression to HypergraphNode objects."""
        nodes = []
        
        for node_expr in nodes_expr.operands:
            if not isinstance(node_expr, SchemeExpression):
                continue
            
            # Extract node information
            name = "unknown"
            node_type = "unknown"
            properties = {}
            tensor_data = None
            
            for operand in node_expr.operands:
                if isinstance(operand, SchemeExpression):
                    if operand.operator == "name" and operand.operands:
                        name = str(operand.operands[0])
                    elif operand.operator == "type" and operand.operands:
                        node_type = str(operand.operands[0])
                    elif operand.operator == "properties":
                        properties = self._scheme_to_properties(operand)
                    elif operand.operator == "tensor":
                        # For now, skip tensor reconstruction in scheme conversion
                        pass
            
            nodes.append(HypergraphNode(name, node_type, properties, tensor_data))
        
        return nodes
    
    def _scheme_to_links(self, links_expr: SchemeExpression, atomspace: AtomSpace) -> List[HypergraphLink]:
        """Convert Scheme links expression to HypergraphLink objects."""
        links = []
        
        for link_expr in links_expr.operands:
            if not isinstance(link_expr, SchemeExpression):
                continue
            
            # Extract link information
            link_type = "unknown"
            strength = 1.0
            node_names = []
            properties = {}
            
            for operand in link_expr.operands:
                if isinstance(operand, SchemeExpression):
                    if operand.operator == "type" and operand.operands:
                        link_type = str(operand.operands[0])
                    elif operand.operator == "strength" and operand.operands:
                        strength = float(operand.operands[0])
                    elif operand.operator == "connects":
                        for node_ref in operand.operands:
                            if isinstance(node_ref, SchemeExpression) and node_ref.operator == "node-ref":
                                if node_ref.operands:
                                    node_names.append(str(node_ref.operands[0]))
                    elif operand.operator == "properties":
                        properties = self._scheme_to_properties(operand)
            
            # Find actual nodes
            nodes = []
            for node_name in node_names:
                node = atomspace.get_node(node_name)
                if node:
                    nodes.append(node)
            
            if len(nodes) >= 2:
                links.append(HypergraphLink(nodes, link_type, properties, strength))
        
        return links
    
    def _scheme_to_properties(self, props_expr: SchemeExpression) -> Dict[str, Any]:
        """Convert Scheme properties expression to dictionary."""
        properties = {}
        
        for prop_expr in props_expr.operands:
            if isinstance(prop_expr, SchemeExpression):
                key = prop_expr.operator
                if len(prop_expr.operands) == 1:
                    if isinstance(prop_expr.operands[0], SchemeExpression):
                        # Nested properties
                        properties[key] = self._scheme_to_properties(prop_expr.operands[0])
                    else:
                        # Simple value
                        properties[key] = prop_expr.operands[0]
                else:
                    # List of values
                    properties[key] = prop_expr.operands
        
        return properties
    
    def create_cognitive_grammar_pattern(
        self, 
        pattern_name: str, 
        components: List[str],
        relationships: List[Tuple[str, str, str]]
    ) -> SchemeExpression:
        """
        Create a cognitive grammar pattern in Scheme representation.
        
        Parameters
        ----------
        pattern_name : str
            Name of the cognitive pattern
        components : List[str]
            List of component node types
        relationships : List[Tuple[str, str, str]]
            List of (source, target, relation_type) tuples
            
        Returns
        -------
        SchemeExpression
            Scheme representation of the cognitive grammar pattern
        """
        # Create component expressions
        component_exprs = [
            SchemeExpression("component", [comp]) for comp in components
        ]
        
        # Create relationship expressions
        relation_exprs = []
        for source, target, rel_type in relationships:
            relation_exprs.append(
                SchemeExpression(
                    "relation",
                    [
                        SchemeExpression("source", [source]),
                        SchemeExpression("target", [target]),
                        SchemeExpression("type", [rel_type])
                    ]
                )
            )
        
        return SchemeExpression(
            "CognitivePattern",
            [
                SchemeExpression("name", [pattern_name]),
                SchemeExpression("components", component_exprs),
                SchemeExpression("relationships", relation_exprs)
            ]
        )
    
    def parse_scheme_string(self, scheme_str: str) -> SchemeExpression:
        """Parse a Scheme string into a SchemeExpression."""
        return self.parser.parse(scheme_str)
    
    def validate_scheme_syntax(self, scheme_str: str) -> Dict[str, Any]:
        """
        Validate Scheme syntax and cognitive grammar compliance.
        
        Parameters
        ----------
        scheme_str : str
            Scheme string to validate
            
        Returns
        -------
        dict
            Validation results
        """
        results = {
            "valid_syntax": False,
            "cognitive_compliant": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Parse the scheme string
            expr = self.parser.parse(scheme_str)
            results["valid_syntax"] = True
            
            # Check cognitive grammar compliance
            if self._is_cognitive_compliant(expr):
                results["cognitive_compliant"] = True
            else:
                results["warnings"].append("Not using standard cognitive primitives")
            
        except Exception as e:
            results["errors"].append(f"Syntax error: {e}")
        
        return results
    
    def _is_cognitive_compliant(self, expr: SchemeExpression) -> bool:
        """Check if expression uses standard cognitive primitives."""
        if expr.operator in self._cognitive_primitives.values():
            return True
        
        if expr.operator in ["AtomSpace", "CognitivePattern", "nodes", "links"]:
            return True
        
        # Check recursively
        for operand in expr.operands:
            if isinstance(operand, SchemeExpression):
                if not self._is_cognitive_compliant(operand):
                    return False
        
        return False