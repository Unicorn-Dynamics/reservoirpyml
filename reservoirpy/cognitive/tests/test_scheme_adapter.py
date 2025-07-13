"""
Tests for Scheme Adapter
========================

Test suite for Scheme S-expression cognitive grammar adapter.
"""

import pytest
from ..scheme_adapter import SchemeAdapter, SchemeExpression, SchemeParser
from ..hypergraph import AtomSpace, HypergraphNode, HypergraphLink


class TestSchemeParser:
    """Test SchemeParser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = SchemeParser()
    
    def test_simple_expression(self):
        """Test parsing simple S-expression."""
        scheme_str = "(+ 1 2)"
        expr = self.parser.parse(scheme_str)
        
        assert isinstance(expr, SchemeExpression)
        assert expr.operator == "+"
        assert expr.operands == [1, 2]
    
    def test_nested_expression(self):
        """Test parsing nested S-expression."""
        scheme_str = "(ReservoirNode (name test) (units 50))"
        expr = self.parser.parse(scheme_str)
        
        assert expr.operator == "ReservoirNode"
        assert len(expr.operands) == 2


class TestSchemeAdapter:
    """Test SchemeAdapter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = SchemeAdapter()
    
    def test_simple_atomspace_to_scheme(self):
        """Test converting simple AtomSpace to Scheme."""
        atomspace = AtomSpace("test")
        
        node1 = HypergraphNode("node1", "reservoir")
        node2 = HypergraphNode("node2", "readout")
        link = HypergraphLink([node1, node2], "connection")
        
        atomspace.add_link(link)
        
        scheme_expr = self.adapter.atomspace_to_scheme(atomspace)
        
        assert isinstance(scheme_expr, SchemeExpression)
        assert scheme_expr.operator == "AtomSpace"
    
    def test_scheme_to_atomspace(self):
        """Test converting Scheme back to AtomSpace."""
        # Create a simple scheme expression
        nodes_expr = SchemeExpression("nodes", [
            SchemeExpression("ReservoirNode", [
                SchemeExpression("name", ["test_node"]),
                SchemeExpression("type", ["reservoir"])
            ])
        ])
        
        atomspace_expr = SchemeExpression("AtomSpace", [
            SchemeExpression("name", ["test"]),
            nodes_expr,
            SchemeExpression("links", [])
        ])
        
        atomspace = self.adapter.scheme_to_atomspace(atomspace_expr)
        
        assert isinstance(atomspace, AtomSpace)
        assert atomspace.name == "test"
    
    def test_cognitive_pattern_creation(self):
        """Test creating cognitive grammar patterns."""
        pattern = self.adapter.create_cognitive_grammar_pattern(
            "test_pattern",
            ["input", "reservoir", "output"],
            [("input", "reservoir", "connection"), ("reservoir", "output", "connection")]
        )
        
        assert isinstance(pattern, SchemeExpression)
        assert pattern.operator == "CognitivePattern"
    
    def test_scheme_validation(self):
        """Test Scheme syntax validation."""
        valid_scheme = "(ReservoirNode (name test))"
        invalid_scheme = "(ReservoirNode (name test"
        
        valid_results = self.adapter.validate_scheme_syntax(valid_scheme)
        invalid_results = self.adapter.validate_scheme_syntax(invalid_scheme)
        
        assert valid_results["valid_syntax"]
        assert not invalid_results["valid_syntax"]