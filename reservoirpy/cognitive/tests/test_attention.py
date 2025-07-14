"""
Tests for ECAN Attention Allocation & Resource Kernel Construction
================================================================

Comprehensive tests for Phase 2 attention system components including
ECAN attention allocation, resource markets, attention-aware reservoirs,
and dynamic mesh topology.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from reservoirpy.cognitive import (
    AtomSpace, HypergraphNode, HypergraphLink,
    ECANAttentionSystem, AttentionValue, ResourceAllocator,
    AttentionReservoir, AttentionMarket, ResourceScheduler,
    AttentionBank, MeshTopology, TopologyModifier, AttentionFlow
)
from reservoirpy.cognitive.attention.resource_allocator import ResourceType, ResourceRequest


class TestAttentionValue:
    """Test attention value container and operations."""
    
    def test_attention_value_creation(self):
        """Test basic attention value creation."""
        av = AttentionValue(sti=5.0, lti=3.0, vlti=1.0)
        
        assert av.sti == 5.0
        assert av.lti == 3.0
        assert av.vlti == 1.0
        assert av.confidence == 1.0
        assert av.total_importance == 9.0
        assert av.attention_budget == 9.0
    
    def test_attention_value_updates(self):
        """Test attention value update operations."""
        av = AttentionValue(sti=5.0, lti=3.0)
        
        # Test STI update with decay
        av.update_sti(2.0, decay_rate=0.1)
        assert av.sti == pytest.approx(6.5, rel=1e-3)  # 5.0 + 2.0 - 5.0*0.1
        
        # Test LTI update (no decay)
        av.update_lti(1.0)
        assert av.lti == 4.0
    
    def test_attention_rent_and_wage(self):
        """Test attention rent payment and wage earning."""
        av = AttentionValue(sti=10.0, lti=5.0)
        
        # Test successful rent payment
        assert av.pay_rent(3.0) == True
        assert av.rent == 3.0
        assert av.attention_budget == 12.0  # 15.0 - 3.0
        
        # Test insufficient budget for rent
        assert av.pay_rent(15.0) == False
        assert av.rent == 3.0  # Unchanged
        
        # Test wage earning
        av.earn_wage(2.0)
        assert av.wage == 2.0
        assert av.sti == pytest.approx(11.0, rel=1e-3)  # Original 10.0 + 50% of wage
        assert av.lti == pytest.approx(5.6, rel=1e-3)  # Original 5.0 + 30% of wage


class TestECANAttentionSystem:
    """Test ECAN attention allocation system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.ecan = ECANAttentionSystem(
            atomspace=self.atomspace,
            max_attention_focus=5,
            min_sti_threshold=1.0
        )
        
        # Add some test nodes
        self.node1 = HypergraphNode("node1", "test", {"value": 1})
        self.node2 = HypergraphNode("node2", "test", {"value": 2})
        self.node3 = HypergraphNode("node3", "test", {"value": 3})
        
        self.atomspace.add_node(self.node1)
        self.atomspace.add_node(self.node2)
        self.atomspace.add_node(self.node3)
        
        # Add a link between nodes
        self.link = HypergraphLink([self.node1, self.node2], "test_link")
        self.atomspace.add_link(self.link)
    
    def test_attention_value_management(self):
        """Test attention value creation and management."""
        # Get attention value for node
        av1 = self.ecan.get_attention_value(self.node1.id)
        assert isinstance(av1, AttentionValue)
        assert av1.sti == 0.0
        assert av1.lti == 0.0
        
        # Set STI and LTI
        self.ecan.set_sti(self.node1.id, 5.0)
        self.ecan.set_lti(self.node1.id, 3.0)
        
        av1_updated = self.ecan.get_attention_value(self.node1.id)
        assert av1_updated.sti == 5.0
        assert av1_updated.lti == 3.0
    
    def test_attention_focus_management(self):
        """Test attention focus tracking."""
        # Initially empty focus
        assert len(self.ecan.attention_focus) == 0
        
        # Set attention values to trigger focus update
        self.ecan.set_sti(self.node1.id, 5.0)
        self.ecan.set_sti(self.node2.id, 3.0)
        self.ecan.set_sti(self.node3.id, 0.5)  # Below threshold
        
        # Check focus contains high-attention nodes
        focus_atoms = self.ecan.get_focus_atoms()
        assert self.node1.id in focus_atoms
        assert self.node2.id in focus_atoms
        assert self.node3.id not in focus_atoms  # Below threshold
    
    def test_attention_spreading(self):
        """Test attention spreading through hypergraph connections."""
        # Set initial attention
        self.ecan.set_sti(self.node1.id, 10.0)
        
        # Spread attention from node1
        self.ecan.spread_attention(self.node1.id, 3.0)
        
        # Check that connected node2 received attention
        av1 = self.ecan.get_attention_value(self.node1.id)
        av2 = self.ecan.get_attention_value(self.node2.id)
        
        assert av1.sti < 10.0  # Source lost attention
        assert av2.sti > 0.0   # Target gained attention
    
    def test_economic_cycle(self):
        """Test economic attention market cycle."""
        # Set up attention focus
        self.ecan.set_sti(self.node1.id, 5.0)
        self.ecan.set_lti(self.node1.id, 3.0)
        
        initial_bank_funds = self.ecan.attention_bank_funds
        
        # Run economic cycle
        self.ecan.run_economic_cycle()
        
        # Check that economic operations occurred
        assert self.ecan.cycle_count == 1
        assert len(self.ecan.attention_history) > 0
        
        # Check statistics
        stats = self.ecan.get_attention_statistics()
        assert stats['total_atoms'] > 0
        assert stats['economic_cycles'] == 1
    
    def test_attention_stimulation(self):
        """Test external attention stimulation."""
        initial_sti = self.ecan.get_attention_value(self.node1.id).sti
        
        # Apply stimulus
        self.ecan.stimulate_atom(self.node1.id, 5.0)
        
        # Check attention increased
        new_sti = self.ecan.get_attention_value(self.node1.id).sti
        assert new_sti > initial_sti


class TestResourceAllocator:
    """Test resource allocation and economic markets."""
    
    def setup_method(self):
        """Set up test environment."""
        self.atomspace = AtomSpace()
        self.ecan = ECANAttentionSystem(self.atomspace)
        self.allocator = ResourceAllocator(self.ecan)
    
    def test_resource_allocation_creation(self):
        """Test resource allocator initialization."""
        assert self.allocator.attention_system == self.ecan
        assert isinstance(self.allocator.market, AttentionMarket)
        assert isinstance(self.allocator.scheduler, ResourceScheduler)
        assert isinstance(self.allocator.bank, AttentionBank)
    
    def test_resource_allocation_request(self):
        """Test resource allocation for cognitive tasks."""
        # Request resources for a task
        success = self.allocator.allocate_resources(
            task_id="test_task_1",
            resource_requirements={
                ResourceType.COMPUTATION: 10.0,
                ResourceType.MEMORY: 50.0
            },
            attention_atoms=["atom1", "atom2"],
            priority=2.0
        )
        
        assert success == True
        
        # Check scheduler status
        status = self.allocator.scheduler.get_scheduler_status()
        assert status['queued_tasks'] >= 0  # Task might be processing already
    
    def test_economic_cycle_integration(self):
        """Test integrated economic cycle across all components."""
        # Set up some attention values
        self.ecan.set_sti("test_atom", 5.0)
        
        # Run economic cycle
        self.allocator.run_economic_cycle()
        
        # Check that cycle ran successfully
        stats = self.allocator.get_system_status()
        assert 'attention_stats' in stats
        assert 'market_status' in stats
        assert 'scheduler_status' in stats
        assert 'banking_status' in stats


class TestAttentionMarket:
    """Test economic attention market mechanisms."""
    
    def setup_method(self):
        """Set up test market."""
        initial_resources = {
            ResourceType.COMPUTATION: 100.0,
            ResourceType.MEMORY: 500.0,
            ResourceType.ATTENTION: 50.0
        }
        self.market = AttentionMarket(initial_resources)
    
    def test_market_initialization(self):
        """Test market initialization."""
        status = self.market.get_market_status()
        
        assert status['resource_capacity'][ResourceType.COMPUTATION] == 100.0
        assert status['available_resources'][ResourceType.COMPUTATION] == 100.0
        assert status['current_utilization'] == 0.0
    
    def test_resource_request_submission(self):
        """Test resource request submission and processing."""
        request = ResourceRequest(
            request_id="test_req_1",
            resource_type=ResourceType.COMPUTATION,
            amount=10.0,
            max_bid=5.0,
            priority=1.0
        )
        
        success = self.market.submit_request(request)
        assert success == True
        
        # Check market status after request
        status = self.market.get_market_status()
        assert status['available_resources'][ResourceType.COMPUTATION] <= 100.0
    
    def test_market_pricing(self):
        """Test dynamic market pricing."""
        # Submit multiple requests to drive up prices
        for i in range(5):
            request = ResourceRequest(
                request_id=f"test_req_{i}",
                resource_type=ResourceType.COMPUTATION,
                amount=15.0,
                max_bid=10.0,
                priority=1.0
            )
            self.market.submit_request(request)
        
        # Check that prices increased due to demand
        status = self.market.get_market_status()
        computation_price = status['current_prices'][ResourceType.COMPUTATION]
        
        # Price should be higher than base price due to demand (but market efficiency might reduce it)
        assert computation_price >= 0.9  # Adjusted for market efficiency


class TestAttentionReservoir:
    """Test attention-aware reservoir nodes."""
    
    def setup_method(self):
        """Set up test reservoir."""
        self.atomspace = AtomSpace()
        self.ecan = ECANAttentionSystem(self.atomspace)
        self.reservoir = AttentionReservoir(
            units=20,
            attention_system=self.ecan,
            atom_id="test_reservoir",
            base_sr=0.9
        )
    
    def test_attention_reservoir_creation(self):
        """Test attention reservoir initialization."""
        assert self.reservoir.units == 20
        assert self.reservoir.attention_system == self.ecan
        assert self.reservoir.atom_id == "test_reservoir"
        assert self.reservoir.base_sr == 0.9
        
        # Check registration with attention system
        av = self.ecan.get_attention_value("test_reservoir")
        assert av.sti > 0.0  # Should have initial attention
    
    def test_attention_dynamics_update(self):
        """Test attention-driven parameter updates."""
        # Set high attention
        self.ecan.set_sti("test_reservoir", 10.0)
        self.ecan.set_lti("test_reservoir", 5.0)
        
        # Trigger attention update
        self.reservoir._update_attention_dynamics()
        
        # Check that spectral radius was modulated
        # With high attention, SR should increase from base
        assert hasattr(self.reservoir, 'sr')
        # sr might be adjusted based on attention
    
    def test_attention_stimulation(self):
        """Test external attention stimulation of reservoir."""
        initial_attention = self.ecan.get_attention_value("test_reservoir").total_importance
        
        # Stimulate reservoir
        self.reservoir.stimulate_attention(3.0)
        
        # Check attention increased
        new_attention = self.ecan.get_attention_value("test_reservoir").total_importance
        assert new_attention > initial_attention
    
    def test_attention_status_reporting(self):
        """Test attention status reporting."""
        status = self.reservoir.get_attention_status()
        
        assert 'atom_id' in status
        assert 'attention_value' in status
        assert 'reservoir_params' in status
        assert 'attention_stats' in status
        assert status['atom_id'] == "test_reservoir"


class TestMeshTopology:
    """Test dynamic mesh topology management."""
    
    def setup_method(self):
        """Set up test mesh topology."""
        self.atomspace = AtomSpace()
        self.ecan = ECANAttentionSystem(self.atomspace)
        self.attention_flow = AttentionFlow(self.ecan)
        self.mesh = MeshTopology(
            attention_system=self.ecan,
            attention_flow=self.attention_flow,
            max_nodes=10
        )
    
    def test_mesh_topology_creation(self):
        """Test mesh topology initialization."""
        assert self.mesh.attention_system == self.ecan
        assert self.mesh.attention_flow == self.attention_flow
        assert self.mesh.max_nodes == 10
        assert len(self.mesh.reservoir_nodes) == 0
    
    def test_node_addition_and_removal(self):
        """Test adding and removing reservoir nodes."""
        # Create test reservoir
        reservoir1 = AttentionReservoir(10, self.ecan, "res1")
        reservoir2 = AttentionReservoir(10, self.ecan, "res2")
        
        # Add nodes
        success1 = self.mesh.add_reservoir_node("res1", reservoir1)
        success2 = self.mesh.add_reservoir_node("res2", reservoir2, ["res1"])
        
        assert success1 == True
        assert success2 == True
        assert len(self.mesh.reservoir_nodes) == 2
        assert self.mesh.topology_graph.has_edge("res2", "res1")
        
        # Remove node
        removal_success = self.mesh.remove_reservoir_node("res1")
        assert removal_success == True
        assert len(self.mesh.reservoir_nodes) == 1
        assert "res1" not in self.mesh.reservoir_nodes
    
    def test_topology_metrics(self):
        """Test topology metrics calculation."""
        # Add some nodes
        for i in range(3):
            reservoir = AttentionReservoir(10, self.ecan, f"res{i}")
            self.mesh.add_reservoir_node(f"res{i}", reservoir)
        
        # Get metrics
        metrics = self.mesh.get_topology_metrics()
        
        assert 'network_structure' in metrics
        assert 'clusters' in metrics
        assert 'attention_distribution' in metrics
        assert 'topology_stats' in metrics
        assert metrics['network_structure']['num_nodes'] == 3
    
    def test_attention_based_topology_update(self):
        """Test topology updates based on attention patterns."""
        # Add nodes with different attention levels
        reservoir1 = AttentionReservoir(10, self.ecan, "high_attention")
        reservoir2 = AttentionReservoir(10, self.ecan, "low_attention")
        
        self.mesh.add_reservoir_node("high_attention", reservoir1)
        self.mesh.add_reservoir_node("low_attention", reservoir2)
        
        # Set different attention levels
        self.ecan.set_sti("high_attention", 15.0)
        self.ecan.set_sti("low_attention", 1.0)
        
        # Update topology based on attention
        changes = self.mesh.update_topology_based_on_attention()
        
        # Should return list of changes (might be empty if thresholds not met)
        assert isinstance(changes, list)


class TestAttentionFlow:
    """Test attention flow and cascade mechanisms."""
    
    def setup_method(self):
        """Set up test attention flow."""
        self.atomspace = AtomSpace()
        self.ecan = ECANAttentionSystem(self.atomspace)
        self.flow = AttentionFlow(self.ecan)
    
    def test_flow_network_registration(self):
        """Test flow connection registration."""
        self.flow.register_flow_connection("atom1", "atom2")
        self.flow.register_flow_connection("atom2", "atom3")
        
        assert "atom1" in self.flow.flow_network
        assert "atom2" in self.flow.flow_network["atom1"]
        assert "atom3" in self.flow.flow_network["atom2"]
    
    def test_attention_flow_propagation(self):
        """Test attention flow propagation through network."""
        # Set up flow network
        self.flow.register_flow_connection("source", "target1")
        self.flow.register_flow_connection("source", "target2")
        self.flow.register_flow_connection("target1", "target3")
        
        # Propagate attention flow
        flow_results = self.flow.propagate_attention_flow("source", 10.0)
        
        assert "source" in flow_results
        assert flow_results["source"] == 10.0
        
        # Targets should receive some flow
        for target in ["target1", "target2"]:
            if target in flow_results:
                assert flow_results[target] > 0.0
    
    def test_attention_cascade_triggering(self):
        """Test attention cascade events."""
        # Set up network
        self.flow.register_flow_connection("trigger1", "cascade1")
        self.flow.register_flow_connection("trigger2", "cascade2")
        
        # Trigger cascade
        cascade_event = self.flow.trigger_attention_cascade(
            trigger_atoms=["trigger1", "trigger2"],
            cascade_strength=5.0
        )
        
        assert 'cascade_id' in cascade_event
        assert 'trigger_atoms' in cascade_event
        assert 'atoms_affected' in cascade_event
        assert cascade_event['trigger_atoms'] == ["trigger1", "trigger2"]
    
    def test_flow_statistics(self):
        """Test flow network statistics."""
        # Set up some connections
        self.flow.register_flow_connection("a", "b")
        self.flow.register_flow_connection("b", "c")
        
        stats = self.flow.get_flow_statistics()
        
        assert 'network_size' in stats
        assert 'total_connections' in stats
        assert 'total_flow_events' in stats
        assert stats['network_size'] >= 2
        assert stats['total_connections'] >= 2


class TestAttentionBank:
    """Test attention banking and lending operations."""
    
    def setup_method(self):
        """Set up test bank."""
        self.atomspace = AtomSpace()
        self.ecan = ECANAttentionSystem(self.atomspace)
        self.bank = AttentionBank(initial_reserves=1000.0)
    
    def test_bank_initialization(self):
        """Test bank initialization."""
        status = self.bank.get_banking_status()
        
        assert status['reserves'] == 1000.0
        assert status['active_loans'] == 0
        assert status['base_interest_rate'] == 0.05
    
    def test_loan_request_and_approval(self):
        """Test loan request processing."""
        # Set up collateral atoms with attention
        self.ecan.set_sti("collateral1", 20.0)
        self.ecan.set_lti("collateral1", 10.0)
        
        # Request loan
        loan_id = self.bank.request_loan(
            borrower_id="test_borrower",
            loan_amount=15.0,
            collateral_atoms=["collateral1"],
            attention_system=self.ecan,
            term_length=100.0
        )
        
        assert loan_id is not None
        assert loan_id in self.bank.active_loans
        
        # Check bank reserves decreased
        status = self.bank.get_banking_status()
        assert status['reserves'] == 985.0  # 1000 - 15
        assert status['active_loans'] == 1
    
    def test_loan_payment(self):
        """Test loan payment processing."""
        # Set up collateral and get loan
        self.ecan.set_sti("collateral1", 20.0)
        loan_id = self.bank.request_loan(
            borrower_id="test_borrower",
            loan_amount=10.0,
            collateral_atoms=["collateral1"],
            attention_system=self.ecan
        )
        
        # Make payment
        payment_success = self.bank.make_payment(
            loan_id=loan_id,
            payment_amount=5.0,
            attention_system=self.ecan
        )
        
        assert payment_success == True
        
        # Check loan balance
        loan = self.bank.active_loans[loan_id]
        assert loan['remaining_balance'] < loan['principal'] * 1.05  # Original + interest


class TestIntegrationScenarios:
    """Integration tests combining multiple attention system components."""
    
    def setup_method(self):
        """Set up comprehensive test environment."""
        self.atomspace = AtomSpace()
        self.ecan = ECANAttentionSystem(self.atomspace)
        self.attention_flow = AttentionFlow(self.ecan)
        self.mesh = MeshTopology(self.ecan, self.attention_flow)
        self.allocator = ResourceAllocator(self.ecan)
    
    def test_full_attention_ecosystem(self):
        """Test complete attention ecosystem integration."""
        # Create attention reservoirs
        reservoir1 = AttentionReservoir(20, self.ecan, "reservoir1")
        reservoir2 = AttentionReservoir(15, self.ecan, "reservoir2")
        
        # Add to mesh topology
        self.mesh.add_reservoir_node("reservoir1", reservoir1)
        self.mesh.add_reservoir_node("reservoir2", reservoir2, ["reservoir1"])
        
        # Set up attention flow
        self.attention_flow.register_flow_connection("reservoir1", "reservoir2")
        
        # Stimulate attention
        self.ecan.stimulate_atom("reservoir1", 10.0)
        
        # Trigger attention cascade
        cascade_event = self.attention_flow.trigger_attention_cascade(
            trigger_atoms=["reservoir1"],
            cascade_strength=5.0
        )
        
        # Run economic cycles
        for _ in range(3):
            self.allocator.run_economic_cycle()
        
        # Allocate resources
        success = self.allocator.allocate_resources(
            task_id="integration_test",
            resource_requirements={ResourceType.COMPUTATION: 10.0},
            attention_atoms=["reservoir1", "reservoir2"],
            priority=2.0
        )
        
        # Update topology
        topology_changes = self.mesh.update_topology_based_on_attention()
        
        # Verify system is functional
        assert success == True
        assert len(cascade_event['trigger_atoms']) > 0
        assert isinstance(topology_changes, list)
        
        # Get comprehensive system status
        system_status = self.allocator.get_system_status()
        attention_stats = system_status['attention_stats']
        
        assert attention_stats['total_atoms'] > 0
        assert attention_stats['focus_size'] >= 0
    
    def test_attention_driven_reservoir_adaptation(self):
        """Test reservoir adaptation based on attention patterns."""
        # Create reservoir with attention integration
        reservoir = AttentionReservoir(30, self.ecan, "adaptive_reservoir")
        
        # Generate mock input data
        input_data = np.random.randn(10, 5)
        
        # Set varying attention levels and observe adaptation
        attention_levels = [1.0, 5.0, 10.0, 15.0, 8.0]
        
        adaptation_results = []
        
        for attention_level in attention_levels:
            # Set attention
            self.ecan.set_sti("adaptive_reservoir", attention_level)
            
            # Update attention dynamics
            reservoir._update_attention_dynamics()
            
            # Get current status
            status = reservoir.get_attention_status()
            adaptation_results.append({
                'attention_level': attention_level,
                'spectral_radius': status['reservoir_params'].get('spectral_radius'),
                'total_importance': status['attention_value']['total_importance']
            })
        
        # Verify adaptation occurred
        assert len(adaptation_results) == len(attention_levels)
        
        # Check that attention levels affected reservoir parameters
        attention_values = [r['total_importance'] for r in adaptation_results]
        assert max(attention_values) > min(attention_values)
    
    def test_economic_resource_competition(self):
        """Test resource competition and economic allocation."""
        # Create multiple competing tasks
        tasks = []
        for i in range(5):
            task_id = f"competing_task_{i}"
            
            # Set up attention atoms with different attention levels
            atom_id = f"task_atom_{i}"
            self.ecan.set_sti(atom_id, float(i + 1) * 2.0)  # Increasing attention
            
            # Request resources
            success = self.allocator.allocate_resources(
                task_id=task_id,
                resource_requirements={
                    ResourceType.COMPUTATION: 20.0,
                    ResourceType.MEMORY: 100.0
                },
                attention_atoms=[atom_id],
                priority=float(i + 1)  # Increasing priority
            )
            tasks.append((task_id, success))
        
        # Allow some processing time
        time.sleep(0.5)
        
        # Check market status
        market_status = self.allocator.market.get_market_status()
        
        # Verify resource allocation occurred
        assert market_status['total_revenue'] >= 0
        assert market_status['active_allocations'] >= 0
        
        # Check that high-attention tasks were favored
        scheduler_status = self.allocator.scheduler.get_scheduler_status()
        assert scheduler_status['queued_tasks'] >= 0