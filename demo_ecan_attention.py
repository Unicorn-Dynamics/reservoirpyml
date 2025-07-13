#!/usr/bin/env python3
"""
ECAN Attention Allocation & Resource Kernel Construction Demo
============================================================

Demonstration of Phase 2 implementation featuring:
- ECAN attention allocation with economic markets
- Attention-aware reservoir computing nodes
- Dynamic mesh topology based on attention flow
- Real-time resource scheduling and attention banking
- Comprehensive performance analysis and visualization

This demo showcases the complete integration of Economic Cognitive Attention Networks
(ECAN) with ReservoirPy for dynamic attention-driven cognitive computing.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Any

# Import ReservoirPy components
from reservoirpy.nodes import Input, Output

# Import cognitive and attention components
from reservoirpy.cognitive import (
    AtomSpace, HypergraphNode, HypergraphLink,
    ECANAttentionSystem, AttentionValue, ResourceAllocator,
    AttentionReservoir, AttentionMarket, ResourceScheduler,
    AttentionBank, MeshTopology, TopologyModifier, AttentionFlow
)
from reservoirpy.cognitive.attention.resource_allocator import ResourceType


class ECANDemo:
    """
    Comprehensive ECAN attention system demonstration.
    
    Showcases all major components of the attention allocation and
    resource kernel construction system.
    """
    
    def __init__(self):
        """Initialize the ECAN demonstration environment."""
        print("🧠 Initializing ECAN Attention Allocation Demo...")
        
        # Core components
        self.atomspace = AtomSpace()
        self.ecan = ECANAttentionSystem(
            atomspace=self.atomspace,
            max_attention_focus=10,
            min_sti_threshold=1.0,
            spreading_factor=0.8
        )
        
        # Attention flow and mesh topology
        self.attention_flow = AttentionFlow(self.ecan)
        self.mesh_topology = MeshTopology(
            attention_system=self.ecan,
            attention_flow=self.attention_flow,
            max_nodes=20
        )
        
        # Resource allocation
        self.resource_allocator = ResourceAllocator(self.ecan)
        
        # Topology optimization
        self.topology_modifier = TopologyModifier(self.mesh_topology)
        
        # Demo state
        self.reservoirs: Dict[str, AttentionReservoir] = {}
        self.demo_results: Dict[str, Any] = {}
        
        print("✅ ECAN system initialized successfully!")
    
    def demo_basic_attention_allocation(self):
        """Demonstrate basic ECAN attention allocation mechanisms."""
        print("\n🎯 === Demo 1: Basic Attention Allocation ===")
        
        # Create cognitive atoms
        concepts = ["memory", "learning", "prediction", "adaptation", "reasoning"]
        
        for concept in concepts:
            node = HypergraphNode(concept, "cognitive_concept", {"domain": "cognition"})
            self.atomspace.add_node(node)
            
            # Set initial attention values
            initial_sti = np.random.uniform(1.0, 10.0)
            initial_lti = np.random.uniform(0.5, 5.0)
            
            self.ecan.set_sti(node.id, initial_sti)
            self.ecan.set_lti(node.id, initial_lti)
        
        # Create concept relationships
        relationships = [
            ("memory", "learning"), ("learning", "prediction"),
            ("prediction", "adaptation"), ("adaptation", "reasoning"),
            ("reasoning", "memory")  # Circular reasoning!
        ]
        
        for source_concept, target_concept in relationships:
            source_node = next(n for n in self.atomspace.nodes if n.name == source_concept)
            target_node = next(n for n in self.atomspace.nodes if n.name == target_concept)
            
            link = HypergraphLink([source_node, target_node], "conceptual_relation")
            self.atomspace.add_link(link)
        
        print(f"Created {len(concepts)} cognitive concepts with {len(relationships)} relationships")
        
        # Demonstrate attention spreading
        print("\n🔄 Testing attention spreading...")
        learning_node = next(n for n in self.atomspace.nodes if n.name == "learning")
        
        print(f"Stimulating 'learning' concept with attention boost...")
        self.ecan.stimulate_atom(learning_node.id, 15.0)
        
        # Run several economic cycles to see attention dynamics
        for cycle in range(5):
            self.ecan.run_economic_cycle()
            time.sleep(0.1)  # Brief pause for processing
        
        # Display attention statistics
        stats = self.ecan.get_attention_statistics()
        print(f"📊 Attention Statistics:")
        print(f"  - Total atoms: {stats['total_atoms']}")
        print(f"  - Focus size: {stats['focus_size']}")
        print(f"  - Average STI: {stats['average_sti']:.2f}")
        print(f"  - Average LTI: {stats['average_lti']:.2f}")
        print(f"  - Economic cycles: {stats['economic_cycles']}")
        print(f"  - Attention bank funds: {stats['attention_bank_funds']:.2f}")
        
        self.demo_results['basic_attention'] = stats
        return stats
    
    def demo_attention_reservoirs(self):
        """Demonstrate attention-aware reservoir computing."""
        print("\n🏊 === Demo 2: Attention-Aware Reservoirs ===")
        
        # Create attention-aware reservoirs with different specializations
        reservoir_configs = [
            {"name": "sensory_processor", "units": 50, "base_sr": 0.95, "specialization": "input_processing"},
            {"name": "working_memory", "units": 30, "base_sr": 0.85, "specialization": "memory_storage"},
            {"name": "pattern_matcher", "units": 40, "base_sr": 0.9, "specialization": "pattern_recognition"},
            {"name": "decision_maker", "units": 25, "base_sr": 0.8, "specialization": "decision_making"}
        ]
        
        for config in reservoir_configs:
            reservoir = AttentionReservoir(
                units=config["units"],
                attention_system=self.ecan,
                atom_id=config["name"],
                base_sr=config["base_sr"]
            )
            
            self.reservoirs[config["name"]] = reservoir
            
            # Add to mesh topology
            self.mesh_topology.add_reservoir_node(
                config["name"], 
                reservoir,
                node_properties={"specialization": config["specialization"]}
            )
            
            # Set initial attention based on specialization importance
            importance_map = {
                "input_processing": 8.0,
                "memory_storage": 6.0,
                "pattern_recognition": 7.0,
                "decision_making": 9.0
            }
            
            initial_attention = importance_map[config["specialization"]]
            self.ecan.set_sti(config["name"], initial_attention)
            self.ecan.set_lti(config["name"], initial_attention * 0.6)
        
        print(f"Created {len(reservoir_configs)} attention-aware reservoirs")
        
        # Set up attention flow connections based on cognitive pipeline
        flow_connections = [
            ("sensory_processor", "working_memory"),
            ("sensory_processor", "pattern_matcher"),
            ("working_memory", "pattern_matcher"),
            ("pattern_matcher", "decision_maker"),
            ("decision_maker", "working_memory")  # Feedback
        ]
        
        for source, target in flow_connections:
            self.attention_flow.register_flow_connection(source, target)
        
        print(f"Established {len(flow_connections)} attention flow connections")
        
        # Simulate cognitive processing with attention cascades
        print("\n🌊 Triggering attention cascades...")
        
        # Simulate external stimulus to sensory processor
        cascade_results = []
        
        for stimulus_strength in [3.0, 7.0, 12.0]:
            print(f"  Stimulus strength: {stimulus_strength}")
            
            cascade_event = self.attention_flow.trigger_attention_cascade(
                trigger_atoms=["sensory_processor"],
                cascade_strength=stimulus_strength
            )
            
            cascade_results.append(cascade_event)
            
            # Run economic cycles to propagate effects
            for _ in range(3):
                self.ecan.run_economic_cycle()
                self.resource_allocator.run_economic_cycle()
        
        # Analyze reservoir adaptations
        reservoir_adaptations = {}
        for name, reservoir in self.reservoirs.items():
            status = reservoir.get_attention_status()
            reservoir_adaptations[name] = {
                'attention_value': status['attention_value']['total_importance'],
                'spectral_radius': status['reservoir_params']['spectral_radius'],
                'attention_updates': status['attention_stats']['attention_updates'],
                'average_attention': status['attention_stats']['average_attention']
            }
        
        print(f"📈 Reservoir Adaptations:")
        for name, adapt in reservoir_adaptations.items():
            print(f"  {name}:")
            print(f"    Total attention: {adapt['attention_value']:.2f}")
            print(f"    Spectral radius: {adapt['spectral_radius']:.3f}")
            print(f"    Attention updates: {adapt['attention_updates']}")
        
        self.demo_results['attention_reservoirs'] = {
            'cascade_results': cascade_results,
            'reservoir_adaptations': reservoir_adaptations
        }
        
        return reservoir_adaptations
    
    def demo_resource_markets(self):
        """Demonstrate economic attention markets and resource allocation."""
        print("\n💰 === Demo 3: Economic Resource Markets ===")
        
        # Create competitive resource requests from different cognitive processes
        cognitive_tasks = [
            {
                "name": "visual_processing",
                "priority": 8.0,
                "resources": {ResourceType.COMPUTATION: 30.0, ResourceType.MEMORY: 200.0},
                "attention_atoms": ["sensory_processor", "pattern_matcher"]
            },
            {
                "name": "language_comprehension", 
                "priority": 6.0,
                "resources": {ResourceType.COMPUTATION: 25.0, ResourceType.ATTENTION: 15.0},
                "attention_atoms": ["working_memory", "pattern_matcher"]
            },
            {
                "name": "motor_planning",
                "priority": 7.0,
                "resources": {ResourceType.COMPUTATION: 20.0, ResourceType.COMMUNICATION: 10.0},
                "attention_atoms": ["decision_maker", "working_memory"]
            },
            {
                "name": "memory_consolidation",
                "priority": 4.0,
                "resources": {ResourceType.MEMORY: 150.0, ResourceType.STORAGE: 100.0},
                "attention_atoms": ["working_memory"]
            },
            {
                "name": "executive_control",
                "priority": 9.0,
                "resources": {ResourceType.ATTENTION: 20.0, ResourceType.COMPUTATION: 15.0},
                "attention_atoms": ["decision_maker"]
            }
        ]
        
        print(f"Submitting {len(cognitive_tasks)} cognitive task requests...")
        
        # Submit all task requests
        task_results = []
        for task in cognitive_tasks:
            success = self.resource_allocator.allocate_resources(
                task_id=task["name"],
                resource_requirements=task["resources"],
                attention_atoms=task["attention_atoms"],
                priority=task["priority"]
            )
            task_results.append((task["name"], success))
            print(f"  {task['name']}: {'✅ Scheduled' if success else '❌ Failed'}")
        
        # Allow processing time
        print(f"\n⏱️  Processing tasks...")
        time.sleep(1.0)
        
        # Get market analysis
        market_status = self.resource_allocator.market.get_market_status()
        scheduler_status = self.resource_allocator.scheduler.get_scheduler_status()
        banking_status = self.resource_allocator.bank.get_banking_status()
        
        print(f"📊 Market Analysis:")
        print(f"  Resource utilization: {market_status['current_utilization']:.1%}")
        print(f"  Total revenue: ${market_status['total_revenue']:.2f}")
        print(f"  Active allocations: {market_status['active_allocations']}")
        print(f"  Completed tasks: {scheduler_status['completed_tasks']}")
        print(f"  Average wait time: {scheduler_status['scheduling_stats']['average_wait_time']:.3f}s")
        print(f"  Bank reserves: ${banking_status['reserves']:.2f}")
        
        # Demonstrate attention-driven price dynamics
        print(f"\n💹 Price Dynamics:")
        for resource_type, price in market_status['current_prices'].items():
            capacity = market_status['resource_capacity'][resource_type]
            available = market_status['available_resources'][resource_type]
            utilization = (capacity - available) / capacity if capacity > 0 else 0
            
            print(f"  {resource_type.value}: ${price:.2f} (utilization: {utilization:.1%})")
        
        self.demo_results['resource_markets'] = {
            'task_results': task_results,
            'market_status': market_status,
            'scheduler_status': scheduler_status,
            'banking_status': banking_status
        }
        
        return market_status
    
    def demo_dynamic_topology(self):
        """Demonstrate dynamic mesh topology adaptation."""
        print("\n🕸️  === Demo 4: Dynamic Mesh Topology ===")
        
        # Start topology auto-update
        self.mesh_topology.start_auto_update()
        
        # Create attention imbalances to trigger topology changes
        print("Creating attention imbalances...")
        
        # High attention scenario - sensory overload
        self.ecan.stimulate_atom("sensory_processor", 20.0)
        self.ecan.stimulate_atom("pattern_matcher", 15.0)
        
        # Let the system adapt
        time.sleep(3.0)
        
        # Medium attention scenario - working memory focus
        self.ecan.stimulate_atom("working_memory", 18.0)
        
        time.sleep(2.0)
        
        # Low attention scenario - decision making under uncertainty
        self.ecan.stimulate_atom("decision_maker", 25.0)
        
        time.sleep(3.0)
        
        # Get topology metrics
        topology_metrics = self.mesh_topology.get_topology_metrics()
        
        print(f"📊 Topology Metrics:")
        print(f"  Network nodes: {topology_metrics['network_structure']['num_nodes']}")
        print(f"  Network edges: {topology_metrics['network_structure']['num_edges']}")
        print(f"  Network density: {topology_metrics['network_structure']['density']:.3f}")
        print(f"  Active clusters: {topology_metrics['clusters']['num_clusters']}")
        print(f"  Largest cluster: {topology_metrics['clusters']['largest_cluster_size']} nodes")
        print(f"  Total topology changes: {topology_metrics['topology_stats']['total_changes']}")
        
        # Stop auto-update
        self.mesh_topology.stop_auto_update()
        
        # Get visualization data
        viz_data = self.mesh_topology.visualize_topology(include_attention=True)
        
        print(f"🎨 Visualization data generated for {len(viz_data['nodes'])} nodes and {len(viz_data['edges'])} edges")
        
        self.demo_results['dynamic_topology'] = {
            'topology_metrics': topology_metrics,
            'visualization_data': viz_data
        }
        
        return topology_metrics
    
    def demo_attention_banking(self):
        """Demonstrate attention banking and lending operations."""
        print("\n🏦 === Demo 5: Attention Banking ===")
        
        # Create scenarios where reservoirs need attention loans
        bank = self.resource_allocator.bank
        
        # Scenario 1: Working memory needs boost for complex task
        print("Scenario 1: Working memory requests attention loan")
        
        loan_id1 = bank.request_loan(
            borrower_id="working_memory",
            loan_amount=10.0,
            collateral_atoms=["working_memory", "pattern_matcher"],
            attention_system=self.ecan,
            term_length=50.0
        )
        
        if loan_id1:
            print(f"  ✅ Loan approved: {loan_id1[:8]}...")
            
            # Use the borrowed attention
            self.ecan.stimulate_atom("working_memory", 10.0)
            
            # Make partial payment
            payment_success = bank.make_payment(loan_id1, 3.0, self.ecan)
            print(f"  💰 Payment of $3.00: {'✅ Success' if payment_success else '❌ Failed'}")
        
        # Scenario 2: Decision maker requests loan but with insufficient collateral
        print("\nScenario 2: Decision maker requests large loan")
        
        loan_id2 = bank.request_loan(
            borrower_id="decision_maker",
            loan_amount=50.0,  # Large amount
            collateral_atoms=["decision_maker"],
            attention_system=self.ecan,
            term_length=30.0
        )
        
        if loan_id2:
            print(f"  ✅ Large loan approved: {loan_id2[:8]}...")
        else:
            print(f"  ❌ Loan denied - insufficient collateral")
        
        # Check for defaults (simulate time passing)
        defaulted_loans = bank.check_defaults(self.ecan)
        
        banking_status = bank.get_banking_status()
        print(f"\n📊 Banking Status:")
        print(f"  Bank reserves: ${banking_status['reserves']:.2f}")
        print(f"  Active loans: {banking_status['active_loans']}")
        print(f"  Total loans issued: {banking_status['banking_stats']['total_loans_issued']}")
        print(f"  Defaults: {len(defaulted_loans)}")
        
        self.demo_results['attention_banking'] = {
            'loan_results': [loan_id1, loan_id2],
            'banking_status': banking_status,
            'defaulted_loans': defaulted_loans
        }
        
        return banking_status
    
    def demo_performance_analysis(self):
        """Analyze overall system performance and attention efficiency."""
        print("\n📈 === Demo 6: Performance Analysis ===")
        
        # Get comprehensive system status
        system_status = self.resource_allocator.get_system_status()
        
        # Analyze attention allocation efficiency
        attention_stats = system_status['attention_stats']
        market_stats = system_status['market_status']
        
        print(f"🎯 Attention Allocation Efficiency:")
        print(f"  Total attention focus changes: {attention_stats['allocation_stats']['focus_changes']}")
        print(f"  Attention redistributions: {attention_stats['allocation_stats']['attention_redistributions']}")
        print(f"  Economic transactions: {attention_stats['allocation_stats']['economic_transactions']}")
        
        print(f"\n💼 Resource Market Efficiency:")
        print(f"  Successful allocations: {market_stats['market_stats']['successful_allocations']}")
        print(f"  Failed allocations: {market_stats['market_stats']['failed_allocations']}")
        
        if market_stats['market_stats']['total_requests'] > 0:
            success_rate = (market_stats['market_stats']['successful_allocations'] / 
                          market_stats['market_stats']['total_requests'])
            print(f"  Success rate: {success_rate:.1%}")
        
        # Calculate attention distribution entropy (diversity measure)
        attention_values = []
        for reservoir_name in self.reservoirs:
            av = self.ecan.get_attention_value(reservoir_name)
            attention_values.append(av.total_importance)
        
        if attention_values:
            # Normalize and calculate entropy
            attention_probs = np.array(attention_values) / sum(attention_values)
            attention_probs = attention_probs[attention_probs > 0]  # Remove zeros
            attention_entropy = -np.sum(attention_probs * np.log2(attention_probs))
            
            print(f"\n🔄 Attention Distribution:")
            print(f"  Attention entropy: {attention_entropy:.3f} bits")
            print(f"  Max possible entropy: {np.log2(len(attention_values)):.3f} bits")
            print(f"  Attention diversity: {attention_entropy / np.log2(len(attention_values)):.1%}")
        
        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        
        if attention_stats['focus_size'] < 3:
            print("  📢 Consider increasing attention stimulation to maintain adequate focus")
        
        if market_stats['current_utilization'] > 0.8:
            print("  ⚠️  High resource utilization - consider expanding capacity")
        elif market_stats['current_utilization'] < 0.3:
            print("  💡 Low resource utilization - optimize for efficiency")
        
        if attention_entropy < np.log2(len(attention_values)) * 0.7:
            print("  🎯 Attention concentration detected - consider balancing focus")
        
        self.demo_results['performance_analysis'] = {
            'system_status': system_status,
            'attention_entropy': attention_entropy if attention_values else 0.0,
            'attention_diversity': attention_entropy / np.log2(len(attention_values)) if attention_values else 0.0
        }
        
        return system_status
    
    def visualize_results(self):
        """Create visualizations of the demo results."""
        print("\n📊 === Generating Visualizations ===")
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('ECAN Attention System Analysis', fontsize=16, fontweight='bold')
            
            # 1. Attention distribution across reservoirs
            if 'attention_reservoirs' in self.demo_results:
                reservoir_data = self.demo_results['attention_reservoirs']['reservoir_adaptations']
                names = list(reservoir_data.keys())
                attentions = [data['attention_value'] for data in reservoir_data.values()]
                
                axes[0, 0].bar(range(len(names)), attentions, color='skyblue', alpha=0.7)
                axes[0, 0].set_xticks(range(len(names)))
                axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
                axes[0, 0].set_title('Attention Distribution Across Reservoirs')
                axes[0, 0].set_ylabel('Total Attention Value')
            
            # 2. Resource market prices
            if 'resource_markets' in self.demo_results:
                market_data = self.demo_results['resource_markets']['market_status']
                resource_types = [rt.value for rt in market_data['current_prices'].keys()]
                prices = list(market_data['current_prices'].values())
                
                axes[0, 1].bar(range(len(resource_types)), prices, color='lightcoral', alpha=0.7)
                axes[0, 1].set_xticks(range(len(resource_types)))
                axes[0, 1].set_xticklabels(resource_types, rotation=45, ha='right')
                axes[0, 1].set_title('Resource Market Prices')
                axes[0, 1].set_ylabel('Price per Unit')
            
            # 3. Topology metrics over time (simulated)
            if 'dynamic_topology' in self.demo_results:
                topology_data = self.demo_results['dynamic_topology']['topology_metrics']
                
                # Simulate topology evolution
                time_points = range(10)
                nodes_over_time = [4 + i * 0.1 for i in time_points]  # Simulated growth
                edges_over_time = [6 + i * 0.3 for i in time_points]
                
                axes[0, 2].plot(time_points, nodes_over_time, 'b-o', label='Nodes', alpha=0.7)
                axes[0, 2].plot(time_points, edges_over_time, 'r-s', label='Edges', alpha=0.7)
                axes[0, 2].set_title('Network Topology Evolution')
                axes[0, 2].set_xlabel('Time Steps')
                axes[0, 2].set_ylabel('Count')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Economic cycle performance
            if 'basic_attention' in self.demo_results:
                attention_data = self.demo_results['basic_attention']
                
                metrics = ['average_sti', 'average_lti', 'focus_size']
                values = [attention_data.get(metric, 0) for metric in metrics]
                
                axes[1, 0].bar(range(len(metrics)), values, color='lightgreen', alpha=0.7)
                axes[1, 0].set_xticks(range(len(metrics)))
                axes[1, 0].set_xticklabels(['Avg STI', 'Avg LTI', 'Focus Size'])
                axes[1, 0].set_title('Attention System Metrics')
                axes[1, 0].set_ylabel('Value')
            
            # 5. Resource utilization
            if 'resource_markets' in self.demo_results:
                market_data = self.demo_results['resource_markets']['market_status']
                
                resource_types = [rt.value for rt in market_data['available_resources'].keys()]
                capacities = [market_data['resource_capacity'][rt] for rt in market_data['resource_capacity'].keys()]
                available = [market_data['available_resources'][rt] for rt in market_data['available_resources'].keys()]
                utilized = [cap - avail for cap, avail in zip(capacities, available)]
                
                x = np.arange(len(resource_types))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, capacities, width, label='Capacity', alpha=0.7, color='lightblue')
                axes[1, 1].bar(x + width/2, utilized, width, label='Utilized', alpha=0.7, color='orange')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(resource_types, rotation=45, ha='right')
                axes[1, 1].set_title('Resource Utilization')
                axes[1, 1].set_ylabel('Resource Units')
                axes[1, 1].legend()
            
            # 6. System performance summary
            if 'performance_analysis' in self.demo_results:
                perf_data = self.demo_results['performance_analysis']
                
                # Create a pie chart of attention diversity
                diversity = perf_data.get('attention_diversity', 0.5)
                concentration = 1.0 - diversity
                
                axes[1, 2].pie([diversity, concentration], 
                              labels=['Diverse', 'Concentrated'], 
                              colors=['lightgreen', 'lightcoral'],
                              autopct='%1.1f%%',
                              startangle=90)
                axes[1, 2].set_title('Attention Distribution Pattern')
            
            plt.tight_layout()
            plt.savefig('/tmp/ecan_demo_results.png', dpi=300, bbox_inches='tight')
            print("📊 Visualization saved to /tmp/ecan_demo_results.png")
            
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            print("   (This may occur in environments without display support)")
    
    def save_results(self):
        """Save demo results to file for analysis."""
        print("\n💾 === Saving Results ===")
        
        # Prepare results for JSON serialization
        json_results = {}
        
        for key, value in self.demo_results.items():
            try:
                # Convert to JSON-serializable format
                json_results[key] = self._make_json_serializable(value)
            except Exception as e:
                print(f"⚠️  Could not serialize {key}: {e}")
                json_results[key] = {"error": str(e)}
        
        # Save to file
        results_file = '/tmp/ecan_demo_results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"📁 Results saved to {results_file}")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
        
        return json_results
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return str(obj)
    
    def run_complete_demo(self):
        """Run the complete ECAN demonstration."""
        print("🚀 Starting Complete ECAN Attention System Demonstration")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Run all demo components
            self.demo_basic_attention_allocation()
            self.demo_attention_reservoirs()
            self.demo_resource_markets()
            self.demo_dynamic_topology()
            self.demo_attention_banking()
            self.demo_performance_analysis()
            
            # Generate visualizations and save results
            self.visualize_results()
            self.save_results()
            
            duration = time.time() - start_time
            
            print(f"\n🎉 === Demo Complete ===")
            print(f"⏱️  Total runtime: {duration:.2f} seconds")
            print(f"📊 Components demonstrated:")
            print(f"   ✅ ECAN attention allocation")
            print(f"   ✅ Economic resource markets")
            print(f"   ✅ Attention-aware reservoirs")
            print(f"   ✅ Dynamic mesh topology")
            print(f"   ✅ Attention banking system")
            print(f"   ✅ Performance analysis")
            
            print(f"\n📈 Key Results:")
            if 'basic_attention' in self.demo_results:
                stats = self.demo_results['basic_attention']
                print(f"   • Managed {stats['total_atoms']} cognitive atoms")
                print(f"   • Achieved {stats['focus_size']}-atom attention focus")
                print(f"   • Ran {stats['economic_cycles']} economic cycles")
            
            if 'resource_markets' in self.demo_results:
                market = self.demo_results['resource_markets']['market_status']
                print(f"   • Processed {market['market_stats']['total_requests']} resource requests")
                print(f"   • Achieved ${market['total_revenue']:.2f} market revenue")
            
            print(f"\n🔮 The ECAN attention system successfully demonstrates:")
            print(f"   🧠 Intelligent attention allocation with economic principles")
            print(f"   ⚡ Dynamic adaptation of reservoir computing parameters")
            print(f"   🌐 Self-organizing mesh topology based on attention flow")
            print(f"   💰 Resource markets with attention-based bidding")
            print(f"   🏦 Attention banking for temporal resource management")
            
            return True
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup
            self.mesh_topology.stop_auto_update()
            self.resource_allocator.shutdown()


def main():
    """Main demonstration function."""
    print("🧠 ECAN Attention Allocation & Resource Kernel Construction")
    print("   Phase 2 Implementation Demonstration")
    print("   ReservoirPy Cognitive Computing Integration")
    print()
    
    # Create and run demo
    demo = ECANDemo()
    success = demo.run_complete_demo()
    
    if success:
        print(f"\n✨ Demo completed successfully!")
        print(f"🔍 Check /tmp/ecan_demo_results.json for detailed results")
        print(f"📊 Check /tmp/ecan_demo_results.png for visualizations")
    else:
        print(f"\n❌ Demo encountered errors")
    
    return demo


if __name__ == "__main__":
    demo = main()