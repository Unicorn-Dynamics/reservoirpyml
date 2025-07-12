#!/usr/bin/env python3
"""
Test script for cognitive issues creation logic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'workflows', 'scripts'))

from create_cognitive_issues import PHASE_TEMPLATES

def test_phase_templates():
    """Test that all phase templates are properly defined."""
    print("🧪 Testing phase templates...")
    
    # Check that we have all 6 phases
    assert len(PHASE_TEMPLATES) == 6, f"Expected 6 phases, got {len(PHASE_TEMPLATES)}"
    print("✅ All 6 phases defined")
    
    # Check that each phase has required fields
    for phase_num, template in PHASE_TEMPLATES.items():
        assert isinstance(phase_num, int), f"Phase number should be int, got {type(phase_num)}"
        assert 1 <= phase_num <= 6, f"Phase number should be 1-6, got {phase_num}"
        
        assert 'title' in template, f"Phase {phase_num} missing title"
        assert 'body' in template, f"Phase {phase_num} missing body"
        
        assert template['title'].startswith(f"Phase {phase_num}:"), f"Phase {phase_num} title format incorrect"
        assert len(template['body']) > 1000, f"Phase {phase_num} body too short: {len(template['body'])} chars"
        
        # Check for key sections in body
        body = template['body']
        assert "## Objective" in body, f"Phase {phase_num} missing Objective section"
        assert "## Sub-Steps" in body, f"Phase {phase_num} missing Sub-Steps section"
        assert "## Acceptance Criteria" in body, f"Phase {phase_num} missing Acceptance Criteria section"
        assert "## Dependencies" in body, f"Phase {phase_num} missing Dependencies section"
        assert "## Timeline" in body, f"Phase {phase_num} missing Timeline section"
        
        # Check for ReservoirPy integration mentions
        assert "ReservoirPy" in body, f"Phase {phase_num} doesn't mention ReservoirPy integration"
        
        print(f"✅ Phase {phase_num}: {template['title'][:50]}... ✓")
    
    print("✅ All phase templates validated successfully!")

def test_issue_content_quality():
    """Test that issue content meets quality standards."""
    print("\n🧪 Testing issue content quality...")
    
    reservoir_keywords = ["reservoir", "ReservoirPy", "neural", "dynamics", "spectral", "learning"]
    cognitive_keywords = ["cognitive", "attention", "symbolic", "hypergraph", "consciousness"]
    
    for phase_num, template in PHASE_TEMPLATES.items():
        body = template['body'].lower()
        
        # Check for reservoir computing integration
        reservoir_mentions = sum(1 for keyword in reservoir_keywords if keyword.lower() in body)
        assert reservoir_mentions >= 2, f"Phase {phase_num} insufficient ReservoirPy integration mentions"
        
        # Check for cognitive computing concepts
        cognitive_mentions = sum(1 for keyword in cognitive_keywords if keyword.lower() in body)
        assert cognitive_mentions >= 2, f"Phase {phase_num} insufficient cognitive concepts"
        
        # Check for actionable tasks (checkboxes)
        task_count = body.count('- [ ]')
        assert task_count >= 5, f"Phase {phase_num} needs more actionable tasks: {task_count}"
        
        print(f"✅ Phase {phase_num}: {task_count} tasks, {reservoir_mentions} reservoir refs, {cognitive_mentions} cognitive refs")
    
    print("✅ Issue content quality validated!")

def test_phase_dependencies():
    """Test that phase dependencies are logical."""
    print("\n🧪 Testing phase dependencies...")
    
    # Phase 1 should have minimal dependencies
    phase1_body = PHASE_TEMPLATES[1]['body']
    assert "Phase" not in phase1_body.split("Dependencies")[1].split("Timeline")[0], "Phase 1 shouldn't depend on other phases"
    
    # Later phases should reference earlier ones
    for phase_num in range(2, 7):
        body = PHASE_TEMPLATES[phase_num]['body']
        dependencies_section = body.split("Dependencies")[1].split("Timeline")[0]
        
        # Should mention at least one earlier phase
        earlier_phase_mentioned = any(f"Phase {i}" in dependencies_section for i in range(1, phase_num))
        assert earlier_phase_mentioned, f"Phase {phase_num} should reference earlier phases"
        
        print(f"✅ Phase {phase_num}: Dependencies properly structured")
    
    print("✅ Phase dependencies validated!")

def main():
    """Run all tests."""
    print("🚀 Starting Cognitive Grammar Network Integration Tests\n")
    
    try:
        test_phase_templates()
        test_issue_content_quality()
        test_phase_dependencies()
        
        print("\n🎉 All tests passed! The cognitive integration workflow is ready.")
        print("\n📋 Summary:")
        print(f"   • {len(PHASE_TEMPLATES)} phase templates defined")
        print(f"   • Complete ReservoirPy integration planned")
        print(f"   • Comprehensive cognitive computing roadmap")
        print(f"   • Production-ready issue creation workflow")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()