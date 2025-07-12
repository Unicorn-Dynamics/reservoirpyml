#!/usr/bin/env python3
"""
Preview script for cognitive integration issues.
Shows what issues would be created without actually creating them.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from create_cognitive_issues import PHASE_TEMPLATES

def preview_issues(phase_input='all'):
    """Preview issues that would be created."""
    print("🔍 Cognitive Grammar Network Integration - Issue Preview")
    print("=" * 60)
    
    # Determine phases to preview
    if phase_input.lower() == 'all':
        phases = list(range(1, 7))
        print(f"📋 Previewing all {len(phases)} phases\n")
    else:
        try:
            phase_num = int(phase_input)
            if 1 <= phase_num <= 6:
                phases = [phase_num]
                print(f"📋 Previewing Phase {phase_num}\n")
            else:
                print("❌ Phase number must be between 1 and 6")
                return
        except ValueError:
            print("❌ Invalid input. Use 'all' or a number 1-6")
            return
    
    # Preview each phase
    for i, phase_num in enumerate(phases, 1):
        template = PHASE_TEMPLATES[phase_num]
        
        print(f"## Issue {i}: {template['title']}")
        print("-" * 50)
        
        # Count tasks and extract key info
        body = template['body']
        task_count = body.count('- [ ]')
        duration_line = [line for line in body.split('\n') if 'Duration' in line]
        duration = duration_line[0].split('**Duration**:')[1].strip() if duration_line else "Not specified"
        
        print(f"📝 Tasks: {task_count} actionable items")
        print(f"⏱️  Duration: {duration}")
        print(f"🏷️  Labels: enhancement, cognitive-integration, phase-{phase_num}, reservoir-computing")
        
        # Extract objective
        objective_start = body.find("## Objective") + len("## Objective")
        objective_end = body.find("## Sub-Steps")
        objective = body[objective_start:objective_end].strip()
        print(f"🎯 Objective: {objective[:100]}...")
        
        # Extract key integration points
        reservoir_mentions = body.lower().count('reservoirpy')
        cognitive_mentions = body.lower().count('cognitive')
        attention_mentions = body.lower().count('attention')
        
        print(f"🔗 Integration: {reservoir_mentions} ReservoirPy refs, {cognitive_mentions} cognitive refs, {attention_mentions} attention refs")
        
        if i < len(phases):
            print("\n" + "="*60 + "\n")
    
    # Summary
    total_tasks = sum(PHASE_TEMPLATES[p]['body'].count('- [ ]') for p in phases)
    total_phases = len(phases)
    
    print(f"\n🎯 Preview Summary:")
    print(f"   • {total_phases} issue(s) would be created")
    print(f"   • {total_tasks} total actionable tasks")
    print(f"   • Complete ReservoirPy cognitive integration roadmap")
    print(f"   • Estimated 36-48 weeks for full implementation")
    
    print(f"\n🚀 To create these issues, run the 'Cognitive Grammar Network Integration' workflow")

def main():
    """Main preview function."""
    if len(sys.argv) > 1:
        phase_input = sys.argv[1]
    else:
        phase_input = 'all'
    
    preview_issues(phase_input)

if __name__ == "__main__":
    main()