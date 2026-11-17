"""Example 02: Dynamic Mixture of Experts (MoE) Tool Routing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aetheris_microbrains as mb

def main():
    print("=== Aetheris Microbrains: Top-2 MoE Tool Routing ===")
    
    router = mb.get_tool_routing_moe()
    task = "read configuration file and write patch to repository"
    print(f"Task description: '{task}'")
    
    decision = router.route(task)
    print("\nRouting Decision:")
    print(f"  Primary Tool:       {decision.selected_tool}")
    print(f"  Secondary Tool:     {decision.secondary_tool}")
    print(f"  Top-2 Experts:      {decision.top_experts}")
    print(f"  Confidence:         {decision.confidence:.1%}")
    print(f"  Conformal Passed:   {decision.conformal_passed}")
    print(f"  Rationale:          {decision.routing_rationale}")

if __name__ == "__main__":
    main()
