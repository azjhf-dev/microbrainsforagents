"""Example 04: Pre-Commit Counterfactual Simulation & Constitutional Veto Gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aetheris_microbrains as mb

def main():
    print("=== Aetheris Microbrains: Counterfactual Pre-Commit Simulation ===")
    
    sim = mb.get_counterfactual_pipeline()

    # Case 1: Safe clean patch
    safe_diff = "--- a/math.py\n+++ b/math.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n-    pass\n+    return a + b\n"
    print("\n--- Testing Safe Patch ---")
    outcome_safe = sim.simulate_edit_impact("src/math.py", safe_diff)
    print("Allowed:", outcome_safe.allowed)
    print("Status:", outcome_safe.status)
    print("Step Rollouts:", outcome_safe.step_predictions)

    # Case 2: Malformed risky patch (syntax hazard)
    risky_diff = "def calculate():\n    SyntaxError eval(raw_input)\n"
    print("\n--- Testing Risky Patch ---")
    outcome_risky = sim.simulate_edit_impact("src/math.py", risky_diff)
    print("Allowed:", outcome_risky.allowed)
    print("Status:", outcome_risky.status)
    print("Veto Reason:", outcome_risky.veto_reason)

if __name__ == "__main__":
    main()
