"""Example 03: 2048-D Vector Symbolic Architecture (VSA) Holographic Weight Memory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aetheris_microbrains as mb

def main():
    print("=== Aetheris Microbrains: Holographic Weight Space VSA ===")
    
    vsa = mb.get_weight_space_vsa()
    print(f"VSA Memory Vector Dimensions: {vsa.dim}")
    print(f"Registered Domain Codebooks: {len(vsa.domain_codebook)}")

    target_domain = "tool_routing_v2"
    print(f"\nSearching nearest functional domains for '{target_domain}' in weight space...")
    nearest = vsa.find_nearest_domains(target_domain, top_k=3)
    
    for rank, item in enumerate(nearest, 1):
        print(f"  #{rank}: {item['domain']} (Cosine Similarity: {item['similarity']:.4f})")

if __name__ == "__main__":
    main()
