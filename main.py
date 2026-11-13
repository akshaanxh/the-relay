"""
================================================================================
THE RELAY: INTERACTIVE CLI & DEMO ENTRYPOINT
================================================================================
Author: Candidate (ML Engineering Intern Role, Vaani Research)
Usage:
  - Interactive Query:
      python main.py --query "What is the Mimi neural audio codec bitrate in Moshi?"
  - Live Groq Query:
      python main.py --query "What is the scaling factor in Scaled Dot-Product Attention?" --backend real
  - Run Benchmark Suite:
      python main.py --benchmark
  - Display Official Tables:
      python main.py --tables
================================================================================
"""

import argparse
import sys
import os

# Ensure src is in sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline import RelayPipeline
from compute_tables import run_analysis


def print_banner():
    banner = r"""
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║   THE RELAY: Rate-Constrained Adaptive Research Assistant                 ║
  ║   ML Engineering Intern Assignment — Vaani Research                       ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_interactive_demo(query: str, backend: str, setting: str):
    print(f"\n[*] Initializing Pipeline [Backend: {backend.upper()} | Setting: {setting.upper()}]...")
    try:
        pipeline = RelayPipeline(retrieval_setting=setting, backend=backend)
    except Exception as e:
        print(f"[!] Error loading retriever index: {e}")
        return

    print(f"[*] Processing Query: \"{query}\"")
    print("=" * 75)
    
    res = pipeline.run(query, request_id="cli_demo_req", scenario="interactive_cli")
    
    print("\n--- [RESULT & CITATION] ---")
    print(res["answer_text"])
    print("\n--- [TELEMETRY & WATERFALL METRICS] ---")
    print(f"Status           : {res['status'].upper()}")
    print(f"Compliance Grade : {res['answer_grade'].upper()}")
    print(f"First Token TTFT : {res['first_token_ms']} ms")
    print(f"Total Latency    : {res['total_ms']} ms")
    print(f"Wait Time        : {res['waiting_ms']} ms (Queue/Backoff)")
    print(f"Working Time     : {res['working_ms']} ms (Provider Compute)")
    print(f"Retries          : {res['retries']}")
    print(f"Tokens In / Out  : {res['tokens_in']} prompt / {res['tokens_out']} completion")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="The Relay - Research Assistant Pipeline")
    parser.add_argument("--query", type=str, help="Input question to ask the pipeline")
    parser.add_argument("--backend", type=str, choices=["real", "stand-in"], default="stand-in", help="Model backend")
    parser.add_argument("--setting", type=str, choices=["setting_a", "setting_b", "setting_c"], default="setting_a", help="Retrieval setting")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--tables", action="store_true", help="Recompute and display Tables 1, 2a, 2b, 3 from results.csv")
    
    args = parser.parse_args()
    print_banner()

    if args.tables:
        print("[*] Recomputing official tables from results.csv...\n")
        run_analysis()
        return

    if args.benchmark:
        print("[*] Running benchmark suite (this will take ~1-2 minutes)...\n")
        from run_benchmarks import run_retrieval_ablation, run_load_scenarios
        run_retrieval_ablation()
        run_load_scenarios()
        print("\n[*] Recomputing tables from freshly generated results.csv...\n")
        run_analysis()
        return

    if args.query:
        run_interactive_demo(args.query, args.backend, args.setting)
    else:
        # Default test query if none provided
        sample_q = "What is the scaling factor in Scaled Dot-Product Attention and why is it used?"
        print(f"[*] No query provided. Executing default technical query:\n    \"{sample_q}\"\n")
        run_interactive_demo(sample_q, args.backend, args.setting)


if __name__ == "__main__":
    main()
