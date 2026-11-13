"""
================================================================================
THE RELAY: INTERACTIVE CLI & DEMO ENTRYPOINT
================================================================================
Vaani Research — Production AI Systems Project
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

# Try importing Rich for rich terminal rendering
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False


def print_banner():
    if RICH_AVAILABLE:
        banner_text = Text()
        banner_text.append("THE RELAY: Rate-Constrained Adaptive Research Assistant\n", style="bold cyan")
        banner_text.append("Vaani Research — Production AI Systems Engine", style="dim white")
        console.print(Panel(banner_text, border_style="bright_blue", expand=False))
    else:
        print("""
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║   THE RELAY: Rate-Constrained Adaptive Research Assistant                 ║
  ║   Vaani Research — Production AI Systems Engine                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
        """)


def run_interactive_demo(query: str, backend: str, setting: str):
    if RICH_AVAILABLE:
        console.print(f"[bold green]▶ Initializing Pipeline[/bold green] [dim]Backend: {backend.upper()} | Setting: {setting.upper()}[/dim]")
    else:
        print(f"\n[*] Initializing Pipeline [Backend: {backend.upper()} | Setting: {setting.upper()}]...")

    try:
        pipeline = RelayPipeline(retrieval_setting=setting, backend=backend)
    except Exception as e:
        print(f"[!] Error loading retriever index: {e}")
        return

    if RICH_AVAILABLE:
        console.print(f"[bold yellow]▶ Query:[/bold yellow] [white]\"{query}\"[/white]\n")
    else:
        print(f"[*] Processing Query: \"{query}\"")
        print("=" * 75)
    
    res = pipeline.run(query, request_id="cli_demo_req", scenario="interactive_cli")
    
    if RICH_AVAILABLE:
        # Render Rich Answer Panel
        ans_panel = Panel(
            res["answer_text"],
            title="[bold green]Answer & In-Line Citations[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        console.print(ans_panel)

        # Render Rich Telemetry Table
        telemetry_table = Table(title="Execution Telemetry & Latency Waterfall", border_style="cyan")
        telemetry_table.add_column("Metric", style="bold white")
        telemetry_table.add_column("Value", style="cyan")
        telemetry_table.add_column("Description", style="dim white")

        status_style = "bold green" if res["status"] == "ok" else "bold red"
        telemetry_table.add_row("Execution Status", f"[{status_style}]{res['status'].upper()}[/{status_style}]", "Upstream API lifecycle status")
        telemetry_table.add_row("Compliance Grade", f"[bold yellow]{res['answer_grade'].upper()}[/bold yellow]", "Automated citation & factuality guardrail")
        telemetry_table.add_row("Time to First Token (TTFT)", f"{res['first_token_ms']} ms", "Streaming response inception latency")
        telemetry_table.add_row("Total Wall Time", f"{res['total_ms']} ms", "End-to-end request duration")
        telemetry_table.add_row("Waiting Time (Queue/Backoff)", f"{res['waiting_ms']} ms", "Time spent waiting for rate limit buckets")
        telemetry_table.add_row("Working Time (Compute)", f"{res['working_ms']:.1f} ms", "Time spent actively generating tokens")
        telemetry_table.add_row("Retries Attempted", f"{res['retries']}", "Exponential backoff attempts")
        telemetry_table.add_row("Token Budget (In / Out)", f"{res['tokens_in']} / {res['tokens_out']}", "Prompt vs Completion token counts")

        console.print(telemetry_table)
    else:
        print("\n--- [RESULT & CITATION] ---")
        print(res["answer_text"])
        print("\n--- [TELEMETRY & WATERFALL METRICS] ---")
        print(f"Status           : {res['status'].upper()}")
        print(f"Compliance Grade : {res['answer_grade'].upper()}")
        print(f"First Token TTFT : {res['first_token_ms']} ms")
        print(f"Total Latency    : {res['total_ms']} ms")
        print(f"Wait Time        : {res['waiting_ms']} ms (Queue/Backoff)")
        print(f"Working Time     : {res['working_ms']:.1f} ms (Provider Compute)")
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
        if RICH_AVAILABLE:
            console.print("[bold cyan]▶ Recomputing official tables from results.csv...[/bold cyan]\n")
        else:
            print("[*] Recomputing official tables from results.csv...\n")
        run_analysis()
        return

    if args.benchmark:
        if RICH_AVAILABLE:
            console.print("[bold yellow]▶ Running benchmark suite across all scenarios (1-2 mins)...[/bold yellow]\n")
        else:
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
        sample_q = "What is the scaling factor in Scaled Dot-Product Attention and why is it used?"
        if RICH_AVAILABLE:
            console.print(f"[dim]No query provided. Executing default query: \"{sample_q}\"[/dim]\n")
        else:
            print(f"[*] No query provided. Executing default technical query:\n    \"{sample_q}\"\n")
        run_interactive_demo(sample_q, args.backend, args.setting)


if __name__ == "__main__":
    main()
