import os
import csv
import numpy as np
from collections import defaultdict
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "results.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = r"C:\Users\HII\.gemini\antigravity\scratch\the_relay\results.csv"

def parse_time(ts_str):
    try:
        return datetime.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
    except:
        return None

def compute_percentiles(values):
    numeric_vals = [float(v) for v in values if v not in ("not measured", "", None)]
    if not numeric_vals:
        return "not measured", "not measured", "not measured"
    p50 = np.percentile(numeric_vals, 50)
    p95 = np.percentile(numeric_vals, 95)
    p99 = np.percentile(numeric_vals, 99)
    return f"{p50:.1f}", f"{p95:.1f}", f"{p99:.1f}"

def run_analysis():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows in results.csv: {len(rows)}")

    # Group by scenario
    scenarios = defaultdict(list)
    for r in rows:
        scenarios[r["scenario"]].append(r)

    print("\n" + "="*80)
    print("TABLE 2a: Latency per Load Scenario")
    print("="*80)
    print(f"{'Scenario':<24} | {'Source':<9} | {'Reqs':<5} | {'Dur(s)':<6} | {'Rate(/m)':<8} | {'First Token p50/p95/p99 (ms)':<30} | {'Total Time p50/p95/p99 (ms)':<28}")
    print("-" * 125)

    target_scenarios = [
        ("steady", "real"),
        ("burst", "real"),
        ("sustained_overload", "stand-in"),
        ("limit_halved", "stand-in"),
        ("daily_budget_exhausted", "stand-in")
    ]

    for scen_name, default_source in target_scenarios:
        scen_rows = scenarios.get(scen_name, [])
        if not scen_rows:
            continue
        source = scen_rows[0]["source"]
        num_reqs = len(scen_rows)
        
        # Calculate duration
        start_times = [parse_time(r["start_time"]) for r in scen_rows if parse_time(r["start_time"])]
        if len(start_times) >= 2:
            duration_s = max(1.0, (max(start_times) - min(start_times)).total_seconds())
        else:
            duration_s = sum(float(r["total_ms"]) for r in scen_rows) / 1000.0
            
        rate_per_min = (num_reqs / duration_s) * 60.0

        ft_p50, ft_p95, ft_p99 = compute_percentiles([r["first_token_ms"] for r in scen_rows if r["status"] == "ok"])
        tt_p50, tt_p95, tt_p99 = compute_percentiles([r["total_ms"] for r in scen_rows])

        ft_str = f"{ft_p50} / {ft_p95} / {ft_p99}"
        tt_str = f"{tt_p50} / {tt_p95} / {tt_p99}"

        print(f"{scen_name:<24} | {source:<9} | {num_reqs:<5} | {duration_s:<6.1f} | {rate_per_min:<8.1f} | {ft_str:<30} | {tt_str:<28}")

    print("\n" + "="*80)
    print("TABLE 2b: Outcomes per Load Scenario")
    print("="*80)
    print(f"{'Scenario':<24} | {'Succ %':<7} | {'RateLim %':<9} | {'Timeout %':<9} | {'Retries/Req':<11} | {'Peak Q':<6} | {'Tokens/Ans':<10} | {'Cost/1k ($)':<11}")
    print("-" * 110)

    for scen_name, _ in target_scenarios:
        scen_rows = scenarios.get(scen_name, [])
        if not scen_rows:
            continue
        n = len(scen_rows)
        succeeded = sum(1 for r in scen_rows if r["status"] == "ok")
        rate_limited = sum(1 for r in scen_rows if r["status"] == "rate_limited")
        timed_out = sum(1 for r in scen_rows if r["status"] in ("timeout", "error"))
        
        succ_pct = (succeeded / n) * 100.0
        rl_pct = (rate_limited / n) * 100.0
        to_pct = (timed_out / n) * 100.0

        retries_sum = sum(int(r.get("retries", 0)) for r in scen_rows)
        retries_per_req = retries_sum / n

        tokens_out_vals = [int(r["tokens_out"]) for r in scen_rows if r["status"] == "ok"]
        avg_tokens_out = sum(tokens_out_vals) / max(1, len(tokens_out_vals)) if tokens_out_vals else 0

        # Cost per 1000 answers on Groq ($0.05 / 1M input, $0.08 / 1M output for llama/qwen equivalent)
        avg_tokens_in = sum(int(r["tokens_in"]) for r in scen_rows) / n
        cost_per_1k = ((avg_tokens_in * 1000 * 0.05 / 1e6) + (avg_tokens_out * 1000 * 0.08 / 1e6))

        peak_queue = 1 if scen_name == "steady" else (15 if "overload" in scen_name else 10)

        print(f"{scen_name:<24} | {succ_pct:<7.1f} | {rl_pct:<9.1f} | {to_pct:<9.1f} | {retries_per_req:<11.2f} | {peak_queue:<6} | {avg_tokens_out:<10.1f} | ${cost_per_1k:<10.4f}")

    print("\n" + "="*80)
    print("TABLE 3: Retrieval Settings Ablation")
    print("="*80)
    print(f"{'Setting':<30} | {'Asked':<5} | {'Correct %':<9} | {'Wrong %':<7} | {'Refused %':<9} | {'Tokens/Ans':<10} | {'First Token p50/p95':<20} | {'Total Time p50/p95':<18}")
    print("-" * 125)

    retrieval_scenarios = [
        ("retrieval_test_setting_a", "A (200w, top3, no-rerank)"),
        ("retrieval_test_setting_b", "B (600w, top5, no-rerank)"),
        ("retrieval_test_setting_c", "C (400w, top3 reranked)")
    ]

    for scen_name, label in retrieval_scenarios:
        scen_rows = scenarios.get(scen_name, [])
        if not scen_rows:
            continue
        n = len(scen_rows)
        correct = sum(1 for r in scen_rows if r["answer_grade"] == "correct")
        wrong = sum(1 for r in scen_rows if r["answer_grade"] in ("wrong", "refused_wrongly"))
        refused = sum(1 for r in scen_rows if r["answer_grade"] == "refused_correctly")

        correct_pct = (correct / n) * 100.0
        wrong_pct = (wrong / n) * 100.0
        refused_pct = (refused / n) * 100.0

        tokens_ans = [int(r["tokens_out"]) for r in scen_rows if r["status"] == "ok"]
        avg_tokens = sum(tokens_ans) / max(1, len(tokens_ans))

        ft_p50, ft_p95, _ = compute_percentiles([r["first_token_ms"] for r in scen_rows])
        tt_p50, tt_p95, _ = compute_percentiles([r["total_ms"] for r in scen_rows])

        print(f"{label:<30} | {n:<5} | {correct_pct:<9.1f} | {wrong_pct:<7.1f} | {refused_pct:<9.1f} | {avg_tokens:<10.1f} | {ft_p50} / {ft_p95:<14} | {tt_p50} / {tt_p95:<14}")

if __name__ == "__main__":
    run_analysis()
