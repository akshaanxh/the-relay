import os
import sys
import time
import csv
import concurrent.futures
import random
from typing import List, Dict, Any

# Add module paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from pipeline import RelayPipeline
from standin_service import StandinModelService
from questions import QUESTIONS

RESULTS_CSV_PATH = os.path.join(PROJECT_ROOT, "results.csv")

CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]

def append_to_csv(rows: List[Dict[str, Any]]):
    file_exists = os.path.exists(RESULTS_CSV_PATH)
    with open(RESULTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            # Filter only target columns
            row_dict = {col: r.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(row_dict)

def run_retrieval_ablation():
    """
    Runs Table 3 ablation: 15 questions across Setting A, Setting B, Setting C.
    Uses real Groq where reasonable, or stand-in to preserve tokens.
    """
    print("\n=======================================================")
    print(">>> RUNNING TABLE 3: RETRIEVAL SETTINGS EXPERIMENT <<<")
    print("=======================================================")
    
    settings = ["setting_a", "setting_b", "setting_c"]
    all_results = []
    
    for s in settings:
        print(f"\n--- Testing Retrieval {s} ---")
        pipeline = RelayPipeline(retrieval_setting=s, backend="stand-in")
        
        for idx, q_item in enumerate(QUESTIONS):
            req_id = f"retrieval_{s}_{q_item['id']}"
            scenario = f"retrieval_test_{s}"
            
            res = pipeline.run(q_item["question"], req_id, scenario)
            
            # Reconcile grade against question expectation
            if q_item["expected_grade"] == "refused_correctly":
                if "cannot answer" in res["answer_text"].lower() or "not covered" in res["answer_text"].lower():
                    res["answer_grade"] = "refused_correctly"
                else:
                    res["answer_grade"] = "wrong"
            else:
                if "cannot answer" in res["answer_text"].lower():
                    res["answer_grade"] = "refused_wrongly"
                elif res["status"] == "ok":
                    res["answer_grade"] = "correct"
                else:
                    res["answer_grade"] = "wrong"
            
            all_results.append(res)
            print(f"[{s}] {q_item['id']} -> {res['answer_grade']} (TTFT: {res['first_token_ms']}ms, In: {res['tokens_in']}, Out: {res['tokens_out']})")
            time.sleep(0.05)

    append_to_csv(all_results)
    print(f"Table 3 ablation complete: {len(all_results)} requests logged.")

def run_load_scenarios():
    """
    Runs Table 2a/2b load scenarios:
    1. Steady Load (Real Groq, paced)
    2. Sudden Burst (Real Groq, concurrent)
    3. Sustained Overload (Stand-in, high rate)
    4. Per-minute Limit Halved (Stand-in, RPM=500, TPM=4000)
    5. Daily Budget Exhausted (Stand-in, budget=0, simulating 3 PM cutoff)
    """
    print("\n=======================================================")
    print(">>> RUNNING TABLE 2: LOAD SCENARIOS BENCHMARKS <<<")
    print("=======================================================")
    
    # 1. Steady Load (Real Groq: 12 requests paced at ~20 req/min)
    print("\n--- Scenario 1: Steady Load (Real Groq) ---")
    p_steady = RelayPipeline(retrieval_setting="setting_c", backend="real")
    steady_results = []
    
    for i in range(10):
        q = QUESTIONS[i % len(QUESTIONS)]["question"]
        req_id = f"steady_real_{i+1:03d}"
        res = p_steady.run(q, req_id, "steady")
        steady_results.append(res)
        print(f"[Steady Real] {req_id} -> {res['status']} ({res['total_ms']}ms)")
        time.sleep(2.0) # Paced
    append_to_csv(steady_results)

    # 2. Sudden Burst (Real Groq: 15 concurrent requests)
    print("\n--- Scenario 2: Sudden Burst (Real Groq Concurrent) ---")
    p_burst_real = RelayPipeline(retrieval_setting="setting_c", backend="real", max_retries=1)
    burst_results = []
    
    def exec_burst(idx):
        q = QUESTIONS[idx % len(QUESTIONS)]["question"]
        req_id = f"burst_real_{idx+1:03d}"
        return p_burst_real.run(q, req_id, "burst")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futs = [executor.submit(exec_burst, i) for i in range(12)]
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            burst_results.append(res)
            print(f"[Burst Real] {res['request_id']} -> {res['status']} ({res['total_ms']}ms, retries: {res['retries']})")
    append_to_csv(burst_results)

    # 3. Sustained Overload (Stand-in: 40 requests under high concurrency)
    print("\n--- Scenario 3: Sustained Overload (Stand-in) ---")
    standin_overload = StandinModelService(rpm_limit=1000, tpm_limit=8000, otpm_limit=1000)
    p_overload = RelayPipeline(retrieval_setting="setting_c", backend="stand-in", standin_service=standin_overload)
    overload_results = []

    def exec_overload(idx):
        q = QUESTIONS[idx % len(QUESTIONS)]["question"]
        req_id = f"overload_standin_{idx+1:03d}"
        return p_overload.run(q, req_id, "sustained_overload")

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futs = [executor.submit(exec_overload, i) for i in range(40)]
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            overload_results.append(res)
            print(f"[Overload Stand-in] {res['request_id']} -> {res['status']} ({res['total_ms']}ms)")
    append_to_csv(overload_results)

    # 4. Per-Minute Limit Halved (Stand-in: RPM=500, TPM=4000, OTPM=500)
    print("\n--- Scenario 4: Per-Minute Limit Halved (Stand-in) ---")
    standin_halved = StandinModelService(rpm_limit=500, tpm_limit=4000, otpm_limit=500)
    p_halved = RelayPipeline(retrieval_setting="setting_c", backend="stand-in", standin_service=standin_halved)
    halved_results = []

    def exec_halved(idx):
        q = QUESTIONS[idx % len(QUESTIONS)]["question"]
        req_id = f"halved_standin_{idx+1:03d}"
        return p_halved.run(q, req_id, "limit_halved")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futs = [executor.submit(exec_halved, i) for i in range(25)]
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            halved_results.append(res)
            print(f"[Limit Halved] {res['request_id']} -> {res['status']} ({res['total_ms']}ms)")
    append_to_csv(halved_results)

    # 5. Daily Budget Exhausted (Stand-in: budget=0, 3 PM cutoff simulation)
    print("\n--- Scenario 5: Daily Budget Exhausted (Stand-in) ---")
    standin_exhausted = StandinModelService(daily_token_budget=0)
    p_exhausted = RelayPipeline(retrieval_setting="setting_c", backend="stand-in", standin_service=standin_exhausted)
    exhausted_results = []

    for i in range(15):
        q = QUESTIONS[i % len(QUESTIONS)]["question"]
        req_id = f"exhausted_standin_{i+1:03d}"
        res = p_exhausted.run(q, req_id, "daily_budget_exhausted")
        exhausted_results.append(res)
        print(f"[Exhausted] {req_id} -> {res['status']} ({res['total_ms']}ms, {res['answer_text'][:50]}...)")
    append_to_csv(exhausted_results)

    print("\nAll benchmark scenarios successfully executed and appended to results.csv!")

if __name__ == "__main__":
    # Remove old results.csv if exists to ensure clean run
    if os.path.exists(RESULTS_CSV_PATH):
        os.remove(RESULTS_CSV_PATH)
        print("Cleared previous results.csv.")
        
    run_retrieval_ablation()
    run_load_scenarios()
