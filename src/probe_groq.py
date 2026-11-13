import os
import time
import json
import statistics
import requests

API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "qwen/qwen3.8-27b"
URL = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def probe_single(prompt="Explain attention mechanism in one sentence.", stream=True):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.2,
        "stream": stream
    }
    
    t0 = time.perf_counter()
    resp = requests.post(URL, json=payload, headers=headers, stream=stream)
    
    status = resp.status_code
    rate_limits = {k: v for k, v in resp.headers.items() if "ratelimit" in k.lower()}
    
    ttft_ms = None
    total_ms = None
    chunks_count = 0
    full_text = ""
    
    if status == 200:
        if stream:
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        try:
                            c_json = json.loads(decoded[6:])
                            delta = c_json["choices"][0]["delta"].get("content", "")
                            full_text += delta
                            chunks_count += 1
                        except:
                            pass
            total_ms = (time.perf_counter() - t0) * 1000
        else:
            total_ms = (time.perf_counter() - t0) * 1000
            ttft_ms = total_ms
            full_text = resp.json()["choices"][0]["message"]["content"]
    else:
        total_ms = (time.perf_counter() - t0) * 1000
        error_body = resp.text
        return {
            "status": status,
            "error": error_body,
            "total_ms": total_ms,
            "rate_limits": rate_limits
        }

    return {
        "status": status,
        "ttft_ms": ttft_ms if ttft_ms else total_ms,
        "total_ms": total_ms,
        "chunks": chunks_count,
        "text": full_text.strip(),
        "rate_limits": rate_limits
    }

if __name__ == "__main__":
    print(f"=== Probing Groq API ({MODEL}) ===")
    results = []
    
    for i in range(8):
        print(f"Probe request {i+1}/8...")
        res = probe_single(f"Briefly describe self-attention step {i+1}")
        results.append(res)
        time.sleep(0.5)

    ttfts = [r["ttft_ms"] for r in results if r["status"] == 200 and r["ttft_ms"]]
    totals = [r["total_ms"] for r in results if r["status"] == 200]
    
    print("\n--- Summary ---")
    print(f"Successful: {len(totals)} / {len(results)}")
    if totals:
        print(f"TTFT (ms)  -> p50: {statistics.median(ttfts):.1f}, min: {min(ttfts):.1f}, max: {max(ttfts):.1f}")
        print(f"Total (ms) -> p50: {statistics.median(totals):.1f}, min: {min(totals):.1f}, max: {max(totals):.1f}")
        print("Last Rate Limit Headers:")
        for k, v in results[-1].get("rate_limits", {}).items():
            print(f"  {k}: {v}")
            
    with open(r"C:\Users\HII\.gemini\antigravity\scratch\the_relay\groq_calibration.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved probe results to groq_calibration.json")
