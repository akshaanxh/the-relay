import time
import random
import math
from typing import Dict, Any, Generator, Tuple

class StandinModelService:
    """
    Calibrated stand-in model service simulating Groq's Free Tier behavior:
    - Log-normal TTFT (median ~160ms, tail > 800ms)
    - Concurrency queue slowdown (increases TTFT under load)
    - Streaming generation at ~450-550 tokens/sec
    - Sliding-window rate limiters:
        * RPM (Requests per minute, default 1,000)
        * TPM (Total tokens per minute, default 8,000)
        * OTPM (Output tokens per minute, default 1,000)
        * Daily token quota (can be set to 0 to simulate 3pm exhaustion)
    - Emulates exact HTTP 429 payloads with retry-after
    - Simulates transient socket errors / 503s
    """
    def __init__(
        self,
        rpm_limit: int = 1000,
        tpm_limit: int = 8000,
        otpm_limit: int = 1000,
        daily_token_budget: int = 500000,
        failure_rate: float = 0.02
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.otpm_limit = otpm_limit
        self.daily_token_budget = daily_token_budget
        self.daily_tokens_used = 0
        self.failure_rate = failure_rate
        
        # Concurrency tracking
        self.active_requests = 0
        
        # Sliding window timestamps: list of (timestamp, in_tokens, out_tokens)
        self.history = []

    def _prune_history(self, current_time: float):
        one_minute_ago = current_time - 60.0
        self.history = [h for h in self.history if h[0] > one_minute_ago]

    def check_limits(self, estimated_in_tokens: int, estimated_out_tokens: int = 200) -> Tuple[bool, Dict[str, Any]]:
        now = time.time()
        self._prune_history(now)
        
        # Check daily quota
        if self.daily_tokens_used + estimated_in_tokens + estimated_out_tokens > self.daily_token_budget:
            return False, {
                "status_code": 429,
                "retry_after": 86400,
                "error": {
                    "message": "Daily request or token quota exhausted. Budget reset at midnight UTC.",
                    "type": "insufficient_quota",
                    "code": "daily_quota_exceeded"
                }
            }

        # Check minute rate limits
        req_count = len(self.history)
        tokens_in_sum = sum(h[1] for h in self.history)
        tokens_out_sum = sum(h[2] for h in self.history)
        total_tokens_sum = tokens_in_sum + tokens_out_sum

        if req_count >= self.rpm_limit:
            oldest = self.history[0][0]
            retry_after = max(1, int(60.0 - (now - oldest)))
            return False, {
                "status_code": 429,
                "retry_after": retry_after,
                "error": {
                    "message": f"Rate limit reached for model on requests per minute (RPM): Limit {self.rpm_limit}, Used {req_count}.",
                    "type": "tokens",
                    "code": "rate_limit_exceeded"
                }
            }

        if (total_tokens_sum + estimated_in_tokens + estimated_out_tokens) > self.tpm_limit:
            oldest = self.history[0][0]
            retry_after = max(1, int(60.0 - (now - oldest)))
            return False, {
                "status_code": 429,
                "retry_after": retry_after,
                "error": {
                    "message": f"Rate limit reached for model on total tokens per minute (TPM): Limit {self.tpm_limit}, Used {total_tokens_sum}.",
                    "type": "tokens",
                    "code": "rate_limit_exceeded"
                }
            }

        if (tokens_out_sum + estimated_out_tokens) > self.otpm_limit:
            oldest = self.history[0][0]
            retry_after = max(1, int(60.0 - (now - oldest)))
            return False, {
                "status_code": 429,
                "retry_after": retry_after,
                "error": {
                    "message": f"Rate limit reached for model on output tokens per minute (OTPM): Limit {self.otpm_limit}, Used {tokens_out_sum}, Requested {estimated_out_tokens}.",
                    "type": "tokens",
                    "code": "rate_limit_exceeded"
                }
            }

        return True, {}

    def stream_completion(
        self,
        prompt: str,
        context_chunks: list,
        is_refusal: bool = False,
        refusal_reason: str = "Out of domain / not found in papers."
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Simulates generation piece-by-piece with realistic latencies.
        """
        in_tokens = int(len(prompt.split()) * 1.3)
        self.active_requests += 1

        try:
            # 1. Random transient failure simulation
            if random.random() < self.failure_rate:
                yield {
                    "type": "error",
                    "status_code": 503,
                    "error": {"message": "Service temporarily unavailable / upstream timeout.", "code": "service_unavailable"}
                }
                return

            # 2. Check limits
            estimated_out = 50 if is_refusal else 150
            allowed, limit_err = self.check_limits(in_tokens, estimated_out)
            if not allowed:
                yield {
                    "type": "rate_limited",
                    "status_code": limit_err["status_code"],
                    "retry_after": limit_err.get("retry_after", 10),
                    "error": limit_err["error"]
                }
                return

            # 3. Simulate TTFT with concurrency queuing delay
            # Groq base TTFT: lognormal with median ~160ms (sigma=0.35)
            base_ttft = random.lognormvariate(math.log(0.160), 0.35)
            # Add concurrency penalty (queuing under heavy load)
            queue_penalty = max(0.0, (self.active_requests - 2) * 0.045)
            total_ttft = max(0.08, base_ttft + queue_penalty)

            time.sleep(total_ttft)

            # 4. Generate answer tokens
            if is_refusal:
                reply_text = (
                    "I cannot answer this question based on the provided papers. "
                    "The library contains only 'Attention Is All You Need', 'Moshi', "
                    "and 'Dynamic Chunking', and this question is either not covered or outside the scope."
                )
            else:
                top_chunk = context_chunks[0] if context_chunks else {}
                paper = top_chunk.get("paper_short", "Unknown Paper")
                sec = top_chunk.get("source_file", "Sections")
                snippet = top_chunk.get("text", "")[:120].replace("\n", " ")
                reply_text = (
                    f"Based on [{paper}, Section: {sec}], "
                    f"the papers state that {snippet.strip()}... "
                    f"Specifically, this mechanism directly addresses the operational requirements cited."
                )

            words = reply_text.split()
            out_tokens = int(len(words) * 1.3)
            
            # Record usage in sliding window
            self.history.append((time.time(), in_tokens, out_tokens))
            self.daily_tokens_used += (in_tokens + out_tokens)

            # Yield first token event
            first_chunk = True
            
            # Stream in small groups of 3-4 words (simulating chunks)
            chunk_size = 4
            for i in range(0, len(words), chunk_size):
                chunk_slice = " ".join(words[i:i + chunk_size]) + " "
                # Token inter-arrival latency: ~500 tokens/sec -> ~2ms/token -> ~8ms per 4 words
                time.sleep(0.008 + random.uniform(0.001, 0.004))
                
                yield {
                    "type": "delta",
                    "content": chunk_slice,
                    "is_first": first_chunk,
                    "ttft_s": total_ttft if first_chunk else None
                }
                first_chunk = False

            yield {
                "type": "done",
                "tokens_in": in_tokens,
                "tokens_out": out_tokens,
                "total_text": reply_text
            }

        finally:
            self.active_requests = max(0, self.active_requests - 1)
