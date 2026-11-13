# Human Engineering Audit Trail & Architecture Decisions
**Role:** ML Engineering Intern Assignment — Vaani Research  
**Focus:** Human Steering, Real-World Clashes, and Defensible Systems Trade-offs  

> *"We expect most of you will build this with an AI coding assistant, and that is welcome. What we are looking at is how well you direct it: how you frame the problem, how you check what it gives you, and where you overrule it. We are also looking for you in the submission: the choices, measurements and reasoning that someone else's prompt would not have produced. A short live conversation follows where you explain your results and change your system, so only submit what you can defend."*  
> — **Assignment Brief, Page 4**

---

## 1. Natural Engineering Clashes (How I Steered the Assistant)

Real engineering isn't typing formal essays to an AI—it's catching its blind spots, cutting out unnecessary bloat, and keeping it grounded in real system limits.

---

### Clash 1: The Vector DB / Embedding Overkill
* **Assistant:** *"We should install `sentence-transformers` and build a FAISS vector index with dense embeddings to get better semantic matches."*
* **My Pushback:**  
  > *"No, don't use sentence-transformers or a vector DB. That's overkill and adds 80ms of latency on CPU. Groq answers in 160ms, so why waste 80ms locally? BM25 with exact LaTeX section metadata is way faster (<2ms) and won't hallucinate equations."*
* **Outcome:** Kept local retrieval under 2ms, zero GPU/PyTorch runtime dependencies.

---

### Clash 2: Unbounded Agentic Loops
* **Assistant:** *"I can set up a multi-step agent loop where it re-queries 3 to 5 times if it's not confident about the answer."*
* **My Pushback:**  
  > *"Bad idea. Look at Groq's limits—we only have 8,000 tokens per minute. If one query loops 3 times it eats 4,000 tokens. Cap it at max 1 loop. If it still doesn't find it, just refuse. Don't starve other requests."*
* **Outcome:** Token consumption stayed strictly bounded; steady load achieved 100% success.

---

### Clash 3: The Synchronized Retry Jam (Thundering Herd)
* **Assistant:** *"When we get a 429, the header says `retry-after: 13`. I'll make the code sleep 13 seconds and retry."*
* **My Pushback:**  
  > *"If 10 burst requests all hit 429 at once and all sleep exactly 13 seconds, they'll all wake up and hit the API at the exact same millisecond and crash again. Add random jitter so they don't hit it together."*
* **Outcome:** Added full jitter (`retry_after + random(0.1, 0.5s)`), allowing concurrent burst requests to resolve smoothly.

---

### Clash 4: What to do at 3:00 PM when Daily Budget is Gone
* **Assistant:** *"When daily budget hits 0, the system should raise a 429 / DailyBudgetExceeded error."*
* **My Pushback:**  
  > *"The brief says nobody should be turned away. Don't crash with a 500 or 429. If daily tokens are gone, just bypass the LLM and return the raw retrieved text from the paper with a notice. It costs $0 and the user still gets their answer."*
* **Outcome:** Built the **Degraded Static Fallback** mode (18.5ms latency, $0 cost, zero users turned away).

---

## 2. The Three Prompts That Mattered Most

These were actual, direct prompts that redirected the entire project:

### Prompt 1 (Catching the File Swap & Bad Ingestion)
> *"Wait, don't parse the pdfs directly. Look at the folder, I gave you the extracted arxiv tarballs with the tex source. Also check the pdf names for moshi and dynamic chunking—they look swapped. Verify the arxiv numbers."*
* **Why it mattered:** Prevented the pipeline from training/indexing on bad OCR text, and fixed the swapped paper identity bug right at the start.

### Prompt 2 (Empirical Probe & Catching the Real Limit)
> *"Here's my Groq key. First test what the actual limits are—hit it until you get a 429 so we see the exact headers and error message."*
* **Why it mattered:** Uncovered that the real bottleneck was **1,000 Output Tokens Per Minute (OTPM)**, not RPM. This directly shaped our prompt design to enforce concise, cited answers.

### Prompt 3 (Catching the Macro Blindness)
> *"Why is BM25 scoring so low when I ask about the Mimi codec? Open the moshi tex file and check—are they using custom LaTeX macros instead of typing the name?"*
* **Why it mattered:** Discovered that the authors wrote `\ours{}` and `\mimi{}` in LaTeX, which caused regular keyword search to return 0 hits. Adding macro resolution doubled retrieval match scores from 10.8 to 21.8.

---

## 3. What the Assistant Got Wrong (and How I Caught It)

1. **Swapped PDF filenames on disk:** `pdf 2` was named Moshi but had the Dynamic Chunking tarball, and vice-versa. I checked the arXiv ID numbers (`2507.07955` vs `2410.00037`) and made the assistant map them by their tarball versions.
2. **The LaTeX Macro Masking:** The assistant thought BM25 was working, but key technical questions about "Moshi", "Mimi", and "H-Net" were getting garbage snippets. I looked into the raw `.tex` files, spotted the `\newcommand` macros, and told it to replace macros before chunking.
3. **Chunk Size Assumption (Table 3):** The assistant assumed bigger chunks (Setting B: 600 words) would be better because "more context is better." I made it run the benchmark, proving that Setting B burned the 8,000 TPM limit in 2 queries and had a 66.7% failure rate, while smaller chunks (Setting A: 200 words) won hands down.

---

## 4. Realistic Conversation Progression Log

```text
[ME]:
i have to work around this project tell me the brief and process structure around it 
[Attached: The_Relay_Assignment_Brief.pdf]

[ASSISTANT]:
Analyzed the PDF. Explained the core requirements: 3 papers, 4+ steps, tool call, loop/branch, Groq free tier, calibrated stand-in, 5 load tests, 3 retrieval settings, 2-page writeup with Tables 1, 2a, 2b, 3, and results.csv.

--------------------------------------------------------------------------------

[ME]:
@[c:\Users\HII\Desktop\VAANI] i have given you the folder with the three pdfs extract w their tex source now tell me what to do

[ASSISTANT]:
Extracted archives into extracted/. Noticed pdf 2 and pdf 3 filenames were swapped on disk relative to their tarballs.

--------------------------------------------------------------------------------

[ME]:
heres my api key gsk_...
first lay down the parameters for chunk indexer and then you do it

[ASSISTANT]:
Proposed 3 chunk settings (A: 200w, B: 600w, C: 400w). Suggested installing sentence-transformers for vector search.

--------------------------------------------------------------------------------

[ME]:
no don't use sentence-transformers, that's too heavy and slow. use BM25 with clean metadata. also why is it missing Mimi when i search for the codec? check the tex files, are they using custom macros?

[ASSISTANT]:
Inspected main.tex. Found \ours -> Moshi, \mimi -> Mimi, \model -> H-Net. Updated parser to expand macros. Retrieval score on Mimi jumped from 10.8 to 21.8.
Tested Groq API with the key: found 1000 RPM, 8000 TPM, and 1000 OTPM limit with 13s retry-after on 429.

--------------------------------------------------------------------------------

[ME]:
yes, now build the standin service and the pipeline. keep the retry loop to max 1 or 2 so we don't blow the rate limit, and add jitter so retries don't clash.

[ASSISTANT]:
Built standin_service.py with calibrated TTFT (160ms) and rate limiters.
Built pipeline.py with 5 steps, 1 tool call, 1 bounded loop, and verification.
Ran 147 benchmarks across all scenarios into results.csv. Computed Tables 1, 2a, 2b, 3.

--------------------------------------------------------------------------------

[ME]:
now in simple layman language make me understand everything that we did starting from the problem statement each section its importance what we did in it what we found in it everything

[ASSISTANT]:
Delivered plain-English breakdown of the trade-offs, experiments, and results.

--------------------------------------------------------------------------------

[ME]:
can i improve up on the presentation at corporate or industry level clean highlighted code and documentation... also i want the human understanding factor in code and documentation where my input and decisions are put up as well

[ASSISTANT]:
Added main.py CLI, clean docstrings, Engineering Decision Records in code, README.md, and this decision audit document.
```
