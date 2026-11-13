# 15 Custom Evaluation Questions across the 3 arXiv papers and out-of-domain boundaries

QUESTIONS = [
    {
        "id": "q01",
        "question": "What is the scaling factor in Scaled Dot-Product Attention and why is it used?",
        "expected_paper": "Attention",
        "expected_grade": "correct",
        "category": "attention_core"
    },
    {
        "id": "q02",
        "question": "What are the dimensions d_k, d_v and number of heads h in the Transformer base model?",
        "expected_paper": "Attention",
        "expected_grade": "correct",
        "category": "attention_params"
    },
    {
        "id": "q03",
        "question": "What BLEU score did the big Transformer model achieve on WMT 2014 English-to-German?",
        "expected_paper": "Attention",
        "expected_grade": "correct",
        "category": "attention_results"
    },
    {
        "id": "q04",
        "question": "What is the frame rate and latency of the Mimi neural audio codec in Moshi?",
        "expected_paper": "Moshi",
        "expected_grade": "correct",
        "category": "moshi_codec"
    },
    {
        "id": "q05",
        "question": "How does Moshi implement Inner Monologue during conversational spoken dialogue?",
        "expected_paper": "Moshi",
        "expected_grade": "correct",
        "category": "moshi_architecture"
    },
    {
        "id": "q06",
        "question": "What is the architecture and pretraining token volume of the Helium language model?",
        "expected_paper": "Moshi",
        "expected_grade": "correct",
        "category": "moshi_model"
    },
    {
        "id": "q07",
        "question": "What is the overall theoretical round-trip latency achieved by Moshi in real-time interaction?",
        "expected_paper": "Moshi",
        "expected_grade": "correct",
        "category": "moshi_latency"
    },
    {
        "id": "q08",
        "question": "How does the Dynamic Chunking boundary predictor determine chunk boundaries in H-Net?",
        "expected_paper": "DynamicChunking",
        "expected_grade": "correct",
        "category": "dynamic_chunking_method"
    },
    {
        "id": "q09",
        "question": "What pooling mechanism is used in Dynamic Chunking to aggregate intra-chunk representations?",
        "expected_paper": "DynamicChunking",
        "expected_grade": "correct",
        "category": "dynamic_chunking_pool"
    },
    {
        "id": "q10",
        "question": "On which long-context or language modeling benchmarks is Dynamic Chunking evaluated?",
        "expected_paper": "DynamicChunking",
        "expected_grade": "correct",
        "category": "dynamic_chunking_eval"
    },
    {
        "id": "q11",
        "question": "How does Moshi use dynamic chunking for audio frames?",
        "expected_paper": None,
        "expected_grade": "refused_correctly",
        "category": "adversarial_cross_paper"
    },
    {
        "id": "q12",
        "question": "What is the attention mechanism used in Whisper for speech recognition?",
        "expected_paper": None,
        "expected_grade": "refused_correctly",
        "category": "out_of_library_llm"
    },
    {
        "id": "q13",
        "question": "What learning rate schedule was used to train LLaMA-3?",
        "expected_paper": None,
        "expected_grade": "refused_correctly",
        "category": "out_of_library_llm"
    },
    {
        "id": "q14",
        "question": "What is the capital of France?",
        "expected_paper": None,
        "expected_grade": "refused_correctly",
        "category": "out_of_domain_trivia"
    },
    {
        "id": "q15",
        "question": "How many parameters does the GPT-4 model contain?",
        "expected_paper": None,
        "expected_grade": "refused_correctly",
        "category": "out_of_library_llm"
    }
]
