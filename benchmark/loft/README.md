# LOFT RAG Benchmark

LOFT RAG benchmark integration for sparse-attention-hub. Evaluates long-context Retrieval-Augmented Generation tasks with sparse attention.

## Usage

```python
from sparse_attention_hub.benchmark import create_benchmark_instance

benchmark = create_benchmark_instance("loft_rag", subsets=["hotpotqa_32k"])
metrics = benchmark.run_benchmark(adapter=model_adapter, result_dir="./results")
```

## Available Datasets

- `nq_32k`, `nq_128k`, `nq_1m` (Natural Questions)
- `hotpotqa_32k`, `hotpotqa_128k`, `hotpotqa_1m` (HotpotQA)
- `musique_32k`, `musique_128k`, `musique_1m` (MuSiQue)
- `qampari_32k`, `qampari_128k`, `qampari_1m` (Qampari)
- `quest_32k`, `quest_128k`, `quest_1m` (QuestEval)

## Metrics

- **EM** (Exact Match): Primary metric for single-value tasks
- **F1**: Token-level F1 score
- **Subspan EM**: Subspan exact match for multi-value tasks
- **Coverage**: Coverage metric for multi-value RAG

Datasets are hosted on HuggingFace Hub at `f20180301/loft-rag-{dataset}-{length}`.

