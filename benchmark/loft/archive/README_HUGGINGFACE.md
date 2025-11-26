# LOFT RAG Datasets on HuggingFace Hub

## Overview

The LOFT RAG datasets have been converted to HuggingFace format and can be pushed to HuggingFace Hub for easy access.

## Pushing Datasets

### Prerequisites

1. **HuggingFace Account**: Create account at https://huggingface.co
2. **Login**: 
   ```bash
   huggingface-cli login
   ```
   Or set `HF_TOKEN` environment variable

### Push All Datasets

```bash
python3 benchmark/loft/push_to_huggingface.py \
  --datasets_dir /data/sparse_attention_hub/loft_rag_datasets \
  --hub_username YOUR_USERNAME \
  --private  # or --public
```

### Push Single Dataset

```python
from datasets import load_from_disk
from huggingface_hub import login

login()  # or use token

dataset = load_from_disk("/data/sparse_attention_hub/loft_rag_datasets/nq_32k_dev")
dataset.push_to_hub("YOUR_USERNAME/loft-rag-nq-32k-dev", private=True)
```

## Using Datasets from Hub

### Load Single Dataset

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("YOUR_USERNAME/loft-rag-nq-32k-dev")
df = dataset['train'].to_pandas()  # or 'test' if split='test'

# Use in benchmark
from benchmark.loft import LoftRAG
benchmark = LoftRAG(subsets_to_run=["nq_32k"])
results = benchmark.run_benchmark(adapter, ...)
```

### Load Multiple Datasets

```python
from datasets import load_dataset

datasets = [
    "YOUR_USERNAME/loft-rag-nq-32k-dev",
    "YOUR_USERNAME/loft-rag-nq-32k-test",
    "YOUR_USERNAME/loft-rag-hotpotqa-32k-dev",
]

dfs = []
for repo_id in datasets:
    ds = load_dataset(repo_id)
    df = ds['train'].to_pandas()
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
```

## Dataset Naming Convention

Format: `{username}/loft-rag-{dataset}-{length}-{split}`

Examples:
- `darvog/loft-rag-nq-32k-dev`
- `darvog/loft-rag-nq-32k-test`
- `darvog/loft-rag-hotpotqa-128k-dev`

## Dataset Structure

Each dataset contains:
- `context`: Full prompt context (corpus + few-shot examples)
- `question`: Query separator + query format + query text
- `answer_prefix`: "Final Answer: "
- `answers`: List of ground truth answers
- `task`: Task identifier (e.g., "nq_32k")
- `max_new_tokens`: Maximum tokens for generation (256)

## Benefits

1. **Easy Access**: No need to manage local files
2. **Version Control**: Datasets are versioned on Hub
3. **Sharing**: Easy to share with collaborators
4. **Reproducibility**: Exact dataset versions preserved
5. **No Local Storage**: Datasets downloaded on-demand

## Privacy

- **Private repos**: Only you (and collaborators) can access
- **Public repos**: Anyone can access (good for sharing)

Recommendation: Start with private, make public when ready.

## Size Considerations

- 32k datasets: ~1-20 MB each
- 128k datasets: ~4-45 MB each  
- 1m datasets: ~30-350 MB each

Total for all RAG datasets: ~2-3 GB (manageable)

## Integration with Benchmark

The benchmark can be updated to load from HuggingFace Hub:

```python
@register_benchmark("loft_rag")
class LoftRAG(Benchmark):
    benchmark_name: str = "loft_rag"
    huggingface_dataset_id: str = "YOUR_USERNAME/loft-rag"  # Base repo
    
    def _load_datasets(self):
        # Load each subset as config
        dfs = []
        for subset in self.subsets_to_run:
            # Parse subset (e.g., "nq_32k" -> "nq", "32k")
            dataset, length = subset.rsplit("_", 1)
            repo_id = f"{self.huggingface_dataset_id}-{dataset}-{length}-test"
            ds = load_dataset(repo_id)
            df = ds['train'].to_pandas()
            df['task'] = subset
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
```

