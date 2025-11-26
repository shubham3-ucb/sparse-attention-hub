---
license: apache-2.0
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- long-context
- rag
- retrieval-augmented-generation
- loft
size_categories:
- 1K<n<10K
---

# LOFT RAG - {dataset_name} ({length})

## Dataset Description

This dataset is part of the LOFT (Long-context Open Foundation Tasks) benchmark, specifically the RAG (Retrieval-Augmented Generation) task.

- **Dataset**: {dataset_name}
- **Context Length**: {length}
- **Task Type**: RAG (Retrieval-Augmented Generation)
- **Language**: English
- **Source**: LOFT Benchmark (Google DeepMind)

## Dataset Structure

### Data Fields

- `context` (string): Full prompt context including corpus documents and few-shot examples
- `question` (string): Query separator + query format + query text
- `answer_prefix` (string): Prefix for answer generation ("Final Answer: ")
- `answers` (list[string]): Ground truth answers
- `task` (string): Task identifier (e.g., "{dataset_name}_{length}")
- `max_new_tokens` (int64): Maximum tokens for generation (256)

### Data Splits

- `dev`: Development set ({dev_count} examples)
- `test`: Test set ({test_count} examples)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("f20180301/loft-rag-{dataset_name}-{length}")

# Access dev split
dev_data = dataset["dev"]
df_dev = dev_data.to_pandas()

# Access test split
test_data = dataset["test"]
df_test = test_data.to_pandas()

# Example usage
sample = dev_data[0]
context = sample["context"]
question = sample["question"]
answers = sample["answers"]
```

## Dataset Creation

This dataset was converted from LOFT's original format to HuggingFace format using exact LOFT prompt construction to ensure 100% fidelity.

- **Prompt Construction**: Uses LOFT's `PromptRegistry` and `concatenate_chunks()` for exact prompt matching
- **Few-shot Examples**: Preserved exactly as in LOFT (5 examples)
- **Corpus Documents**: Full corpus included in context (corpus-in-context approach)
- **Verification**: All prompts verified to match LOFT originals exactly

## Citation

```bibtex
@article{loft2024,
  title={LOFT: Long-context Open Foundation Tasks},
  author={Google DeepMind},
  year={2024},
  url={https://github.com/google-deepmind/loft}
}
```

## License

Apache 2.0

