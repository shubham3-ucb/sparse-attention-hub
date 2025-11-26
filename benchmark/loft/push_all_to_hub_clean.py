#!/usr/bin/env python3
"""
Push all LOFT RAG datasets to HuggingFace Hub with clean organization.

This script:
1. Combines dev+test splits into DatasetDict
2. Pushes to loft/rag-{dataset}-{length} namespace
3. Creates proper README.md for each dataset
4. Creates main index repo loft/rag
"""

import os
import json
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError


# Dataset metadata
DATASET_INFO = {
    "nq": {"name": "Natural Questions", "description": "Natural Questions RAG dataset"},
    "hotpotqa": {"name": "HotpotQA", "description": "HotpotQA RAG dataset"},
    "musique": {"name": "MuSiQue", "description": "MuSiQue RAG dataset"},
    "qampari": {"name": "Qampari", "description": "Qampari RAG dataset"},
    "quest": {"name": "Quest", "description": "Quest RAG dataset"},
}


def generate_readme(dataset_key: str, length: str, dev_count: int, test_count: int) -> str:
    """Generate README.md for a dataset."""
    base_name = dataset_key.split('_')[0]  # nq, hotpotqa, etc.
    info = DATASET_INFO.get(base_name, {"name": base_name, "description": f"{base_name} RAG dataset"})
    
    dataset_name = info["name"]
    description = info["description"]
    
    # Handle special case for qampari_1m (no dev split)
    splits_text = ""
    if dev_count > 0 and test_count > 0:
        splits_text = f"- `dev`: Development set ({dev_count} examples)\n- `test`: Test set ({test_count} examples)"
    elif test_count > 0:
        splits_text = f"- `test`: Test set ({test_count} examples)"
    
    readme = f"""---
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
- `task` (string): Task identifier (e.g., "{dataset_key}")
- `max_new_tokens` (int64): Maximum tokens for generation (256)

### Data Splits

{splits_text}

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("loft-rag-{dataset_key.replace('_', '-')}")

# Access splits
"""
    
    if dev_count > 0:
        readme += """dev_data = dataset["dev"]
df_dev = dev_data.to_pandas()
"""
    
    if test_count > 0:
        readme += """test_data = dataset["test"]
df_test = test_data.to_pandas()
"""
    
    readme += """
# Example usage
sample = dataset["dev"][0] if "dev" in dataset else dataset["test"][0]
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

## Related Datasets

All LOFT RAG datasets are available under the `loft-rag-*` namespace:
- [Main Index](https://huggingface.co/datasets/loft/rag) - Overview of all datasets

## Citation

```bibtex
@article{{loft2024,
  title={{LOFT: Long-context Open Foundation Tasks}},
  author={{Google DeepMind}},
  year={{2024}},
  url={{https://github.com/google-deepmind/loft}}
}}
```

## License

Apache 2.0
"""
    return readme


def push_dataset_with_splits(
    dev_path: str,
    test_path: str,
    repo_id: str,
    dataset_key: str,
    token: str,
    private: bool = False,
) -> bool:
    """Push a dataset with dev+test splits to HuggingFace Hub."""
    print(f"\n{'='*80}")
    print(f"Pushing: {repo_id}")
    print(f"{'='*80}")
    
    # Load datasets
    splits = {}
    dev_count = 0
    test_count = 0
    
    if dev_path and os.path.exists(dev_path):
        try:
            print(f"Loading dev split from: {dev_path}")
            splits["dev"] = load_from_disk(dev_path)
            dev_count = len(splits["dev"])
            print(f"  ✓ Dev: {dev_count} examples")
        except Exception as e:
            print(f"  ⚠️  Could not load dev split: {e}")
            dev_path = None
    
    if test_path and os.path.exists(test_path):
        try:
            print(f"Loading test split from: {test_path}")
            splits["test"] = load_from_disk(test_path)
            test_count = len(splits["test"])
            print(f"  ✓ Test: {test_count} examples")
        except Exception as e:
            print(f"  ⚠️  Could not load test split: {e}")
            test_path = None
    
    if not splits:
        print(f"  ❌ No splits found!")
        return False
    
    # Create DatasetDict
    dataset_dict = DatasetDict(splits)
    print(f"  ✓ Created DatasetDict with splits: {list(dataset_dict.keys())}")
    
    # Push to hub
    try:
        api = HfApi()
        
        # Push dataset (this creates the repo automatically)
        print(f"\n📤 Pushing dataset to: {repo_id} (public={not private})")
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=token,
        )
        print(f"  ✅ Successfully pushed dataset")
        
        # Wait a moment for repo to be fully created
        import time
        time.sleep(1)
        
        # Upload README
        print(f"📝 Uploading README.md...")
        readme_content = generate_readme(dataset_key, dataset_key.split('_')[-1], dev_count, test_count)
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add dataset card",
        )
        print(f"  ✅ README.md uploaded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_main_index_repo(
    all_datasets: list,
    org_name: str,
    token: str,
    private: bool = False,
) -> None:
    """Create main index repo with overview of all datasets."""
    repo_id = f"{org_name}/rag"  # Main index at loft/rag
    
    print(f"\n{'='*80}")
    print(f"Creating Main Index Repo: {repo_id}")
    print(f"{'='*80}")
    
    # Generate index README
    readme = """---
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
---

# LOFT RAG Datasets

This is the main index for all LOFT (Long-context Open Foundation Tasks) RAG (Retrieval-Augmented Generation) datasets.

## Overview

All datasets are part of the LOFT benchmark and have been converted to HuggingFace format with 100% prompt fidelity to the original LOFT implementation.

## Available Datasets

"""
    
    # Group by dataset
    by_dataset = {}
    for ds_key in sorted(all_datasets):
        parts = ds_key.split('_')
        if len(parts) >= 2:
            base = parts[0]
            length = parts[1]
            if base not in by_dataset:
                by_dataset[base] = []
            by_dataset[base].append((length, ds_key))
    
    # Generate links
    for base_name in sorted(by_dataset.keys()):
        info = DATASET_INFO.get(base_name, {"name": base_name})
        readme += f"\n### {info['name']}\n\n"
        for length, ds_key in sorted(by_dataset[base_name]):
            repo_name = ds_key.replace('_', '-')
            readme += f"- **{length}**: [`loft-rag-{repo_name}`](https://huggingface.co/datasets/loft-rag-{repo_name})\n"
        readme += "\n"
    
    readme += """## Usage

```python
from datasets import load_dataset

# Load any dataset
dataset = load_dataset("loft-rag-nq-32k")

# Access splits
dev_data = dataset["dev"]
test_data = dataset["test"]

# Convert to pandas
df_dev = dev_data.to_pandas()
df_test = test_data.to_pandas()
```

## Dataset Structure

All datasets contain:
- `context`: Full prompt context with corpus documents and few-shot examples
- `question`: Query separator + query format + query text
- `answer_prefix`: Prefix for answer generation
- `answers`: Ground truth answers (list)
- `task`: Task identifier
- `max_new_tokens`: Maximum tokens for generation (256)

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
"""
    
    # Create repo and upload README
    api = HfApi()
    try:
        # Create repo (or update if exists)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True,
        )
        
        # Upload README
        api.upload_file(
            path_or_fileobj=readme.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        
        print(f"  ✅ Main index repo created: {repo_id}")
        print(f"  🔗 https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        print(f"  ❌ Error creating index repo: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Push all LOFT RAG datasets to HuggingFace Hub with clean organization"
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="/data/sparse_attention_hub/loft_rag_datasets",
        help="Directory containing converted datasets",
    )
    parser.add_argument(
        "--org_name",
        type=str,
        default="f20180301",
        help="Organization/namespace name (default: f20180301)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN"),
        help="HuggingFace token (or set HF_TOKEN/HF_HUB_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repos private (default: public)",
    )
    
    args = parser.parse_args()
    
    # Login
    login(token=args.token)
    print(f"✓ Logged in to HuggingFace")
    
    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists():
        print(f"❌ Datasets directory not found: {datasets_dir}")
        return
    
    # Find all datasets and group by dataset+length
    # SKIP topicqa (messy, as requested)
    dataset_groups = {}
    for item in datasets_dir.iterdir():
        if item.is_dir():
            name = item.name
            # Skip topicqa
            if 'topicqa' in name.lower():
                print(f"⚠️  Skipping topicqa dataset: {name}")
                continue
            
            parts = name.split('_')
            if len(parts) >= 3:
                split = parts[-1]  # dev or test
                length = parts[-2]  # 32k, 128k, 1m
                base = '_'.join(parts[:-2])  # nq, hotpotqa, etc.
                
                key = f"{base}_{length}"
                if key not in dataset_groups:
                    dataset_groups[key] = {}
                dataset_groups[key][split] = str(item)
    
    print(f"\n{'='*80}")
    print(f"Found {len(dataset_groups)} dataset groups")
    print(f"{'='*80}")
    
    # Push each dataset group
    pushed_datasets = []
    success_count = 0
    
    for dataset_key in sorted(dataset_groups.keys()):
        splits = dataset_groups[dataset_key]
        dev_path = splits.get('dev')
        test_path = splits.get('test')
        
        if not dev_path and not test_path:
            print(f"⚠️  Skipping {dataset_key}: No splits found")
            continue
        
        # Create repo name (use org_name/loft-rag-... structure)
        repo_name = dataset_key.replace('_', '-')
        repo_id = f"{args.org_name}/loft-rag-{repo_name}"
        
        success = push_dataset_with_splits(
            dev_path=dev_path,
            test_path=test_path,
            repo_id=repo_id,
            dataset_key=dataset_key,
            token=args.token,
            private=args.private,
        )
        
        if success:
            success_count += 1
            pushed_datasets.append(dataset_key)
    
    # Create main index repo
    if pushed_datasets:
        create_main_index_repo(
            all_datasets=pushed_datasets,
            org_name=args.org_name,
            token=args.token,
            private=args.private,
        )
    
    print(f"\n{'='*80}")
    print(f"✅ Push Complete!")
    print(f"{'='*80}")
    print(f"  Pushed: {success_count}/{len(dataset_groups)} datasets")
    print(f"  Organization: {args.org_name}")
    print(f"  Main index: {args.org_name}/rag")
    print(f"\n  🔗 Browse: https://huggingface.co/datasets/{args.org_name}/rag")


if __name__ == "__main__":
    main()

