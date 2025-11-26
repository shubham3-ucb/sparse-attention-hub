#!/usr/bin/env python3
"""
Push LOFT RAG datasets to HuggingFace Hub.

This script pushes all converted LOFT RAG datasets to HuggingFace Hub,
making them easily accessible via load_dataset().
"""

import os
import argparse
from pathlib import Path
from datasets import load_from_disk
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError


def push_dataset_to_hub(
    local_path: str,
    repo_id: str,
    split_name: str = None,
    private: bool = True,
    token: str = None,
) -> None:
    """
    Push a local dataset to HuggingFace Hub.
    
    Args:
        local_path: Path to local dataset directory
        repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
        split_name: Split name (e.g., "dev", "test", "train"). If None, extracts from dataset name.
        private: Whether to make the repo private
        token: HuggingFace token (if not logged in)
    """
    print(f"\n{'='*80}")
    print(f"Pushing: {repo_id}")
    print(f"{'='*80}")
    
    # Load dataset
    print(f"Loading dataset from: {local_path}")
    dataset = load_from_disk(local_path)
    print(f"  ✓ Loaded {len(dataset)} examples")
    
    # Determine split name if not provided
    if split_name is None:
        # Extract split from dataset name (e.g., "nq_32k_dev" -> "dev")
        dataset_name = Path(local_path).name
        if "_dev" in dataset_name:
            split_name = "dev"
        elif "_test" in dataset_name:
            split_name = "test"
        else:
            split_name = "train"  # Default fallback
    
    print(f"  Using split: {split_name}")
    
    # Push to hub with split name
    try:
        print(f"Pushing to: {repo_id} (split: {split_name})")
        dataset.push_to_hub(
            repo_id=repo_id,
            split=split_name,
            private=private,
            token=token,
        )
        print(f"  ✅ Successfully pushed to {repo_id}")
        print(f"  📦 Access with: load_dataset('{repo_id}')['{split_name}']")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower():
            print(f"  ⚠️  Repo already exists, updating...")
            try:
                dataset.push_to_hub(
                    repo_id=repo_id,
                    split=split_name,
                    private=private,
                    token=token,
                    revision="main",
                )
                print(f"  ✅ Successfully updated {repo_id}")
            except Exception as e2:
                print(f"  ❌ Failed to update: {e2}")
        else:
            print(f"  ❌ Failed to push: {e}")
            raise
    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise


def push_all_datasets(
    datasets_dir: str,
    hub_username: str,
    base_repo_name: str = "loft-rag",
    private: bool = True,
    token: str = None,
) -> None:
    """
    Push all datasets from a directory to HuggingFace Hub.
    
    Args:
        datasets_dir: Directory containing converted datasets
        hub_username: Your HuggingFace username
        base_repo_name: Base name for repos (will append dataset info)
        private: Whether repos should be private
        token: HuggingFace token (if not logged in)
    """
    datasets_dir = Path(datasets_dir)
    
    if not datasets_dir.exists():
        print(f"❌ Datasets directory not found: {datasets_dir}")
        return
    
    # Find all dataset directories
    dataset_dirs = [
        d for d in datasets_dir.iterdir()
        if d.is_dir() and not d.name.endswith('.json')
    ]
    
    if not dataset_dirs:
        print(f"❌ No datasets found in {datasets_dir}")
        return
    
    print(f"🚀 Pushing {len(dataset_dirs)} datasets to HuggingFace Hub")
    print(f"   Username: {hub_username}")
    print(f"   Base repo: {base_repo_name}")
    print(f"   Private: {private}")
    
    # Option 1: Push each as separate repo
    print(f"\n📦 Strategy: Separate repos (one per dataset)")
    print(f"   Format: {hub_username}/{base_repo_name}-{{dataset}}-{{length}}-{{split}}")
    
    success_count = 0
    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name  # e.g., "nq_32k_dev"
        
        # Create repo ID
        repo_id = f"{hub_username}/{base_repo_name}-{dataset_name}"
        
        try:
            push_dataset_to_hub(
                local_path=str(dataset_dir),
                repo_id=repo_id,
                private=private,
                token=token,
            )
            success_count += 1
        except Exception as e:
            print(f"  ❌ Failed to push {dataset_name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Push complete! {success_count}/{len(dataset_dirs)} datasets pushed")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Push LOFT RAG datasets to HuggingFace Hub"
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="/data/sparse_attention_hub/loft_rag_datasets",
        help="Directory containing converted datasets",
    )
    parser.add_argument(
        "--hub_username",
        type=str,
        required=True,
        help="Your HuggingFace username",
    )
    parser.add_argument(
        "--base_repo_name",
        type=str,
        default="loft-rag",
        help="Base name for repos",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make repos private (default: True)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repos public",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (if not logged in)",
    )
    
    args = parser.parse_args()
    
    # Handle public/private
    private = not args.public if args.public else args.private
    
    # Check login
    try:
        from huggingface_hub import whoami
        user = whoami(token=args.token)
        print(f"✓ Logged in as: {user.get('name', 'unknown')}")
        if args.hub_username != user.get('name'):
            print(f"⚠️  Warning: Username mismatch. Using {args.hub_username}")
    except Exception:
        if not args.token:
            print("⚠️  Not logged in. Please:")
            print("   1. Run: huggingface-cli login")
            print("   2. Or provide --token")
            return
        print(f"✓ Using provided token")
    
    # Push all datasets
    push_all_datasets(
        datasets_dir=args.datasets_dir,
        hub_username=args.hub_username,
        base_repo_name=args.base_repo_name,
        private=private,
        token=args.token,
    )
    
    print(f"\n📚 Usage after pushing:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{args.hub_username}/{args.base_repo_name}-nq_32k_dev')")
    print(f"   df = dataset['dev'].to_pandas()  # or 'test' if split='test'")


if __name__ == "__main__":
    main()

