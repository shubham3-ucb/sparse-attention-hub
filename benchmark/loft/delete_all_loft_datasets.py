#!/usr/bin/env python3
"""
Delete all LOFT RAG datasets from HuggingFace Hub.

This script deletes all datasets matching LOFT/RAG patterns to allow clean re-upload.
"""

import os
import sys
from huggingface_hub import HfApi, login


def delete_all_loft_datasets(org_name: str, token: str, dry_run: bool = False):
    """Delete all LOFT/RAG datasets from the organization."""
    login(token=token)
    api = HfApi(token=token)
    
    print(f"\n{'='*80}")
    print(f"Finding LOFT/RAG datasets in {org_name}")
    print(f"{'='*80}\n")
    
    # List all datasets in the org
    try:
        all_datasets = api.list_datasets(author=org_name)
        loft_datasets = [
            d for d in all_datasets 
            if 'loft' in d.id.lower() or 'rag' in d.id.lower()
        ]
        
        if not loft_datasets:
            print("✅ No LOFT/RAG datasets found. Nothing to delete.")
            return
        
        print(f"Found {len(loft_datasets)} LOFT/RAG datasets to delete:\n")
        for ds in sorted(loft_datasets, key=lambda x: x.id):
            print(f"  - {ds.id}")
        
        if dry_run:
            print("\n🔍 DRY RUN MODE - No datasets will be deleted")
            return
        
        print(f"\n{'='*80}")
        print(f"Deleting {len(loft_datasets)} datasets...")
        print(f"{'='*80}\n")
        
        deleted_count = 0
        failed_count = 0
        
        for ds in sorted(loft_datasets, key=lambda x: x.id):
            repo_id = ds.id
            try:
                print(f"🗑️  Deleting: {repo_id}...", end=" ", flush=True)
                api.delete_repo(
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print("✅ Deleted")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed: {e}")
                failed_count += 1
        
        print(f"\n{'='*80}")
        print(f"✅ Deletion Complete!")
        print(f"{'='*80}")
        print(f"  Deleted: {deleted_count}/{len(loft_datasets)}")
        if failed_count > 0:
            print(f"  Failed: {failed_count}/{len(loft_datasets)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Delete all LOFT RAG datasets from HuggingFace Hub"
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
        "--dry_run",
        action="store_true",
        help="Dry run mode - list datasets without deleting",
    )
    
    args = parser.parse_args()
    
    delete_all_loft_datasets(
        org_name=args.org_name,
        token=args.token,
        dry_run=args.dry_run,
    )
