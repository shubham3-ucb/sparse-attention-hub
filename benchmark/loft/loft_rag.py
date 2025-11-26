"""LOFT RAG benchmark implementation for long-context retrieval-augmented generation.

This benchmark implements LOFT's RAG evaluation exactly as specified in the LOFT paper,
ensuring 100% fidelity with LOFT's evaluation methodology and metrics.
"""

from typing import Dict, Any, List
import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("loft_rag")
class LoftRag(Benchmark):
    """LOFT RAG benchmark for evaluating long-context retrieval-augmented generation.

    LOFT (Long-context Open Foundation Tasks) RAG evaluates the ability of models to
    answer questions given long retrieved contexts. This benchmark includes:
    
    - Single-value RAG datasets: nq, hotpotqa, musique
    - Multi-value RAG datasets: qampari, quest
    
    Each dataset is available in multiple context lengths: 32k, 128k, 1m.
    
    Metrics:
    - Single-value: EM (Exact Match), Subspan EM, F1
    - Multi-value: EM, Coverage, Subspan EM
    
    All metrics use LOFT's exact evaluation functions, ensuring compatibility with
    published LOFT results.

    Example:
        >>> loft_rag = LoftRag(subsets_to_run=["nq_32k", "hotpotqa_128k"])
        >>> results = loft_rag.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"EM score: {results['nq_32k']['em']}")
    """

    # All available LOFT RAG datasets
    # Format: {dataset}_{length} (e.g., "nq_32k", "qampari_128k")
    all_datasets: List[str] = [
        # Natural Questions (single-value)
        "nq_32k",
        "nq_128k",
        "nq_1m",
        # HotpotQA (single-value)
        "hotpotqa_32k",
        "hotpotqa_128k",
        "hotpotqa_1m",
        # MuSiQue (single-value)
        "musique_32k",
        "musique_128k",
        "musique_1m",
        # QAMPARI (multi-value)
        "qampari_32k",
        "qampari_128k",
        "qampari_1m",
        # Quest (multi-value)
        "quest_32k",
        "quest_128k",
        "quest_1m",
    ]
    
    benchmark_name: str = "loft_rag"
    huggingface_dataset_id: str = "f20180301/rag"  # Index repo - actual datasets are under f20180301/loft-rag-{dataset}-{length}

    def _load_datasets(self) -> pd.DataFrame:
        """Load LOFT RAG datasets from HuggingFace Hub.
        
        LOFT RAG datasets are organized as:
        - Repo: f20180301/loft-rag-{dataset}-{length}
        - Splits: dev, test
        - We load both splits and combine them
        
        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading LOFT RAG datasets: {self.subsets_to_run}")
        dfs = []
        
        for subset in self.subsets_to_run:
            try:
                # Parse subset name (e.g., "nq_32k" -> dataset="nq", length="32k")
                parts = subset.split("_")
                if len(parts) < 2:
                    print(f"  ❌ Invalid subset format: {subset} (expected format: dataset_length)")
                    continue
                
                length = parts[-1]  # Last part is length (32k, 128k, 1m)
                dataset = "_".join(parts[:-1])  # Everything before last underscore is dataset name
                
                # Construct HuggingFace dataset ID
                hf_dataset_id = f"f20180301/loft-rag-{dataset}-{length}"
                
                from datasets import load_dataset
                
                # Load both dev and test splits
                try:
                    dataset_dict = load_dataset(hf_dataset_id)
                    
                    # Combine dev and test splits
                    subset_dfs = []
                    for split_name in ["dev", "test"]:
                        if split_name in dataset_dict:
                            split_df = dataset_dict[split_name].to_pandas()
                            split_df["split"] = split_name
                            subset_dfs.append(split_df)
                    
                    if not subset_dfs:
                        print(f"  ❌ No splits found for {subset} ({hf_dataset_id})")
                        continue
                    
                    # Combine splits
                    subset_df = pd.concat(subset_dfs, ignore_index=True)
                    subset_df["task"] = subset  # Ensure task column matches subset name
                    
                    dfs.append(subset_df)
                    print(f"  ✓ Loaded {len(subset_df)} samples from {subset} ({hf_dataset_id})")
                    
                except Exception as load_error:
                    print(f"  ❌ Failed to load {subset} ({hf_dataset_id}): {str(load_error)}")
                    continue
                    
            except Exception as subset_error:
                print(f"  ❌ Error processing subset {subset}: {str(subset_error)}")
                continue
        
        if not dfs:
            raise Exception("No LOFT RAG subsets could be loaded successfully")
        
        # Combine all subset DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")
        
        # Verify required columns exist
        required_columns = ["context", "question", "answers", "task", "answer_prefix", "max_new_tokens"]
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for LOFT RAG results.

        This implements LOFT's exact evaluation logic:
        - Single-value RAG (nq, hotpotqa, musique): EM, Subspan EM, F1
        - Multi-value RAG (qampari, quest): EM, Coverage, Subspan EM

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - task: Dataset/task name (e.g., "nq_32k")
                - predicted_answer: Model's raw predicted answer (includes answer_prefix)
                - answers: Ground truth answers (list of strings)
                - answer_prefix: Answer prefix used (default: "Final Answer: ")

        Returns:
            Dictionary containing computed metrics:
            - Overall metrics aggregated across all tasks
            - Per-task metrics for each dataset+length combination
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Group results by task
        task_groups = results_df.groupby("task")
        task_metrics: Dict[str, Dict[str, float]] = {}
        all_em_scores: List[float] = []
        all_subspan_em_scores: List[float] = []
        all_f1_scores: List[float] = []
        all_coverage_scores: List[float] = []
        
        for task_name, task_df in task_groups:
            try:
                # Calculate metrics for this task using LOFT's exact functions
                metrics = calculate_metrics(task_df)
                
                if "error" in metrics:
                    print(f"  ❌ Error evaluating {task_name}: {metrics['error']}")
                    continue
                
                task_metrics[task_name] = metrics
                
                # Aggregate for overall metrics
                all_em_scores.append(metrics["em"])
                all_subspan_em_scores.append(metrics["subspan_em"])
                
                if "f1" in metrics:
                    all_f1_scores.append(metrics["f1"])
                if "coverage" in metrics:
                    all_coverage_scores.append(metrics["coverage"])
                
                # Print task results
                metric_str = f"EM={metrics['em']:.4f}, Subspan_EM={metrics['subspan_em']:.4f}"
                if "f1" in metrics:
                    metric_str += f", F1={metrics['f1']:.4f}"
                if "coverage" in metrics:
                    metric_str += f", Coverage={metrics['coverage']:.4f}"
                print(f"  ✓ {task_name}: {metric_str}")
                    
            except Exception as e:
                print(f"  ❌ Error evaluating task {task_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Compute overall metrics
        overall_metrics: Dict[str, Any] = {
            "overall": {
                "em": float(sum(all_em_scores) / len(all_em_scores)) if all_em_scores else 0.0,
                "subspan_em": float(sum(all_subspan_em_scores) / len(all_subspan_em_scores)) if all_subspan_em_scores else 0.0,
            },
            "task_metrics": {task: {k: round(v, 4) if isinstance(v, float) else v 
                                   for k, v in metrics.items()} 
                            for task, metrics in task_metrics.items()},
            "summary": {
                "total_tasks": len(task_metrics),
                "total_samples": len(results_df)
            }
        }
        
        # Add F1 and Coverage to overall if available
        if all_f1_scores:
            overall_metrics["overall"]["f1"] = float(sum(all_f1_scores) / len(all_f1_scores))
        if all_coverage_scores:
            overall_metrics["overall"]["coverage"] = float(sum(all_coverage_scores) / len(all_coverage_scores))
        
        # Round overall metrics
        overall_metrics["overall"] = {k: round(v, 4) if isinstance(v, float) else v 
                                      for k, v in overall_metrics["overall"].items()}
        
        return overall_metrics

