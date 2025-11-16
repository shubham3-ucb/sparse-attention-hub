# RULER 16k Benchmark Runner

## Overview

This script (`test_sparse_oracle_ruler16k.py`) runs dense vs. sparse attention evaluation on the **RULER 16k** benchmark, which tests long-context understanding at 16,384 tokens.

## How RULER 16k Works

### Task Structure

RULER 16k has **13 different task types**, each testing different capabilities:

1. **cwe** - Context Window Extension
2. **fwe** - Forward Window Extension  
3. **niah_multikey_1/2/3** - Needle in a Haystack (multiple keys)
4. **niah_multiquery** - Needle in a Haystack (multiple queries)
5. **niah_multivalue** - Needle in a Haystack (multiple values)
6. **niah_single_1/2/3** - Needle in a Haystack (single)
7. **qa_1/2** - Question Answering
8. **vt** - Value Tracing

Each task is a **separate dataset split** with its own samples.

### Sampling Strategy

**Current Implementation: `NUM_SAMPLES` = Total across all tasks**

- Loads all specified tasks (or all 13 by default)
- Combines all samples from all tasks into one list
- Takes the **first `NUM_SAMPLES`** samples total
- This means if you set `NUM_SAMPLES=40`:
  - You get 40 samples total (not 40 per task)
  - Samples are taken sequentially from tasks in order
  - If task 1 has 50 samples, you might get all 40 from task 1, or distributed across tasks

**Example:**
- If you load all 13 tasks and set `NUM_SAMPLES=40`
- You get 40 samples total (could be 10 from task1, 15 from task2, 15 from task3, etc., depending on order)

## Usage

### Basic Run (All Tasks, Default Settings)

```bash
bash run_ruler16k.sh
```

This runs with:
- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct` (default)
- Sparse: `ORACLE_TOPK_HEAVY_SIZE=0.1` (10% sparsity)
- Samples: `NUM_SAMPLES=40` (total across all tasks)
- All 13 tasks

### Custom Configuration

```bash
# Run with specific tasks only
RULER_TASKS="qa_1,qa_2,niah_single_1" NUM_SAMPLES=20 bash run_ruler16k.sh

# Run with different sparsity
ORACLE_TOPK_HEAVY_SIZE=0.5 NUM_SAMPLES=40 bash run_ruler16k.sh

# Run with different model
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct" bash run_ruler16k.sh

# Run with custom output directory
ROOT_OUT="/path/to/results" bash run_ruler16k.sh
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HuggingFace model identifier |
| `NUM_SAMPLES` | `40` | Total number of samples to process (across all tasks) |
| `ORACLE_TOPK_HEAVY_SIZE` | `0.1` | Sparsity level (0.1 = 10%, 1.0 = dense) |
| `PREFILL_CHUNK_SIZE` | `4096` | Chunk size for prefill |
| `MAX_CONTEXT_LENGTH` | `2147483647` | Max context length (no truncation by default) |
| `RULER_TASKS` | (all 13 tasks) | Comma-separated list of tasks to run |
| `ROOT_OUT` | `./output_ruler16k`` | Output directory |

### Available Tasks

All 13 tasks (use comma-separated):
```
cwe, fwe, niah_multikey_1, niah_multikey_2, niah_multikey_3,
niah_multiquery, niah_multivalue, niah_single_1, niah_single_2,
niah_single_3, qa_1, qa_2, vt
```

## Output Structure

```
output_ruler16k/
└── ruler16k_hs0.1_pcs4096_ns40/
    ├── settings.json              # Run configuration
    ├── command.txt                 # Exact command used
    ├── run_command.sh             # Executable rerun script
    ├── log.txt                     # Full log output
    ├── hf_prefill.log              # Sparse attention logs
    ├── raw_results.csv             # All results (dense + sparse)
    ├── test_sparse_results.json    # Results in JSON format
    ├── metrics.json                # Evaluation metrics
    ├── micro_metrics.jsonl         # Detailed attention metrics
    ├── config.json                 # Model/config info
    └── comparison_results/
        ├── dense_results.csv
        ├── sparse_results.csv
        ├── dense_metrics.json
        ├── sparse_metrics.json
        └── comparison_summary.json
```

## Metrics

The script computes:
- **Overall score**: Average string match accuracy across all tasks
- **Task scores**: Per-task string match accuracy
- **Context length scores**: Grouped by context length (always 16384 for RULER 16k)
- **Per-method comparison**: Separate metrics for dense vs. sparse

## Differences from HotpotQA Script

1. **Dataset**: Uses `xAlg-AI/att-hub-ruler-16k` instead of `Xnhyacinth/LongBench`
2. **Data format**: Uses `answer` (string) instead of `answers` (list)
3. **Evaluation**: Uses `ruler16k` benchmark evaluator instead of `longbench`
4. **Tasks**: Supports multiple task subsets (13 tasks vs. single task)
5. **Sampling**: `NUM_SAMPLES` is total across tasks, not per-task

## Notes

- Each sample is processed with both **dense** and **sparse** attention
- Results are saved incrementally (CSV + JSON updated after each sample)
- Metrics are computed incrementally and at the end
- Micro-metrics (attention weight differences) are logged to `micro_metrics.jsonl`

