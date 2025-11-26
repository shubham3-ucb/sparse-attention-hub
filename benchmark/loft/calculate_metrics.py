"""LOFT RAG evaluation metrics - exact implementation from LOFT.

This module contains the exact evaluation functions from LOFT's evaluation codebase,
ensuring 100% fidelity with LOFT's evaluation methodology.
"""

import ast
import collections
import re
import string
import unicodedata
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import scipy.optimize


# ============================================================================
# LOFT's exact normalization functions (from evaluation/utils.py)
# ============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer string. Taken from SQuAD evaluation.
    
    This is LOFT's exact normalization function:
    - Unicode NFD normalization
    - Remove articles (a, an, the)
    - Fix whitespace
    - Remove punctuation
    - Lowercase
    """
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_answers(answers: List[str]) -> List[str]:
    """Normalize a list of answers."""
    return [normalize_answer(answer) for answer in answers]


def get_tokens(s: str) -> List[str]:
    """Get tokens from normalized string. Taken from SQuAD evaluation."""
    if not s:
        return []
    return normalize_answer(s).split()


# ============================================================================
# LOFT's exact single-value RAG metrics (from evaluation/utils.py)
# ============================================================================

def compute_em(gold_answers: List[str], pred_answer: str) -> float:
    """Calculates exact match score. Taken from SQuAD evaluation."""
    return max([float(ga == pred_answer) for ga in gold_answers])


def compute_subspan_em(gold_answers: List[str], pred_answer: str) -> float:
    """Calculates subspan match score."""
    return max([1.0 if ga in pred_answer else 0.0 for ga in gold_answers])


def compute_f1(gold_answers: List[str], pred_answer: str) -> float:
    """Calculates F1 score. Taken from SQuAD evaluation."""
    pred_toks = get_tokens(pred_answer)

    f1_scores = []
    for ga in gold_answers:
        gold_toks = get_tokens(ga)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        if not gold_toks or not pred_toks:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            f1 = float(gold_toks == pred_toks)
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)

    return max(f1_scores)


# ============================================================================
# LOFT's exact multi-value RAG metrics (from evaluation/rag.py)
# ============================================================================

def compute_em_multi_value(
    gold_answers: List[str], pred_answers: List[str]
) -> float:
    """Calculates exact match score for multi-value RAG. Taken from SQuAD evaluation."""
    return float(set(gold_answers) == set(pred_answers))


def compute_coverage(gold_answers: List[str], pred_answers: List[str]) -> float:
    """Calculates coverage of gold_answers in pred_answers."""
    return len(set(pred_answers).intersection(set(gold_answers))) / float(
        len(gold_answers)
    )


def compute_multi_value_subspan_em(
    gold_answers: List[str], pred_answers: List[str]
) -> float:
    """Calculates subspan match score. Adopted from DROP evaluation."""
    scores = np.zeros([len(gold_answers), len(pred_answers)])
    for gold_index, gold_item in enumerate(gold_answers):
        for pred_index, pred_item in enumerate(pred_answers):
            if gold_item in pred_item or pred_item in gold_item:
                scores[gold_index, pred_index] = 1
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-scores)
    aligned_scores = np.zeros(len(gold_answers))
    for r, c in zip(row_ind, col_ind):
        aligned_scores[r] = scores[r, c]
    return float(all(aligned_scores))


# ============================================================================
# Answer extraction from model output (adapted from LOFT's utils.py)
# ============================================================================

def extract_prediction(
    model_output: str, answer_prefix: str = "final answer"
) -> List[str]:
    """Extracts the prediction from the model output.
    
    This is LOFT's exact extraction logic, adapted for our use case.
    Looks for format: "Final Answer: [answer1, answer2]" and extracts the list.
    
    Args:
        model_output: Raw model output string (includes answer_prefix)
        answer_prefix: The prefix to look for (default: "final answer")
    
    Returns:
        List of extracted answers (strings)
    """
    def _escape_single_quotes(s: str):
        # Converts patterns like "['child bride', 'the devil's sleep']" to
        # "['child bride', 'the devil\'s sleep']" to allow for proper parsing.
        pattern = r"([a-zA-Z0-9])'([a-zA-Z0-9])"
        replacement = r"\1\'\2"
        return re.sub(pattern, replacement, s)

    # Remove formatting.
    model_output = model_output.replace("*", "").replace("`", "")
    model_output = model_output.strip().split("\n")
    # Extract the predictions from the model output
    preds = []
    for l in model_output:
        # Turns the string "Final Answer: [1, ...]" into the list [1, ...]
        if "[" in l and "]" in l:
            if answer_prefix not in l.lower():
                # Warning but continue - might still be valid
                pass
            pred_start_index = l.find("[")
            pred_end_index = l.rfind("]") + 1  # Finds the last "]"
            pred_as_str = l[pred_start_index:pred_end_index].strip()
            try:
                pred_as_str = _escape_single_quotes(pred_as_str)
                parsed = ast.literal_eval(pred_as_str)
                # Ensure we return list of strings
                if isinstance(parsed, list):
                    preds = [str(p) for p in parsed]
                else:
                    preds = [str(parsed)]
                break
            except Exception as e:
                # If parsing fails, continue to next line or fallback
                pass
    
    # Fallback: if no list format found, try to extract text after answer_prefix
    if not preds:
        for l in model_output:
            if answer_prefix.lower() in l.lower():
                # Find position after prefix
                prefix_idx = l.lower().find(answer_prefix.lower())
                after_prefix = l[prefix_idx + len(answer_prefix):].strip()
                # Remove colon if present
                after_prefix = after_prefix.lstrip(":").strip()
                if after_prefix:
                    # Try to parse as list, otherwise treat as single answer
                    try:
                        if "[" in after_prefix and "]" in after_prefix:
                            pred_start = after_prefix.find("[")
                            pred_end = after_prefix.rfind("]") + 1
                            pred_as_str = after_prefix[pred_start:pred_end]
                            preds = ast.literal_eval(pred_as_str)
                            if isinstance(preds, list):
                                preds = [str(p) for p in preds]
                            else:
                                preds = [str(preds)]
                        else:
                            # Single answer
                            preds = [after_prefix]
                    except:
                        preds = [after_prefix]
                break
    
    return preds


# ============================================================================
# Main metrics calculation function
# ============================================================================

def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate LOFT RAG metrics for a DataFrame of results.
    
    This function implements LOFT's exact evaluation logic:
    - Single-value RAG (nq, hotpotqa, musique): EM, subspan_EM, F1
    - Multi-value RAG (qampari, quest): EM, coverage, subspan_EM
    
    Args:
        df: DataFrame with columns:
            - predicted_answer: Raw model output (string)
            - answers: Ground truth answers (list of strings)
            - task: Task name (e.g., "nq_32k", "qampari_128k")
            - answer_prefix: Answer prefix used (default: "Final Answer: ")
    
    Returns:
        Dictionary with metrics:
        - For single-value: em, subspan_em, f1
        - For multi-value: em, coverage, subspan_em
        - Overall metrics aggregated across all samples
    """
    if len(df) == 0:
        return {"error": "No results to evaluate"}
    
    # Determine if this is multi-value or single-value RAG
    # Multi-value datasets: qampari, quest
    task_name = df["task"].iloc[0] if "task" in df.columns else ""
    is_multi_value = task_name.startswith("qampari") or task_name.startswith("quest")
    
    # Get answer_prefix (default to "Final Answer: ")
    answer_prefix = df["answer_prefix"].iloc[0] if "answer_prefix" in df.columns else "Final Answer: "
    
    # Process each row
    all_em_scores = []
    all_subspan_em_scores = []
    all_f1_scores = []
    all_coverage_scores = []
    
    for idx, row in df.iterrows():
        # Get ground truth answers
        gold_answers = row["answers"]
        if not isinstance(gold_answers, list):
            gold_answers = [gold_answers] if gold_answers else []
        
        # Normalize gold answers (LOFT's exact normalization)
        gold_answers_normalized = normalize_answers([str(ga) for ga in gold_answers])
        
        # Extract and normalize predicted answers
        predicted_output = str(row["predicted_answer"]) if pd.notna(row["predicted_answer"]) else ""
        pred_answers_raw = extract_prediction(predicted_output, answer_prefix.lower())
        
        if not pred_answers_raw:
            # Empty prediction - all metrics are 0.0 (LOFT behavior)
            all_em_scores.append(0.0)
            all_subspan_em_scores.append(0.0)
            if is_multi_value:
                all_coverage_scores.append(0.0)
            else:
                all_f1_scores.append(0.0)
            continue
        
        # Normalize predicted answers
        pred_answers_normalized = normalize_answers(pred_answers_raw)
        
        if is_multi_value:
            # Multi-value RAG evaluation
            em = compute_em_multi_value(gold_answers_normalized, pred_answers_normalized)
            coverage = compute_coverage(gold_answers_normalized, pred_answers_normalized)
            subspan_em = compute_multi_value_subspan_em(
                gold_answers_normalized, pred_answers_normalized
            )
            
            all_em_scores.append(em)
            all_coverage_scores.append(coverage)
            all_subspan_em_scores.append(subspan_em)
        else:
            # Single-value RAG evaluation
            # LOFT takes first prediction if multiple found (with warning)
            if len(pred_answers_normalized) > 1:
                # Warning but use first (LOFT behavior)
                pass
            
            pred_answer = pred_answers_normalized[0]
            em = compute_em(gold_answers_normalized, pred_answer)
            subspan_em = compute_subspan_em(gold_answers_normalized, pred_answer)
            f1 = compute_f1(gold_answers_normalized, pred_answer)
            
            all_em_scores.append(em)
            all_subspan_em_scores.append(subspan_em)
            all_f1_scores.append(f1)
    
    # Aggregate metrics
    metrics = {
        "em": float(np.mean(all_em_scores)),
        "subspan_em": float(np.mean(all_subspan_em_scores)),
    }
    
    if is_multi_value:
        metrics["coverage"] = float(np.mean(all_coverage_scores))
    else:
        metrics["f1"] = float(np.mean(all_f1_scores))
    
    metrics["num_samples"] = len(df)
    
    return metrics

