# utils/evaluation.py

from typing import List, Dict

def compute_hits_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """
    Compute Hits@k metric.
    """
    return int(any(pred in ground_truth for pred in predictions[:k])) / len(ground_truth)

def compute_f1_score(prediction: str, ground_truth: List[str]) -> float:
    """
    Compute F1 score for a single prediction.
    """
    if prediction in ground_truth:
        precision = recall = 1.0
    else:
        precision = recall = 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def evaluate_performance(results: List[Dict]) -> Dict[str, float]:
    """
    Evaluate model performance based on prediction results.
    
    :param results: List of dictionaries, each containing 'question', 'predicted_answer', and 'true_answer'
    :return: Dictionary with performance metrics
    """
    hits_at_1 = 0
    f1_scores = []

    for result in results:
        predicted = result['predicted_answer']
        ground_truth = result['true_answer']

        # Compute Hits@1
        hits_at_1 += compute_hits_at_k([predicted], ground_truth, k=1)

        # Compute F1 score
        f1_scores.append(compute_f1_score(predicted, ground_truth))

    num_questions = len(results)
    avg_hits_at_1 = hits_at_1 / num_questions
    avg_f1_score = sum(f1_scores) / num_questions

    return {
        'hits@1': avg_hits_at_1,
        'f1_score': avg_f1_score
    }