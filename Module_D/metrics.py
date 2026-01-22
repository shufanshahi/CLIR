"""
Information Retrieval Evaluation Metrics
Implements Precision@K, Recall@K, nDCG@K, MRR, and other IR metrics.
"""

import math
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

class IRMetrics:
    """
    Class for computing standard Information Retrieval evaluation metrics.
    """
    
    @staticmethod
    def precision_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs (ranked)
            k: Cut-off position
            
        Returns:
            Precision@K score
        """
        if k <= 0:
            return 0.0
            
        # Take top k retrieved documents
        top_k = retrieved_docs[:k]
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)
        
        return relevant_in_top_k / k
    
    @staticmethod
    def recall_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs (ranked)
            k: Cut-off position
            
        Returns:
            Recall@K score
        """
        if not relevant_docs:
            return 0.0
            
        # Take top k retrieved documents
        top_k = retrieved_docs[:k]
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    @staticmethod
    def ndcg_at_k(relevance_scores: Dict[str, float], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            relevance_scores: Dictionary mapping doc_id to relevance score (0-1 or binary)
            retrieved_docs: List of retrieved document IDs (ranked)
            k: Cut-off position
            
        Returns:
            nDCG@K score
        """
        if k <= 0:
            return 0.0
        
        # Take top k retrieved documents
        top_k = retrieved_docs[:k]
        
        # Calculate DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(top_k):
            relevance = relevance_scores.get(doc_id, 0.0)
            # DCG formula: rel_i / log2(i + 2)
            dcg += relevance / math.log2(i + 2) if i > 0 else relevance
        
        # Calculate IDCG@K (Ideal DCG)
        # Sort all relevant documents by relevance score
        ideal_relevance = sorted(relevance_scores.values(), reverse=True)
        ideal_relevance_k = ideal_relevance[:k]
        
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance_k):
            idcg += relevance / math.log2(i + 2) if i > 0 else relevance
        
        # Handle division by zero
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    @staticmethod
    def mean_reciprocal_rank(relevant_docs_list: List[Set[str]], retrieved_docs_list: List[List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            relevant_docs_list: List of sets, each containing relevant doc IDs for a query
            retrieved_docs_list: List of lists, each containing retrieved doc IDs for a query
            
        Returns:
            MRR score
        """
        if len(relevant_docs_list) != len(retrieved_docs_list):
            raise ValueError("Number of queries must match in relevant and retrieved lists")
        
        reciprocal_ranks = []
        
        for relevant_docs, retrieved_docs in zip(relevant_docs_list, retrieved_docs_list):
            if not relevant_docs:
                reciprocal_ranks.append(0.0)
                continue
            
            # Find rank of first relevant document
            rr = 0.0
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id in relevant_docs:
                    rr = 1.0 / (i + 1)
                    break
            
            reciprocal_ranks.append(rr)
        
        return np.mean(reciprocal_ranks)
    
    @staticmethod
    def average_precision(relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
        """
        Calculate Average Precision.
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs (ranked)
            
        Returns:
            Average Precision score
        """
        if not relevant_docs:
            return 0.0
        
        precisions = []
        num_relevant_found = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                precisions.append(precision_at_i)
        
        return np.mean(precisions) if precisions else 0.0
    
    @staticmethod
    def mean_average_precision(relevant_docs_list: List[Set[str]], retrieved_docs_list: List[List[str]]) -> float:
        """
        Calculate Mean Average Precision (MAP).
        
        Args:
            relevant_docs_list: List of sets, each containing relevant doc IDs for a query
            retrieved_docs_list: List of lists, each containing retrieved doc IDs for a query
            
        Returns:
            MAP score
        """
        if len(relevant_docs_list) != len(retrieved_docs_list):
            raise ValueError("Number of queries must match in relevant and retrieved lists")
        
        ap_scores = []
        
        for relevant_docs, retrieved_docs in zip(relevant_docs_list, retrieved_docs_list):
            ap = IRMetrics.average_precision(relevant_docs, retrieved_docs)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    @staticmethod
    def hit_rate_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate Hit Rate@K (whether at least one relevant document is in top k).
        
        Args:
            relevant_docs: Set of relevant document IDs
            retrieved_docs: List of retrieved document IDs (ranked)
            k: Cut-off position
            
        Returns:
            Hit Rate@K score (0 or 1)
        """
        if k <= 0 or not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        return 1.0 if any(doc_id in relevant_docs for doc_id in top_k) else 0.0
    
    @staticmethod
    def evaluate_query_set(relevant_docs_dict: Dict[str, Set[str]], 
                          retrieved_docs_dict: Dict[str, List[str]], 
                          k_values: List[int] = [5, 10, 20, 50]) -> Dict[str, Any]:
        """
        Evaluate a set of queries with multiple metrics.
        
        Args:
            relevant_docs_dict: Dictionary mapping query_id to set of relevant doc IDs
            retrieved_docs_dict: Dictionary mapping query_id to list of retrieved doc IDs
            k_values: List of k values for precision/recall calculations
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if set(relevant_docs_dict.keys()) != set(retrieved_docs_dict.keys()):
            raise ValueError("Query IDs must match in relevant and retrieved dictionaries")
        
        results = {
            'num_queries': len(relevant_docs_dict),
            'metrics': {}
        }
        
        # Prepare lists for MRR and MAP calculation
        relevant_list = []
        retrieved_list = []
        
        # Calculate metrics for each k value
        for k in k_values:
            precisions = []
            recalls = []
            hit_rates = []
            ndcgs = []
            
            for query_id in relevant_docs_dict.keys():
                relevant_docs = relevant_docs_dict[query_id]
                retrieved_docs = retrieved_docs_dict[query_id]
                
                # For nDCG, we need relevance scores (binary in this case)
                relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}
                
                # Calculate metrics
                precision_k = IRMetrics.precision_at_k(relevant_docs, retrieved_docs, k)
                recall_k = IRMetrics.recall_at_k(relevant_docs, retrieved_docs, k)
                hit_rate_k = IRMetrics.hit_rate_at_k(relevant_docs, retrieved_docs, k)
                ndcg_k = IRMetrics.ndcg_at_k(relevance_scores, retrieved_docs, k)
                
                precisions.append(precision_k)
                recalls.append(recall_k)
                hit_rates.append(hit_rate_k)
                ndcgs.append(ndcg_k)
                
                relevant_list.append(relevant_docs)
                retrieved_list.append(retrieved_docs)
            
            # Store average metrics for this k
            results['metrics'][f'precision@{k}'] = np.mean(precisions)
            results['metrics'][f'recall@{k}'] = np.mean(recalls)
            results['metrics'][f'hit_rate@{k}'] = np.mean(hit_rates)
            results['metrics'][f'ndcg@{k}'] = np.mean(ndcgs)
        
        # Calculate MRR and MAP (k-independent)
        results['metrics']['mrr'] = IRMetrics.mean_reciprocal_rank(relevant_list, retrieved_list)
        results['metrics']['map'] = IRMetrics.mean_average_precision(relevant_list, retrieved_list)
        
        return results
    
    @staticmethod
    def check_target_metrics(metrics: Dict[str, float], 
                           targets: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Check if metrics meet target thresholds.
        
        Args:
            metrics: Dictionary of computed metrics
            targets: Dictionary of target thresholds (default: assignment targets)
            
        Returns:
            Dictionary with pass/fail status for each metric
        """
        if targets is None:
            targets = {
                'precision@10': 0.6,
                'recall@50': 0.5,
                'ndcg@10': 0.5,
                'mrr': 0.4
            }
        
        status = {}
        overall_pass = True
        
        for metric, target_value in targets.items():
            computed_value = metrics.get(metric, 0.0)
            passed = computed_value >= target_value
            status[metric] = {
                'computed': computed_value,
                'target': target_value,
                'passed': passed,
                'gap': computed_value - target_value
            }
            
            if not passed:
                overall_pass = False
        
        status['overall_passed'] = overall_pass
        
        return status

if __name__ == "__main__":
    # Test the metrics with sample data
    print("Testing IR Metrics...")
    
    # Sample data
    relevant_docs = {'doc1', 'doc3', 'doc5', 'doc8'}
    retrieved_docs = ['doc2', 'doc1', 'doc4', 'doc3', 'doc6', 'doc5', 'doc7', 'doc8', 'doc9', 'doc10']
    
    # Test individual metrics
    precision_10 = IRMetrics.precision_at_k(relevant_docs, retrieved_docs, 10)
    recall_10 = IRMetrics.recall_at_k(relevant_docs, retrieved_docs, 10)
    recall_50 = IRMetrics.recall_at_k(relevant_docs, retrieved_docs, 50)
    
    relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}
    ndcg_10 = IRMetrics.ndcg_at_k(relevance_scores, retrieved_docs, 10)
    
    print(f"Precision@10: {precision_10:.3f}")
    print(f"Recall@10: {recall_10:.3f}")
    print(f"Recall@50: {recall_50:.3f}")
    print(f"nDCG@10: {ndcg_10:.3f}")
    
    # Test MRR with multiple queries
    relevant_docs_list = [
        {'doc1', 'doc3'},      # Query 1
        {'doc5', 'doc8', 'doc2'},  # Query 2
        {'doc4'}               # Query 3
    ]
    
    retrieved_docs_list = [
        ['doc2', 'doc1', 'doc4', 'doc3'],      # Query 1 results
        ['doc1', 'doc2', 'doc5', 'doc6'],      # Query 2 results
        ['doc3', 'doc4', 'doc5']               # Query 3 results
    ]
    
    mrr = IRMetrics.mean_reciprocal_rank(relevant_docs_list, retrieved_docs_list)
    print(f"MRR: {mrr:.3f}")
    
    # Test target checking
    metrics = {
        'precision@10': precision_10,
        'recall@50': recall_50,
        'ndcg@10': ndcg_10,
        'mrr': mrr
    }
    
    status = IRMetrics.check_target_metrics(metrics)
    print(f"\nTarget Status: {'PASSED' if status['overall_passed'] else 'FAILED'}")
    for metric, result in status.items():
        if metric != 'overall_passed':
            print(f"  {metric}: {result['computed']:.3f} (target: {result['target']:.3f}) - {'PASS' if result['passed'] else 'FAIL'}")
