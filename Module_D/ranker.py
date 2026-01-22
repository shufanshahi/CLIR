"""
Ranking and Scoring Module for CLIR System
Implements confidence scoring, ranking functions, and low-confidence warnings.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_C.retriever import CLIRRetriever

class CLIRRanker:
    """
    Main Ranking Class that implements confidence scoring and ranking.
    """
    
    def __init__(self, retriever: CLIRRetriever, confidence_threshold: float = 0.20):
        """
        Initialize the ranker with a retriever instance.
        
        Args:
            retriever: CLIRRetriever instance
            confidence_threshold: Threshold for low-confidence warnings
        """
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Model weights for hybrid scoring (can be tuned)
        self.model_weights = {
            'lexical_bm25': 0.3,
            'semantic': 0.5,
            'fuzzy': 0.2
        }
        
    def rank_query(self, query: str, k: int = 10, model: str = 'hybrid') -> Dict[str, Any]:
        """
        Rank documents for a query with confidence scoring and timing.
        
        Args:
            query: Input query string
            k: Number of top documents to return
            model: Retrieval model ('lexical', 'semantic', 'fuzzy', 'hybrid')
            
        Returns:
            Dictionary containing ranked results, scores, timing, and warnings
        """
        start_time = time.time()
        
        # Initialize timing breakdown
        timing_breakdown = {
            'translation_time': 0,
            'retrieval_time': 0,
            'ranking_time': 0
        }
        
        # Track query processing time
        proc_start = time.time()
        
        # Get results from specified model
        if model == 'lexical':
            results = self.retriever.search_lexical(query, k=k*2)  # Get more for better ranking
        elif model == 'semantic':
            results = self.retriever.search_semantic(query, k=k*2)
        elif model == 'fuzzy':
            results = self.retriever.search_fuzzy(query, k=k*2)
        elif model == 'hybrid':
            # Pass model weights to the hybrid search (supports 3-way fusion)
            results = self.retriever.search_hybrid(
                query, 
                k=k*2, 
                w_lex=self.model_weights.get('lexical_bm25', 0.3),
                w_sem=self.model_weights.get('semantic', 0.5),
                w_fuz=self.model_weights.get('fuzzy', 0.2)
            )
        else:
            raise ValueError(f"Unknown model: {model}")
            
        timing_breakdown['retrieval_time'] = time.time() - proc_start
        
        # Normalize scores and calculate confidence
        rank_start = time.time()
        ranked_results = self._normalize_and_rank(results, model)
        timing_breakdown['ranking_time'] = time.time() - rank_start
        
        # Take top k
        final_results = ranked_results[:k]
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Check for low confidence warning
        low_confidence_warning = None
        if final_results and final_results[0]['confidence_score'] < self.confidence_threshold:
            low_confidence_warning = (
                f"⚠️ Warning: Retrieved results may not be relevant. "
                f"Matching confidence is low (score: {final_results[0]['confidence_score']:.3f})."
            )
        
        return {
            'query': query,
            'model': model,
            'results': final_results,
            'total_time_ms': total_time,
            'timing_breakdown': timing_breakdown,
            'low_confidence_warning': low_confidence_warning,
            'num_results_found': len(final_results)
        }
    
    def _normalize_and_rank(self, results: List[Dict], model: str) -> List[Dict]:
        """
        Normalize scores to [0,1] range and add confidence scoring.
        
        Args:
            results: List of retrieval results
            model: Model type used for retrieval
            
        Returns:
            List of results with normalized scores and confidence
        """
        if not results:
            return []
        
        # Normalize scores based on model type
        if model == 'semantic':
            # Semantic scores are already cosine similarity (roughly 0-1)
            for result in results:
                result['confidence_score'] = max(0.0, result['score'])  # Clip negative values
                
        elif model == 'fuzzy':
            # Fuzzy scores are already normalized (0-1)
            for result in results:
                result['confidence_score'] = result['score']
                
        else:  # lexical, hybrid, etc.
            # These scores need normalization
            scores = [r['score'] for r in results]
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                
                if max_score > min_score:
                    # Min-max normalization
                    for result in results:
                        norm_score = (result['score'] - min_score) / (max_score - min_score)
                        result['confidence_score'] = norm_score
                else:
                    # All scores are the same
                    for result in results:
                        result['confidence_score'] = 0.5
        
        # Sort by confidence score (descending)
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Add rank information
        for i, result in enumerate(results):
            result['rank'] = i + 1
            
        return results
    
    def batch_rank(self, queries: List[str], k: int = 10, model: str = 'hybrid') -> List[Dict]:
        """
        Rank multiple queries in batch.
        
        Args:
            queries: List of query strings
            k: Number of top documents per query
            model: Retrieval model to use
            
        Returns:
            List of ranking results for each query
        """
        batch_results = []
        
        for query in queries:
            try:
                result = self.rank_query(query, k=k, model=model)
                batch_results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing query '{query}': {e}")
                batch_results.append({
                    'query': query,
                    'error': str(e),
                    'results': [],
                    'total_time_ms': 0,
                    'num_results_found': 0
                })
        
        return batch_results
    
    def get_model_comparison(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Get results from all models for comparison.
        
        Args:
            query: Input query
            k: Number of results per model
            
        Returns:
            Dictionary with results from all models
        """
        models = ['lexical', 'semantic', 'fuzzy', 'hybrid']
        comparison = {}
        
        for model in models:
            try:
                result = self.rank_query(query, k=k, model=model)
                comparison[model] = result
            except Exception as e:
                self.logger.error(f"Error with model {model} for query '{query}': {e}")
                comparison[model] = {'error': str(e)}
        
        return {
            'query': query,
            'models': comparison,
            'timestamp': time.time()
        }
    
    def set_model_weights(self, weights: Dict[str, float]):
        """
        Update model weights for hybrid scoring.
        
        Args:
            weights: Dictionary of model weights (should sum to 1.0)
        """
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        self.model_weights.update(weights)
        self.logger.info(f"Updated model weights: {self.model_weights}")
    
    def get_performance_stats(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate performance statistics from batch results.
        
        Args:
            batch_results: List of ranking results
            
        Returns:
            Performance statistics dictionary
        """
        if not batch_results:
            return {}
        
        # Filter out error results
        valid_results = [r for r in batch_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results found'}
        
        # Calculate timing statistics
        total_times = [r['total_time_ms'] for r in valid_results]
        retrieval_times = [r['timing_breakdown']['retrieval_time'] * 1000 for r in valid_results]
        ranking_times = [r['timing_breakdown']['ranking_time'] * 1000 for r in valid_results]
        
        # Calculate confidence statistics
        all_confidences = []
        for result in valid_results:
            if result['results']:
                all_confidences.append(result['results'][0]['confidence_score'])
        
        stats = {
            'num_queries': len(valid_results),
            'timing': {
                'total_time': {
                    'mean_ms': np.mean(total_times),
                    'std_ms': np.std(total_times),
                    'min_ms': np.min(total_times),
                    'max_ms': np.max(total_times)
                },
                'retrieval_time': {
                    'mean_ms': np.mean(retrieval_times),
                    'std_ms': np.std(retrieval_times),
                    'min_ms': np.min(retrieval_times),
                    'max_ms': np.max(retrieval_times)
                },
                'ranking_time': {
                    'mean_ms': np.mean(ranking_times),
                    'std_ms': np.std(ranking_times),
                    'min_ms': np.min(ranking_times),
                    'max_ms': np.max(ranking_times)
                }
            },
            'confidence': {
                'mean_top_score': np.mean(all_confidences) if all_confidences else 0,
                'std_top_score': np.std(all_confidences) if all_confidences else 0,
                'min_top_score': np.min(all_confidences) if all_confidences else 0,
                'max_top_score': np.max(all_confidences) if all_confidences else 0
            },
            'low_confidence_queries': len([r for r in valid_results if r.get('low_confidence_warning')]),
            'avg_results_per_query': np.mean([r['num_results_found'] for r in valid_results])
        }
        
        return stats

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    try:
        retriever = CLIRRetriever()
        ranker = CLIRRanker(retriever)
        
        query = "election"
        print(f"\nTesting ranking for query: {query}")
        
        # Test single model ranking
        result = ranker.rank_query(query, k=5, model='hybrid')
        print(f"\nResults for {result['model']} model:")
        print(f"Total time: {result['total_time_ms']:.2f} ms")
        if result['low_confidence_warning']:
            print(f"Warning: {result['low_confidence_warning']}")
        
        for i, res in enumerate(result['results'][:3]):
            print(f"  {i+1}. [{res['confidence_score']:.3f}] {res['title'][:50]} ({res['language']})")
        
        # Test model comparison
        comparison = ranker.get_model_comparison(query, k=3)
        print(f"\nModel comparison for query: {query}")
        for model, comp_result in comparison['models'].items():
            if 'error' not in comp_result and comp_result['results']:
                top_score = comp_result['results'][0]['confidence_score']
                print(f"  {model}: {top_score:.3f} confidence")
            else:
                print(f"  {model}: Error")
                
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Module A, B, and C are properly set up with indexed data.")
