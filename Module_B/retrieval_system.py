"""
Module B - Complete Retrieval System
Integrates query processing with document indexing for cross-lingual retrieval
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_A.indexer import DocumentIndexer
from Module_B.query_processor import QueryProcessor
from typing import List, Dict, Tuple
import re


class CrossLingualRetrievalSystem:
    """
    Complete CLIR system combining indexing and query processing
    """
    
    def __init__(self, index_path='Module_A/document_index.pkl'):
        """
        Initialize retrieval system
        
        Args:
            index_path: Path to saved index file
        """
        self.indexer = DocumentIndexer()
        self.query_processor = QueryProcessor()
        
        # Load index if exists
        if os.path.exists(index_path):
            self.indexer.load_index(index_path)
        else:
            print(f"Warning: Index file '{index_path}' not found.")
            print("Please run indexer.py first to build the index.")
    
    def tokenize(self, text):
        """Simple tokenization"""
        text = text.lower()
        tokens = re.findall(r'\b[\w]+\b', text)
        return tokens
    
    def retrieve_tfidf(
        self,
        query_terms: List[str],
        target_language: str = None,
        top_k: int = 10
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve documents using TF-IDF scoring
        
        Args:
            query_terms: List of query terms (can be expanded)
            target_language: Filter by language ('bn', 'en', or None for both)
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score, metadata) tuples
        """
        scores = {}
        
        # Calculate TF-IDF scores for each document
        for term in query_terms:
            term_lower = term.lower()
            
            if term_lower in self.indexer.inverted_index:
                for doc_id in self.indexer.inverted_index[term_lower]:
                    # Filter by language if specified
                    if target_language:
                        doc_lang = self.indexer.doc_metadata[doc_id]['language']
                        if doc_lang != target_language:
                            continue
                    
                    tfidf_score = self.indexer.calculate_tf_idf(term_lower, doc_id)
                    
                    if doc_id not in scores:
                        scores[doc_id] = 0.0
                    scores[doc_id] += tfidf_score
        
        # Sort by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Add metadata
        results = [
            (doc_id, score, self.indexer.doc_metadata[doc_id])
            for doc_id, score in ranked_docs
        ]
        
        return results
    
    def retrieve_bm25(
        self,
        query_terms: List[str],
        target_language: str = None,
        top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75
    ) -> List[Tuple[int, float, Dict]]:
        """
        Retrieve documents using BM25 scoring
        
        Args:
            query_terms: List of query terms
            target_language: Filter by language
            top_k: Number of results to return
            k1: BM25 parameter
            b: BM25 parameter
            
        Returns:
            List of (doc_id, score, metadata) tuples
        """
        scores = {}
        
        # Calculate BM25 scores
        for term in query_terms:
            term_lower = term.lower()
            
            if term_lower in self.indexer.inverted_index:
                for doc_id in self.indexer.inverted_index[term_lower]:
                    # Filter by language
                    if target_language:
                        doc_lang = self.indexer.doc_metadata[doc_id]['language']
                        if doc_lang != target_language:
                            continue
                    
                    bm25_score = self.indexer.calculate_bm25(term_lower, doc_id, k1, b)
                    
                    if doc_id not in scores:
                        scores[doc_id] = 0.0
                    scores[doc_id] += bm25_score
        
        # Sort by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Add metadata
        results = [
            (doc_id, score, self.indexer.doc_metadata[doc_id])
            for doc_id, score in ranked_docs
        ]
        
        return results
    
    def search(
        self,
        query: str,
        method: str = 'bm25',
        cross_lingual: bool = True,
        expand_query: bool = True,
        remove_stopwords: bool = True,
        use_translation_api: bool = False,
        top_k: int = 10
    ) -> Dict:
        """
        Main search function - complete CLIR pipeline
        
        Args:
            query: User query
            method: Retrieval method ('tfidf' or 'bm25')
            cross_lingual: Whether to search in both languages
            expand_query: Whether to expand query with synonyms
            remove_stopwords: Whether to remove stopwords
            use_translation_api: Whether to use translation API
            top_k: Number of results per language
            
        Returns:
            Dictionary with results for each language
        """
        # Step 1: Process query
        processed = self.query_processor.process_query(
            query,
            remove_stopwords=remove_stopwords,
            expand=expand_query,
            translate=cross_lingual,
            use_api=use_translation_api
        )
        
        results = {
            'query_info': processed,
            'results': {}
        }
        
        # Step 2: Retrieve documents for each language
        for lang, query_info in processed['queries'].items():
            # Get query terms (expanded if enabled)
            query_terms = self.tokenize(' '.join(query_info['expanded_terms']))
            
            # Retrieve using specified method
            if method == 'tfidf':
                docs = self.retrieve_tfidf(query_terms, lang, top_k)
            else:  # bm25
                docs = self.retrieve_bm25(query_terms, lang, top_k)
            
            results['results'][lang] = {
                'query': query_info['query'],
                'expanded_terms': query_info['expanded_terms'],
                'is_translated': query_info['is_translated'],
                'documents': docs,
                'count': len(docs)
            }
        
        return results
    
    def print_results(self, results: Dict, show_body: bool = False):
        """
        Pretty print search results
        
        Args:
            results: Results dictionary from search()
            show_body: Whether to show document body (can be long)
        """
        print("\n" + "="*80)
        print("SEARCH RESULTS")
        print("="*80)
        
        query_info = results['query_info']
        print(f"\nOriginal Query: {query_info['original_query']}")
        print(f"Detected Language: {query_info['detected_language'].upper()}")
        print(f"Normalized Query: {query_info['normalized_query']}")
        
        if query_info['named_entities']:
            print(f"Named Entities: {', '.join(query_info['named_entities'])}")
        
        print("\n" + "-"*80)
        
        for lang, result_info in results['results'].items():
            print(f"\n{'BANGLA' if lang == 'bn' else 'ENGLISH'} RESULTS")
            print(f"Query: {result_info['query']}")
            if result_info['is_translated']:
                print("(Translated)")
            print(f"Found {result_info['count']} documents")
            print("-"*80)
            
            for rank, (doc_id, score, metadata) in enumerate(result_info['documents'], 1):
                print(f"\n{rank}. [Score: {score:.4f}]")
                print(f"   Title: {metadata['title'][:100]}...")
                print(f"   URL: {metadata['url']}")
                print(f"   Date: {metadata['date']}")
                
                if show_body:
                    body_preview = metadata['body'][:200] + "..."
                    print(f"   Body: {body_preview}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Demo of complete retrieval system"""
    
    # Initialize system
    print("Initializing Cross-Lingual Retrieval System...")
    system = CrossLingualRetrievalSystem(index_path='Module_A/document_index.pkl')
    
    # Test queries
    test_queries = [
        "Bangladesh university",
        "ঢাকা শিক্ষা",
        "government policy",
        "পর্যটন স্থান"
    ]
    
    print("\n" + "="*80)
    print("CROSS-LINGUAL INFORMATION RETRIEVAL DEMO")
    print("="*80)
    
    for query in test_queries:
        # Search
        results = system.search(
            query=query,
            method='bm25',
            cross_lingual=True,
            expand_query=True,
            remove_stopwords=True,
            use_translation_api=False,
            top_k=5
        )
        
        # Print results
        system.print_results(results, show_body=False)
        
        input("Press Enter to continue to next query...")


if __name__ == "__main__":
    main()
