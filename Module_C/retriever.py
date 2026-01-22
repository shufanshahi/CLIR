"""
Retrieval Module for CLIR System
Implements Lexical (BM25), Semantic (Embedding-based), and Hybrid retrieval models.
"""

import sys
import os
import json
import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_A.indexer import DocumentIndexer
from Module_B.query_processor import QueryProcessor

class CLIRRetriever:
    """
    Main Retriever Class implementing multiple retrieval strategies.
    """
    
    def __init__(self, index_dir="Module_A/indexed_data", embedding_dir="Module_C/data"):
        print("Initializing CLIR Retriever...")
        
        # 1. Load Module A Index
        self.indexer = DocumentIndexer()
        self.indexer.load_index(index_dir)
        self.avg_doc_length = self._get_avg_doc_length(index_dir)
        
        # 2. Load Module B Query Processor
        self.query_processor = QueryProcessor()
        
        # 3. Load Module C Embeddings
        self.embedding_dir = embedding_dir
        self.embeddings = None
        self.doc_id_mapping = None
        self.model = None
        
        self._load_embeddings()
        
    def _get_avg_doc_length(self, index_dir):
        """Get average document length from stats file."""
        try:
            with open(os.path.join(index_dir, 'index_stats.json'), 'r') as f:
                stats = json.load(f)
                return stats.get('average_doc_length', 0)
        except:
            # Fallback calculation
            total_tokens = sum(m['doc_length'] for m in self.indexer.document_metadata.values())
            total_docs = len(self.indexer.document_metadata)
            return total_tokens / total_docs if total_docs > 0 else 0

    def _load_embeddings(self):
        """Load pre-computed document embeddings and mapping."""
        try:
            emb_path = os.path.join(self.embedding_dir, "doc_embeddings.pt")
            map_path = os.path.join(self.embedding_dir, "doc_id_mapping.json")
            
            if os.path.exists(emb_path) and os.path.exists(map_path):
                print(f"Loading embeddings from {emb_path}...")
                self.embeddings = torch.load(emb_path)
                
                with open(map_path, 'r') as f:
                    self.doc_id_mapping = json.load(f) # List of doc_ids corresponding to rows
                    
                # Load embedding model for query encoding
                from sentence_transformers import SentenceTransformer
                # Same model used in generation
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self.model = SentenceTransformer(model_name)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
                
                # Move embeddings to the same device
                self.embeddings = self.embeddings.to(device)
                print(f"✓ Embeddings loaded to {device}")
                print("✓ Semantic retrieval ready.")
            else:
                print("⚠ Warning: Embeddings not found. Semantic search will be disabled.")
                
        except Exception as e:
            print(f"⚠ Error loading embeddings: {e}")

    def search_lexical(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform Lexical Retrieval using BM25.
        1. Process query (translate/expand)
        2. Score documents using BM25 formula
        """
        start_time = time.time()
        
        # Process query (cross-lingual)
        # We need terms in both languages to match documents in both languages
        processed = self.query_processor.process(query, expand=True, map_nes=True)
        
        # Gather all search terms (original + translated + expanded)
        search_terms = set()
        
        # Add terms from target queries
        for lang, target_q in processed['target_queries'].items():
            tokens = self.indexer.tokenize(target_q, lang, remove_stopwords=True)
            search_terms.update(tokens)
            
        # Add expanded terms
        search_terms.update(processed['expanded_terms'])
        
        # BM25 Constants
        k1 = 1.5
        b = 0.75
        N = self.indexer.total_documents
        avgdl = self.avg_doc_length
        
        scores = defaultdict(float)
        
        for term in search_terms:
            if term not in self.indexer.inverted_index:
                continue
                
            doc_posting = self.indexer.inverted_index[term] # {doc_id: [positions]}
            
            # Calculate IDF
            df = len(doc_posting)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            
            for doc_id, positions in doc_posting.items():
                # Term Frequency in Doc
                tf = len(positions)
                
                # Document Length
                doc_len = self.indexer.document_metadata[doc_id]['doc_length']
                
                # BM25 Component
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                
                scores[doc_id] += idf * (numerator / denominator)
                
        # Sort and take top K
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_docs:
            meta = self.indexer.document_metadata[doc_id]
            results.append({
                'doc_id': doc_id,
                'score': score,
                'title': meta['title'],
                'url': meta['url'],
                'language': meta['language'],
                'model': 'lexical_bm25'
            })
            
        return results

    def search_tfidf(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform Lexical Retrieval using standard TF-IDF.
        Score = TF * IDF
        """
        # Process query
        processed = self.query_processor.process(query, expand=True, map_nes=True)
        search_terms = set()
        for lang, target_q in processed['target_queries'].items():
            tokens = self.indexer.tokenize(target_q, lang, remove_stopwords=True)
            search_terms.update(tokens)
        search_terms.update(processed['expanded_terms'])
        
        N = self.indexer.total_documents
        scores = defaultdict(float)
        
        for term in search_terms:
            if term not in self.indexer.inverted_index:
                continue
            
            doc_posting = self.indexer.inverted_index[term]
            df = len(doc_posting)
            # Standard IDF: log(N / df)
            idf = math.log(N / (df + 1)) + 1 
            
            for doc_id, positions in doc_posting.items():
                # TF: Term Frequency (Raw count)
                tf = len(positions)
                
                # We can normalize TF (e.g., log normalization) if we want, 
                # but standard TF-IDF often uses raw or log(1+tf)
                tf_norm = math.log(1 + tf)
                
                scores[doc_id] += tf_norm * idf
                
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_docs:
            meta = self.indexer.document_metadata[doc_id]
            results.append({
                'doc_id': doc_id,
                'score': score,
                'title': meta['title'],
                'url': meta['url'],
                'language': meta['language'],
                'model': 'lexical_tfidf'
            })
            
        return results

    def search_semantic(self, query: str, k: int = 10) -> List[Dict]:
        """
        Perform Semantic Retrieval using Cosine Similarity.
        1. Encode query
        2. Compute cosine similarity with all doc embeddings
        3. Return top K
        """
        if self.embeddings is None or self.model is None:
            return []
            
        start_time = time.time()
        
        # Translate query if needed (optional, but finding english query to encode might be better if model is english-centric,
        # but LaBSE/multilingual models handle direct bangla well. We'll encode the raw query + translated for better coverage?
        # For simplicity, we encode the input query directly as the model is multilingual.)
        
        # Encode Query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute Cosine Similarity
        # Normalize vectors for dot product -> cosine similarity
        
        # Ensure query is normalized
        import torch.nn.functional as F
        query_embedding = F.normalize(query_embedding, p=2, dim=0)
        
        # Ensure doc embeddings normalized (should be done during generation or here)
        # Doing it here just in case
        doc_embeddings = F.normalize(self.embeddings, p=2, dim=1)
        
        # Similarity scores
        # shape: (num_docs,)
        cos_scores = torch.matmul(doc_embeddings, query_embedding)
        
        # Top K
        # values, indices
        top_results = torch.topk(cos_scores, k=k)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            idx = idx.item()
            score = score.item()
            
            # Map index back to Doc ID
            doc_id = self.doc_id_mapping[idx]
            
            meta = self.indexer.document_metadata[doc_id]
            results.append({
                'doc_id': doc_id,
                'score': score, # Cosine similarity usually -1 to 1
                'title': meta['title'],
                'url': meta['url'],
                'language': meta['language'],
                'model': 'semantic'
            })
            
        return results

    def search_hybrid(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Dict]:
        """
        Perform Hybrid Retrieval (Lexical + Semantic).
        Score = alpha * Normalized_Lexical + (1 - alpha) * Normalized_Semantic
        """
        # Get more results to ensure overlap for reranking
        fetch_k = k * 2
        
        lex_results = self.search_lexical(query, k=fetch_k)
        sem_results = self.search_semantic(query, k=fetch_k)
        
        # Normalize scores to [0, 1] range
        
        # Lexical normalization (MinMax or just Max division)
        if lex_results:
            max_lex = max(r['score'] for r in lex_results)
            if max_lex > 0:
                for r in lex_results:
                    r['norm_score'] = r['score'] / max_lex
            else:
                for r in lex_results: r['norm_score'] = 0
        
        # Semantic normalization (Cosine is -1 to 1, usually 0 to 1 for text)
        if sem_results:
            # Semantic scores are already roughly 0-1, but let's ensure
            for r in sem_results:
                r['norm_score'] = max(0.0, r['score']) # Clip negative
        
        # Combine
        combined_scores = defaultdict(float)
        all_metas = {}
        
        for r in lex_results:
            combined_scores[r['doc_id']] += alpha * r.get('norm_score', 0)
            all_metas[r['doc_id']] = r
            
        for r in sem_results:
            combined_scores[r['doc_id']] += (1 - alpha) * r.get('norm_score', 0)
            if r['doc_id'] not in all_metas:
                all_metas[r['doc_id']] = r
                
        # Sort
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_docs:
            meta = self.indexer.document_metadata[doc_id]
            results.append({
                'doc_id': doc_id,
                'score': score,
                'title': meta['title'],
                'url': meta['url'],
                'language': meta['language'],
                'model': 'hybrid'
            })
            
        return results

    def search_fuzzy(self, query: str, k: int = 10) -> List[Dict]:
        """
        Fuzzy matching on Titles coverage.
        Includes transliteration/translation matching by using Module B's translation.
        Matches 'Bangladesh' with 'বাংলাদেশ' via translation layer.
        """
        try:
            from rapidfuzz import process, fuzz
            
            # 1. Process query to get translations (e.g., 'Bangladesh' -> 'বাংলাদেশ')
            #    We use the query processor to get the target query in the other language.
            processed = self.query_processor.process(query, expand=False, map_nes=False)
            
            # Collect all query variants (original + translated)
            query_variants = set()
            query_variants.add(query)
            for lang, q_trans in processed.get('target_queries', {}).items():
                query_variants.add(q_trans)
            
            # remove empty strings
            query_variants = {q for q in query_variants if q and q.strip()}
            
            all_matches = []
            
            # Extract all titles once
            titles = {doc_id: m['title'] for doc_id, m in self.indexer.document_metadata.items()}
            
            # 2. Run Fuzzy Match for EACH variant
            for q_var in query_variants:
                # Limit result per variant
                # process.extract returns list of (title, score, doc_id) in newer versions 
                # or (title, score, key) depending on version. We assume dict-based extract.
                matches = process.extract(q_var, titles, scorer=fuzz.token_sort_ratio, limit=k)
                for match in matches:
                    # Unpack carefully as rapidfuzz API varies slightly by version
                    # Usually: (match_string, score, key)
                    if len(match) == 3:
                        title, score, doc_id = match
                    else:
                        continue # Skip unexpected format
                        
                    norm_score = score / 100.0
                    all_matches.append((doc_id, norm_score))
            
            # 3. Deduplicate and Sort
            # If a doc is found by multiple variants, take max score
            unique_scores = defaultdict(float)
            for doc_id, score in all_matches:
                unique_scores[doc_id] = max(unique_scores[doc_id], score)
                
            sorted_docs = sorted(unique_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            
            results = []
            for doc_id, score in sorted_docs:
                meta = self.indexer.document_metadata[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'score': score,
                    'title': meta['title'],
                    'url': meta['url'],
                    'language': meta['language'],
                    'model': 'fuzzy'
                })
            return results
            
        except ImportError:
            # Fallback: Simple substring search
            print("⚠ rapidfuzz not installed. Using simple substring match.")
            # ... existing fallback ...
            results = []
            q_lower = query.lower()
            for doc_id, meta in self.indexer.document_metadata.items():
                if q_lower in meta['title'].lower():
                    results.append({
                        'doc_id': doc_id,
                        'score': 1.0,
                        'title': meta['title'],
                        'url': meta['url'],
                        'language': meta['language'],
                        'model': 'fuzzy_simple'
                    })
            return results[:k]

if __name__ == "__main__":
    # Simple test
    retriever = CLIRRetriever()
    
    q = "election"
    print(f"\nQuery: {q}")
    
    print("\n[Lexical Search]")
    for r in retriever.search_lexical(q, k=3):
        print(f"  {r['score']:.4f} | {r['title'][:50]} ({r['language']})")
        
    print("\n[Semantic Search]")
    for r in retriever.search_semantic(q, k=3):
        print(f"  {r['score']:.4f} | {r['title'][:50]} ({r['language']})")

    print("\n[Hybrid Search]")
    for r in retriever.search_hybrid(q, k=3):
        print(f"  {r['score']:.4f} | {r['title'][:50]} ({r['language']})")
