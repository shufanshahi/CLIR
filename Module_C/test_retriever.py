"""
Verification Script for Module C: Retrieval Models.
Tests Lexical, Semantic, Fuzzy, and Hybrid retrieval for Bangla and English queries.
"""

import sys
import os

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_C.retriever import CLIRRetriever

def verify_retrieval_models():
    print("=" * 70)
    print("VERIFYING CLIR RETRIEVAL MODELS (MODULE C)")
    print("=" * 70)
    
    # Initialize Retriever
    try:
        retriever = CLIRRetriever()
        print("✓ Retriever initialized successfully.")
    except Exception as e:
        print(f"✗ Failed to initialize retriever: {e}")
        return

    # User defined test queries
    queries = [
        # English queries
        "bangladesh election", 
        "climate change impact",
        
        # Bangla queries
        "বাংলাদেশ নির্বাচন",
        "জলবায়ু পরিবর্তন",
        
        # Typos (for fuzzy)
        "bangldesh elction" 
    ]
    
    for query in queries:
        print(f"\n{'-'*70}")
        print(f"QUERY: '{query}'")
        print(f"{'-'*70}")
        
        # 1. Lexical Search
        print("\n1. LEXICAL SEARCH (BM25)")
        results_lex = retriever.search_lexical(query, k=3)
        for i, r in enumerate(results_lex, 1):
            print(f"   {i}. [{r['score']:.4f}] {r['title'][:50]}... ({r['language']})")
            print(f"       URL: {r['url']}")
        if not results_lex: print("   No results found.")
            
        # 2. Semantic Search
        print("\n2. SEMANTIC SEARCH (Embeddings)")
        results_sem = retriever.search_semantic(query, k=3)
        for i, r in enumerate(results_sem, 1):
            print(f"   {i}. [{r['score']:.4f}] {r['title'][:50]}... ({r['language']})")
            print(f"       URL: {r['url']}")
        if not results_sem: print("   No results found.")
            
        # 3. Fuzzy Search
        print("\n3. FUZZY SEARCH (Title Matching)")
        results_fuz = retriever.search_fuzzy(query, k=3)
        for i, r in enumerate(results_fuz, 1):
            print(f"   {i}. [{r['score']:.4f}] {r['title'][:50]}... ({r['language']})")
            print(f"       URL: {r['url']}")
        if not results_fuz: print("   No results found.")
            
        # 4. Hybrid Search
        print("\n4. HYBRID SEARCH (Lexical + Semantic)")
        results_hyb = retriever.search_hybrid(query, k=3, alpha=0.4)
        for i, r in enumerate(results_hyb, 1):
            print(f"   {i}. [{r['score']:.4f}] {r['title'][:50]}... ({r['language']})")
            print(f"       URL: {r['url']}")
        if not results_hyb: print("   No results found.")

if __name__ == "__main__":
    verify_retrieval_models()
