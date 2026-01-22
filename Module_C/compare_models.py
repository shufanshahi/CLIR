"""
Comparison and Analysis Script for Module C.
addressing specific assignment requirements:
1. Compare BM25 vs TF-IDF.
2. Analyze failure cases (Synonyms, Paraphrases, Cross-script).
3. Demonstrate Fuzzy/Transliteration matching.
"""

import sys
import os
import time

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_C.retriever import CLIRRetriever

def compare_models():
    print("=" * 80)
    print("CLIR MODEL COMPARISON & ANALYSIS")
    print("=" * 80)
    
    try:
        retriever = CLIRRetriever()
        print("✓ Retriever initialized.\n")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return

    # ---------------------------------------------------------
    # 1. BM25 vs TF-IDF Comparison
    # ---------------------------------------------------------
    print("🔹 TEST 1: BM25 vs TF-IDF COMPARISON")
    print("   Goal: See if BM25 provides better ranking (length normalization)")
    
    query1 = "climate change"
    print(f"\n   Query: '{query1}'")
    
    res_bm25 = retriever.search_lexical(query1, k=3)
    res_tfidf = retriever.search_tfidf(query1, k=3)
    
    print("   [BM25 Results]")
    for r in res_bm25:
        print(f"   - {r['title'][:60]}... (Score: {r['score']:.2f})")
        
    print("   [TF-IDF Results]")
    for r in res_tfidf:
        print(f"   - {r['title'][:60]}... (Score: {r['score']:.2f})")
        
    print("\n   Analysis: BM25 usually penalizes very long documents that spam specific terms,")
    print("   whereas raw TF-IDF might favor longer documents with higher raw counts.")

    # ---------------------------------------------------------
    # 2. Semantic vs Lexical (Synonym/Paraphrase Analysis)
    # ---------------------------------------------------------
    print("\n🔹 TEST 2: SYNONYM & PARAPHRASE ANALYSIS")
    print("   Goal: Show where Lexical fails and Semantic succeeds.")
    
    # Query with words NOT in the text likely, but conceptually same
    # "Global Warming" vs "Climate Change" (Module B might expand, but let's try a tricky one)
    # "Vote casting" -> "Election" / "Polling"
    query2 = "vote casting procedure" 
    
    print(f"\n   Query: '{query2}'")
    
    res_lex = retriever.search_lexical(query2, k=3)
    res_sem = retriever.search_semantic(query2, k=3)
    
    print("   [Lexical (BM25)]")
    if not res_lex: print("   - No results found (Failure Case: Terms not in index)")
    for r in res_lex:
        print(f"   - {r['title'][:60]}... (Score: {r['score']:.2f})")
        
    print("   [Semantic (Embedding)]")
    for r in res_sem:
        print(f"   - {r['title'][:60]}... (Score: {r['score']:.2f})")
        
    print("\n   Analysis: Lexical models fail when exact keywords (or expansions)")
    print("   are missing. Semantic models capture 'meaning' and retrieve relevant docs.")

    # ---------------------------------------------------------
    # 3. Cross-Language / Translation Analysis
    # ---------------------------------------------------------
    print("\n🔹 TEST 3: CROSS-LANGUAGE RETRIEVAL")
    print("   Goal: Verify Bangla query finding English documents.")
    
    query3 = "রোহিঙ্গা সংকট" # Rohingya Crisis
    print(f"\n   Query: '{query3}'")
    
    res_lex = retriever.search_lexical(query3, k=3)
    
    print("   [Lexical Results (via NLLB Translation)]")
    for r in res_lex:
        print(f"   - [{r['language']}] {r['title'][:60]}...")
        
    # ---------------------------------------------------------
    # 4. Fuzzy / Transliteration Analysis
    # ---------------------------------------------------------
    print("\n🔹 TEST 4: FUZZY / TYPO MATCHING")
    print("   Goal: Handle typos or transliterated names.")
    
    queries_typo = ["bangldesh", "sheikh hasina"]
    
    for q in queries_typo:
        print(f"\n   Query: '{q}'")
        res_fuz = retriever.search_fuzzy(q, k=2)
        print("   [Fuzzy Results]")
        for r in res_fuz:
            print(f"   - {r['title'][:60]}... (Score: {r['score']:.2f})")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    compare_models()
