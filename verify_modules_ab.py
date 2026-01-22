"""
Integration Verification for Module A (Indexing) and Module B (Query Processing).
Tests the complete flow: Query -> QueryProcessor -> Index Lookup -> Document Retrieval.
"""

import sys
import os
import time

# Ensure modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Module_A.indexer import DocumentIndexer
from Module_B.query_processor import QueryProcessor

def test_integration(query_list=None):
    print("=" * 60)
    print("CLIR Module A & B Integration Test")
    print("=" * 60)

    # 1. Load Index
    print("\n[Step 1] Loading Index (Module A)...")
    indexer = DocumentIndexer()
    try:
        indexer.load_index("Module_A/indexed_data")
        print(f"✓ Index loaded: {indexer.total_documents} documents")
        print(f"✓ Terms in index: {len(indexer.inverted_index)}")
    except Exception as e:
        print(f"✗ Failed to load index: {e}")
        return

    # 2. Initialize Query Processor
    print("\n[Step 2] Initializing Query Processor (Module B)...")
    try:
        processor = QueryProcessor()
        print("✓ Query Processor initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Query Processor: {e}")
        return

    # 3. Test Queries
    if not query_list:
        query_list = [
            "election commission",
            "নির্বাচন কমিশন",
            "digital bangladesh",
            "ডিজিটাল বাংলাদেশ",
            "climate change",
            "জলবায়ু পরিবর্তন"
        ]

    print("\n[Step 3] Running Test Queries...")
    
    for query in query_list:
        print(f"\n{'-'*60}")
        print(f"Query: '{query}'")
        print(f"{'-'*60}")
        
        # Process Query
        start_time = time.time()
        processed_result = processor.process(query, expand=True, map_nes=True)
        print(f"Detailed Processing Results:")
        print(f"  • Detected Language: {processed_result['detected_language']}")
        print(f"  • Normalized: {processed_result['normalized_query']}")
        print(f"  • Target Queries: {processed_result['target_queries']}")
        print(f"  • Expanded Terms: {processed_result['expanded_terms'][:5]}...") # Show top 5
        print(f"  • Named Entities: {processed_result['named_entities']}")
        
        # Prepare terms for lookup
        # We look up terms from: target queries (both langs) + expanded terms
        search_terms = set()
        
        # Add terms from target queries
        for lang, target_q in processed_result['target_queries'].items():
            # We need to tokenize these target queries to look them up
            tokens = indexer.tokenize(target_q, lang, remove_stopwords=True)
            search_terms.update(tokens)
            
        # Add expanded terms (assuming they are already single words/terms)
        for term in processed_result['expanded_terms']:
            search_terms.add(term)
            
        print(f"  • Search Terms (Total {len(search_terms)}): {list(search_terms)[:10]}...")

        # Look up in Index
        print("\n[Index Lookup]")
        matching_docs = {}  # doc_id -> score (simple TF count for now)
        
        for term in search_terms:
            if term in indexer.inverted_index:
                doc_ids = indexer.inverted_index[term].keys()
                df = indexer.get_document_frequency(term)
                # print(f"    - '{term}': found in {df} docs")
                
                for doc_id in doc_ids:
                    if doc_id not in matching_docs:
                        matching_docs[doc_id] = 0
                    matching_docs[doc_id] += 1  # Simple coordinate matching score
        
        print(f"  • Found {len(matching_docs)} unique documents containing at least one term.")
        
        # Show Top Results
        if matching_docs:
            print("\n[Top Retrieved Documents]")
            # Sort by score (number of matching terms)
            sorted_docs = sorted(matching_docs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                meta = indexer.document_metadata[doc_id]
                print(f"  {rank}. [DocID: {doc_id}] {meta['title'][:60]}...")
                print(f"     Language: {meta['language']}")
                print(f"     URL: {meta['url']}")
                print(f"     Matched Terms: {score}")
        else:
            print("  ⚠ No documents found.")

if __name__ == "__main__":
    test_integration()
