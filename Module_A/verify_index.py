"""
Verification script for CLIR indexing system.
Tests various aspects of the index to ensure it's working correctly.
"""

import json
import os
from collections import Counter
from indexer import DocumentIndexer

def verify_index_structure(indexer):
    """Verify the basic structure of the index."""
    print("=" * 60)
    print("1. VERIFYING INDEX STRUCTURE")
    print("=" * 60)
    
    # Check total documents
    assert indexer.total_documents > 0, "No documents indexed!"
    print(f"✓ Total documents: {indexer.total_documents}")
    
    # Check metadata storage
    assert len(indexer.document_metadata) == indexer.total_documents, \
        "Metadata count doesn't match document count!"
    print(f"✓ Document metadata entries: {len(indexer.document_metadata)}")
    
    # Check inverted index
    assert len(indexer.inverted_index) > 0, "Inverted index is empty!"
    print(f"✓ Unique terms in index: {len(indexer.inverted_index)}")
    
    # Check document frequency
    assert len(indexer.document_frequency) == len(indexer.inverted_index), \
        "Document frequency count doesn't match inverted index!"
    print(f"✓ Document frequency entries: {len(indexer.document_frequency)}")
    
    # Verify required metadata fields
    sample_doc_id = list(indexer.document_metadata.keys())[0]
    sample_doc = indexer.document_metadata[sample_doc_id]
    required_fields = ['title', 'body', 'url', 'date', 'language', 'tokens', 'doc_length']
    for field in required_fields:
        assert field in sample_doc, f"Missing required field: {field}"
    print(f"✓ All required metadata fields present")
    
    print("✓ Index structure is valid!\n")
    return True


def verify_document_samples(indexer):
    """Verify sample documents from each language."""
    print("=" * 60)
    print("2. VERIFYING DOCUMENT SAMPLES")
    print("=" * 60)
    
    # Get sample documents by language
    bangla_docs = [doc_id for doc_id, meta in indexer.document_metadata.items() 
                   if meta['language'] == 'bn']
    english_docs = [doc_id for doc_id, meta in indexer.document_metadata.items() 
                    if meta['language'] == 'en']
    
    print(f"✓ Bangla documents: {len(bangla_docs)}")
    print(f"✓ English documents: {len(english_docs)}")
    
    # Check a Bangla document
    if bangla_docs:
        bn_doc_id = bangla_docs[0]
        bn_doc = indexer.document_metadata[bn_doc_id]
        print(f"\nSample Bangla Document (ID: {bn_doc_id}):")
        print(f"  Title: {bn_doc['title'][:80]}...")
        print(f"  Language: {bn_doc['language']}")
        print(f"  Tokens: {bn_doc['tokens']}")
        print(f"  URL: {bn_doc['url']}")
        assert bn_doc['language'] == 'bn', "Language mismatch!"
        assert bn_doc['tokens'] > 0, "Document has no tokens!"
    
    # Check an English document
    if english_docs:
        en_doc_id = english_docs[0]
        en_doc = indexer.document_metadata[en_doc_id]
        print(f"\nSample English Document (ID: {en_doc_id}):")
        print(f"  Title: {en_doc['title'][:80]}...")
        print(f"  Language: {en_doc['language']}")
        print(f"  Tokens: {en_doc['tokens']}")
        print(f"  URL: {en_doc['url']}")
        assert en_doc['language'] == 'en', "Language mismatch!"
        assert en_doc['tokens'] > 0, "Document has no tokens!"
    
    print("\n✓ Document samples verified!\n")
    return True


def verify_term_lookups(indexer):
    """Verify term lookups in the inverted index."""
    print("=" * 60)
    print("3. VERIFYING TERM LOOKUPS")
    print("=" * 60)
    
    # Get some sample terms from the index
    sample_terms = list(indexer.inverted_index.keys())[:10]
    print(f"Testing {len(sample_terms)} sample terms...")
    
    for term in sample_terms[:5]:  # Test first 5 terms
        doc_ids = list(indexer.inverted_index[term].keys())
        df = indexer.get_document_frequency(term)
        
        print(f"\nTerm: '{term}'")
        print(f"  Document Frequency (DF): {df}")
        print(f"  Appears in documents: {len(doc_ids)}")
        assert df == len(doc_ids), "DF doesn't match actual document count!"
        
        # Check term frequency in first document
        if doc_ids:
            first_doc_id = doc_ids[0]
            tf = indexer.get_term_frequency(term, first_doc_id)
            positions = len(indexer.inverted_index[term][first_doc_id])
            print(f"  TF in doc {first_doc_id}: {tf} (positions: {positions})")
            assert tf == positions, "TF doesn't match position count!"
    
    print("\n✓ Term lookups working correctly!\n")
    return True


def verify_tf_idf_calculations(indexer):
    """Verify TF-IDF calculations."""
    print("=" * 60)
    print("4. VERIFYING TF-IDF CALCULATIONS")
    print("=" * 60)
    
    # Get a sample term
    sample_term = list(indexer.inverted_index.keys())[50]  # Pick a middle term
    doc_ids = list(indexer.inverted_index[sample_term].keys())
    
    if not doc_ids:
        print("No documents found for test term, skipping TF-IDF test")
        return True
    
    test_doc_id = doc_ids[0]
    
    # Calculate TF
    tf = indexer.get_term_frequency(sample_term, test_doc_id)
    print(f"Test term: '{sample_term}'")
    print(f"Test document: {test_doc_id}")
    print(f"Term Frequency (TF): {tf}")
    
    # Calculate DF
    df = indexer.get_document_frequency(sample_term)
    print(f"Document Frequency (DF): {df}")
    
    # Calculate IDF
    idf = indexer.get_inverse_document_frequency(sample_term)
    print(f"Inverse Document Frequency (IDF): {idf:.4f}")
    
    # Calculate TF-IDF
    tf_idf = tf * idf
    print(f"TF-IDF: {tf_idf:.4f}")
    
    # Verify IDF calculation: log(N/df)
    import math
    expected_idf = math.log(indexer.total_documents / df) if df > 0 else 0
    assert abs(idf - expected_idf) < 0.0001, f"IDF calculation error: {idf} != {expected_idf}"
    
    print("\n✓ TF-IDF calculations are correct!\n")
    return True


def verify_language_distribution(indexer):
    """Verify language distribution."""
    print("=" * 60)
    print("5. VERIFYING LANGUAGE DISTRIBUTION")
    print("=" * 60)
    
    lang_counts = Counter(meta['language'] for meta in indexer.document_metadata.values())
    
    print("Language distribution:")
    for lang, count in lang_counts.items():
        lang_name = "Bangla" if lang == "bn" else "English"
        percentage = (count / indexer.total_documents) * 100
        print(f"  {lang_name} ({lang}): {count} documents ({percentage:.1f}%)")
    
    # Verify we have both languages
    assert 'bn' in lang_counts, "No Bangla documents found!"
    assert 'en' in lang_counts, "No English documents found!"
    
    print("\n✓ Language distribution verified!\n")
    return True


def verify_query_simulation(indexer):
    """Simulate a simple query to verify retrieval capability."""
    print("=" * 60)
    print("6. SIMULATING QUERY RETRIEVAL")
    print("=" * 60)
    
    # Test English query
    test_query_en = "bangladesh election"
    print(f"Test English query: '{test_query_en}'")
    
    query_terms = indexer.tokenize(test_query_en, 'en', remove_stopwords=False)
    print(f"Query terms: {query_terms}")
    
    # Find documents containing these terms
    matching_docs = set()
    for term in query_terms:
        if term in indexer.inverted_index:
            doc_ids = set(indexer.inverted_index[term].keys())
            matching_docs.update(doc_ids)
    
    print(f"Documents matching query: {len(matching_docs)}")
    
    # Show top matching documents
    doc_scores = {}
    for doc_id in list(matching_docs)[:5]:
        score = 0
        doc_meta = indexer.document_metadata[doc_id]
        for term in query_terms:
            tf = indexer.get_term_frequency(term, doc_id)
            idf = indexer.get_inverse_document_frequency(term)
            score += tf * idf
        doc_scores[doc_id] = score
    
    # Sort by score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop matching documents (simple TF-IDF):")
    for i, (doc_id, score) in enumerate(sorted_docs[:3], 1):
        doc_meta = indexer.document_metadata[doc_id]
        print(f"\n  {i}. Doc ID: {doc_id} (Score: {score:.2f})")
        print(f"     Title: {doc_meta['title'][:70]}...")
        print(f"     Language: {doc_meta['language']}")
        print(f"     URL: {doc_meta['url']}")
    
    # Test Bangla query
    test_query_bn = "বাংলাদেশ"
    print(f"\n\nTest Bangla query: '{test_query_bn}'")
    
    query_terms_bn = indexer.tokenize(test_query_bn, 'bn', remove_stopwords=False)
    print(f"Query terms: {query_terms_bn}")
    
    matching_docs_bn = set()
    for term in query_terms_bn:
        if term in indexer.inverted_index:
            doc_ids = set(indexer.inverted_index[term].keys())
            matching_docs_bn.update(doc_ids)
    
    print(f"Documents matching query: {len(matching_docs_bn)}")
    
    if matching_docs_bn:
        doc_scores_bn = {}
        for doc_id in list(matching_docs_bn)[:5]:
            score = 0
            for term in query_terms_bn:
                tf = indexer.get_term_frequency(term, doc_id)
                idf = indexer.get_inverse_document_frequency(term)
                score += tf * idf
            doc_scores_bn[doc_id] = score
        
        sorted_docs_bn = sorted(doc_scores_bn.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop matching documents:")
        for i, (doc_id, score) in enumerate(sorted_docs_bn[:3], 1):
            doc_meta = indexer.document_metadata[doc_id]
            print(f"\n  {i}. Doc ID: {doc_id} (Score: {score:.2f})")
            print(f"     Title: {doc_meta['title'][:70]}...")
            print(f"     Language: {doc_meta['language']}")
    
    print("\n✓ Query simulation successful!\n")
    return True


def verify_index_persistence(indexer):
    """Verify that index can be saved and loaded correctly."""
    print("=" * 60)
    print("7. VERIFYING INDEX PERSISTENCE")
    print("=" * 60)
    
    # Get current stats
    original_doc_count = indexer.total_documents
    original_term_count = len(indexer.inverted_index)
    
    # Save index
    test_dir = "Module_A/test_index"
    indexer.save_index(test_dir)
    
    # Create new indexer and load
    new_indexer = DocumentIndexer()
    new_indexer.load_index(test_dir)
    
    # Verify loaded data
    assert new_indexer.total_documents == original_doc_count, \
        "Document count mismatch after loading!"
    assert len(new_indexer.inverted_index) == original_term_count, \
        "Term count mismatch after loading!"
    
    print(f"✓ Saved index with {original_doc_count} documents")
    print(f"✓ Loaded index successfully")
    print(f"✓ Document count matches: {new_indexer.total_documents}")
    print(f"✓ Term count matches: {len(new_indexer.inverted_index)}")
    
    # Verify sample document
    sample_doc_id = list(indexer.document_metadata.keys())[0]
    original_meta = indexer.document_metadata[sample_doc_id]
    loaded_meta = new_indexer.document_metadata[sample_doc_id]
    
    assert original_meta['title'] == loaded_meta['title'], "Metadata mismatch!"
    assert original_meta['url'] == loaded_meta['url'], "URL mismatch!"
    
    print(f"✓ Sample document metadata matches")
    
    # Clean up test directory
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"✓ Cleaned up test directory")
    
    print("\n✓ Index persistence verified!\n")
    return True


def verify_statistics(indexer):
    """Verify index statistics are reasonable."""
    print("=" * 60)
    print("8. VERIFYING INDEX STATISTICS")
    print("=" * 60)
    
    # Check document lengths
    doc_lengths = [meta['doc_length'] for meta in indexer.document_metadata.values()]
    avg_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    min_length = min(doc_lengths) if doc_lengths else 0
    max_length = max(doc_lengths) if doc_lengths else 0
    
    print(f"Document length statistics:")
    print(f"  Average: {avg_length:.2f} tokens")
    print(f"  Minimum: {min_length} tokens")
    print(f"  Maximum: {max_length} tokens")
    
    # Verify average length is reasonable (not too small, not too large)
    assert 50 < avg_length < 2000, f"Average document length seems unreasonable: {avg_length}"
    
    # Check token counts match doc_length
    for doc_id, meta in list(indexer.document_metadata.items())[:10]:
        assert meta['tokens'] == meta['doc_length'], \
            f"Token count mismatch for doc {doc_id}"
    
    print(f"\n✓ Token counts match doc_length")
    
    # Check term frequencies
    sample_term = list(indexer.inverted_index.keys())[100]
    doc_ids_with_term = list(indexer.inverted_index[sample_term].keys())
    
    print(f"\nTerm frequency sample (term: '{sample_term}'):")
    print(f"  Appears in {len(doc_ids_with_term)} documents")
    
    # Check DF values are reasonable
    df_values = list(indexer.document_frequency.values())
    avg_df = sum(df_values) / len(df_values) if df_values else 0
    print(f"  Average DF: {avg_df:.2f}")
    
    print("\n✓ Statistics are reasonable!\n")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("CLIR INDEX VERIFICATION")
    print("=" * 60 + "\n")
    
    # Load the index
    print("Loading index...")
    indexer = DocumentIndexer()
    indexer.load_index("Module_A/indexed_data")
    print(f"✓ Index loaded with {indexer.total_documents} documents\n")
    
    # Run all verification tests
    tests = [
        verify_index_structure,
        verify_document_samples,
        verify_term_lookups,
        verify_tf_idf_calculations,
        verify_language_distribution,
        verify_query_simulation,
        verify_index_persistence,
        verify_statistics,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func(indexer)
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR in {test_func.__name__}: {e}\n")
            failed += 1
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All verification tests passed!")
        print("✓ Index is ready for CLIR retrieval tasks!")
    else:
        print(f"\n✗ {failed} test(s) failed. Please review the errors above.")
    
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

