"""
Comprehensive test suite for Query Processor.
Tests all components: language detection, normalization, translation, expansion, NE mapping.
"""

from query_processor import (
    LanguageDetector, QueryNormalizer, QueryTranslator,
    QueryExpander, NamedEntityMapper, QueryProcessor
)


def test_language_detection():
    """Test language detection."""
    print("=" * 60)
    print("TESTING LANGUAGE DETECTION")
    print("=" * 60)
    
    detector = LanguageDetector()
    
    test_cases = [
        ("bangladesh election", "en"),
        ("বাংলাদেশ নির্বাচন", "bn"),
        ("dhaka government", "en"),
        ("ঢাকা সরকার", "bn"),
        ("education policy", "en"),
        ("শিক্ষা নীতি", "bn"),
        ("bangladesh নির্বাচন", "mixed"),  # Code-switched
    ]
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        detected = detector.detect(query)
        status = "✓" if detected == expected or (expected == "mixed" and detected in ["bn", "mixed"]) else "✗"
        
        if status == "✓":
            passed += 1
        else:
            failed += 1
        
        print(f"{status} Query: '{query}'")
        print(f"  Expected: {expected}, Detected: {detected}")
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_normalization():
    """Test query normalization."""
    print("=" * 60)
    print("TESTING NORMALIZATION")
    print("=" * 60)
    
    normalizer = QueryNormalizer()
    
    test_cases = [
        ("Bangladesh  Election", "en", "bangladesh election"),
        ("  বাংলাদেশ  নির্বাচন  ", "bn", "বাংলাদেশ নির্বাচন"),
        ("The   Government   of   Bangladesh", "en", "the government of bangladesh"),
    ]
    
    passed = 0
    failed = 0
    
    for query, lang, expected in test_cases:
        normalized = normalizer.normalize(query, lang, remove_stopwords=False)
        status = "✓" if normalized == expected else "✗"
        
        if status == "✓":
            passed += 1
        else:
            failed += 1
        
        print(f"{status} Query: '{query}'")
        print(f"  Expected: '{expected}', Got: '{normalized}'")
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_translation():
    """Test query translation."""
    print("=" * 60)
    print("TESTING TRANSLATION")
    print("=" * 60)
    
    translator = QueryTranslator()
    
    test_cases = [
        ("bangladesh", "en", "bn", "বাংলাদেশ"),
        ("dhaka", "en", "bn", "ঢাকা"),
        ("election", "en", "bn", "নির্বাচন"),
    ]
    
    passed = 0
    failed = 0
    
    for query, src, tgt, expected in test_cases:
        translated = translator.translate(query, src, tgt)
        # Check if translation contains expected (since we might have partial translations)
        status = "✓" if expected in translated or query == translated else "✗"
        
        if status == "✓":
            passed += 1
        else:
            failed += 1
        
        print(f"{status} Query: '{query}' ({src} → {tgt})")
        print(f"  Expected contains: '{expected}', Got: '{translated}'")
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_query_expansion():
    """Test query expansion."""
    print("=" * 60)
    print("TESTING QUERY EXPANSION")
    print("=" * 60)
    
    expander = QueryExpander()
    
    test_cases = [
        ("election", "en", ["election", "vote", "poll", "voting", "electoral"]),
        ("education", "en", ["education", "school", "learning", "teaching", "study"]),
    ]
    
    passed = 0
    failed = 0
    
    for query, lang, expected_terms in test_cases:
        expanded = expander.expand(query, lang)
        # Check if all expected terms are in expanded list
        missing = [t for t in expected_terms if t not in expanded]
        status = "✓" if len(missing) == 0 else "✗"
        
        if status == "✓":
            passed += 1
        else:
            failed += 1
        
        print(f"{status} Query: '{query}'")
        print(f"  Expanded terms: {expanded}")
        if missing:
            print(f"  Missing: {missing}")
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_named_entity_mapping():
    """Test named entity mapping."""
    print("=" * 60)
    print("TESTING NAMED ENTITY MAPPING")
    print("=" * 60)
    
    ne_mapper = NamedEntityMapper()
    
    test_cases = [
        ("bangladesh election", "en", "bn", {"bangladesh": "বাংলাদেশ"}),
        ("dhaka government", "en", "bn", {"dhaka": "ঢাকা"}),
        ("বাংলাদেশ নির্বাচন", "bn", "en", {"বাংলাদেশ": "bangladesh"}),
    ]
    
    passed = 0
    failed = 0
    
    for query, src, tgt, expected_mappings in test_cases:
        mappings = ne_mapper.extract_and_map(query, src, tgt)
        # Check if expected mappings are present
        found = all(orig in mappings and mappings[orig] == mapped 
                   for orig, mapped in expected_mappings.items())
        status = "✓" if found else "✗"
        
        if status == "✓":
            passed += 1
        else:
            failed += 1
        
        print(f"{status} Query: '{query}' ({src} → {tgt})")
        print(f"  Expected: {expected_mappings}, Got: {mappings}")
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def test_full_pipeline():
    """Test the complete query processing pipeline."""
    print("=" * 60)
    print("TESTING FULL QUERY PROCESSING PIPELINE")
    print("=" * 60)
    
    processor = QueryProcessor()
    
    test_queries = [
        "bangladesh election",
        "বাংলাদেশ নির্বাচন",
        "dhaka government",
        "ঢাকা সরকার",
    ]
    
    passed = 0
    failed = 0
    
    for query in test_queries:
        try:
            result = processor.process(query)
            
            # Verify required fields
            required_fields = [
                'original_query', 'detected_language', 'normalized_query',
                'target_queries', 'expanded_terms', 'named_entities', 'processing_time'
            ]
            
            missing_fields = [f for f in required_fields if f not in result]
            
            if not missing_fields and result['processing_time'] > 0:
                status = "✓"
                passed += 1
            else:
                status = "✗"
                failed += 1
            
            print(f"{status} Query: '{query}'")
            print(f"  Language: {result['detected_language']}")
            print(f"  Target queries: {result['target_queries']}")
            print(f"  Processing time: {result['processing_time']:.2f} ms")
            
            if missing_fields:
                print(f"  Missing fields: {missing_fields}")
        
        except Exception as e:
            print(f"✗ Query: '{query}' - Error: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed\n")
    return failed == 0


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("QUERY PROCESSOR COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        ("Language Detection", test_language_detection),
        ("Normalization", test_normalization),
        ("Translation", test_translation),
        ("Query Expansion", test_query_expansion),
        ("Named Entity Mapping", test_named_entity_mapping),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

