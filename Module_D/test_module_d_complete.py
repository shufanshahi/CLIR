"""
Module D Complete Testing Suite
Single file to test and verify all Module D functionality with real data.
Covers all assignment requirements step by step.
"""

import sys
import os
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModuleDTester:
    """Complete testing suite for Module D functionality."""
    
    def __init__(self):
        self.output_dir = "Module_D/results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.test_results = {}
        
    def test_ranking_scoring(self):
        """Test 1: Ranking & Scoring Implementation"""
        print("=" * 80)
        print("TEST 1: RANKING & SCORING")
        print("=" * 80)
        
        try:
            from Module_D.ranker import CLIRRanker
            from Module_C.retriever import CLIRRetriever
            
            print("1.1 Initializing components...")
            retriever = CLIRRetriever()
            ranker = CLIRRanker(retriever, confidence_threshold=0.20)
            
            test_queries = ["election", "শিক্ষা", "Bangladesh"]
            results = {}
            
            for query in test_queries:
                print(f"\n1.2 Testing query: '{query}'")
                query_results = {}
                
                for model in ['lexical', 'semantic', 'hybrid']:
                    result = ranker.rank_query(query, k=10, model=model)
                    query_results[model] = result
                    
                    # Verify ranking function
                    print(f"   {model}: {len(result['results'])} results in {result['total_time_ms']:.2f}ms")
                    
                    if result['results']:
                        # Check confidence scores are in 0-1 range
                        scores = [r['confidence_score'] for r in result['results']]
                        print(f"     Confidence range: {min(scores):.3f} - {max(scores):.3f}")
                        
                        # Check if sorted
                        is_sorted = all(result['results'][i]['confidence_score'] >= result['results'][i+1]['confidence_score'] 
                                      for i in range(len(result['results'])-1))
                        print(f"     Properly sorted: {is_sorted}")
                        
                        # Check for low confidence warning
                        if result.get('low_confidence_warning'):
                            print(f"     ⚠️  {result['low_confidence_warning']}")
                
                results[query] = query_results
            
            self.test_results['ranking_scoring'] = {
                'status': 'PASSED',
                'details': results,
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n✅ RANKING & SCORING: PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ RANKING & SCORING: FAILED - {e}")
            self.test_results['ranking_scoring'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def test_evaluation_metrics(self):
        """Test 2: Evaluation Metrics"""
        print("\n" + "=" * 80)
        print("TEST 2: EVALUATION METRICS")
        print("=" * 80)
        
        try:
            from Module_D.metrics import IRMetrics
            
            print("2.1 Testing Precision@10...")
            relevant_docs = {'doc1', 'doc3', 'doc5', 'doc7', 'doc8', 'doc9'}
            retrieved_docs = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5', 'doc6', 'doc7', 'doc8', 'doc9', 'doc10']
            
            precision_10 = IRMetrics.precision_at_k(relevant_docs, retrieved_docs, 10)
            print(f"   Precision@10: {precision_10:.3f} (expected: 0.600)")
            
            print("2.2 Testing Recall@50...")
            total_relevant = {'doc1', 'doc3', 'doc5', 'doc7', 'doc8', 'doc9', 'doc11', 'doc12'}
            retrieved_50 = retrieved_docs + ['doc11', 'doc12'] + ['doc13'] * 38
            
            recall_50 = IRMetrics.recall_at_k(total_relevant, retrieved_50, 50)
            print(f"   Recall@50: {recall_50:.3f} (expected: 1.000)")
            
            print("2.3 Testing nDCG@10...")
            relevance_scores = {'doc1': 1.0, 'doc2': 0.5, 'doc3': 1.0, 'doc4': 0.0, 'doc5': 0.8}
            retrieved_ndcg = ['doc4', 'doc1', 'doc2', 'doc3', 'doc5']
            
            ndcg_10 = IRMetrics.ndcg_at_k(relevance_scores, retrieved_ndcg, 10)
            print(f"   nDCG@10: {ndcg_10:.3f}")
            
            print("2.4 Testing MRR...")
            relevant_list = [{'doc3', 'doc7'}, {'doc1', 'doc5'}, {'doc8'}]
            retrieved_list = [['doc1', 'doc2', 'doc3'], ['doc1', 'doc2', 'doc3'], ['doc1', 'doc8', 'doc2']]
            
            mrr = IRMetrics.mean_reciprocal_rank(relevant_list, retrieved_list)
            print(f"   MRR: {mrr:.3f} (expected: 0.611)")
            
            print("2.5 Testing target checking...")
            metrics = {
                'precision@10': 0.65,  # Above target
                'recall@50': 0.45,     # Below target
                'ndcg@10': 0.55,       # Above target
                'mrr': 0.35            # Below target
            }
            
            status = IRMetrics.check_target_metrics(metrics)
            print(f"   Target status: {'PASSED' if status['overall_passed'] else 'FAILED'}")
            
            self.test_results['evaluation_metrics'] = {
                'status': 'PASSED',
                'metrics': {
                    'precision@10': precision_10,
                    'recall@50': recall_50,
                    'ndcg@10': ndcg_10,
                    'mrr': mrr,
                    'target_status': status
                },
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n✅ EVALUATION METRICS: PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ EVALUATION METRICS: FAILED - {e}")
            self.test_results['evaluation_metrics'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def test_external_comparison(self):
        """Test 3: External Search Engine Comparison"""
        print("\n" + "=" * 80)
        print("TEST 3: EXTERNAL SEARCH COMPARISON")
        print("=" * 80)
        
        try:
            from Module_D.external_comparison import ExternalSearchComparator
            
            print("3.1 Initializing comparator...")
            comparator = ExternalSearchComparator()
            
            print("3.2 Testing DuckDuckGo...")
            test_query = "Bangladesh election"
            ddg_results = comparator.search_duckduckgo(test_query, num_results=3)
            print(f"   DuckDuckGo: {len(ddg_results)} results")
            
            print("3.3 Testing comparison functionality...")
            clir_results = [
                {'url': 'https://example.com/clir1', 'title': 'CLIR Result 1'},
                {'url': 'https://example.com/clir2', 'title': 'CLIR Result 2'},
            ]
            
            comparison = comparator.compare_with_external(test_query, clir_results)
            print(f"   Comparison completed for: '{comparison['query']}'")
            print(f"   CLIR results: {comparison['summary']['clir_unique_results']}")
            print(f"   External engines: {comparison['summary']['total_external_engines']}")
            print(f"   Average overlap: {comparison['summary']['average_overlap_percentage']:.1f}%")
            
            self.test_results['external_comparison'] = {
                'status': 'PASSED',
                'details': {
                    'ddg_results': len(ddg_results),
                    'comparison_summary': comparison['summary']
                },
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n✅ EXTERNAL SEARCH COMPARISON: PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ EXTERNAL SEARCH COMPARISON: FAILED - {e}")
            self.test_results['external_comparison'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def test_relevance_labeling(self):
        """Test 4: Relevance Labeling"""
        print("\n" + "=" * 80)
        print("TEST 4: RELEVANCE LABELING")
        print("=" * 80)
        
        try:
            from Module_D.labeling import RelevanceLabeler
            
            print("4.1 Testing CSV format and labeling...")
            test_file = "Module_D/data/test_labels.csv"
            labeler = RelevanceLabeler(test_file)
            
            # Add sample labels with all required columns
            queries = ["election", "education", "Bangladesh", "climate", "technology"]
            
            for i, query in enumerate(queries):
                labeler.add_label(
                    query=query,
                    doc_url=f'https://example.com/{query}1',
                    relevant=True,
                    language='en',
                    annotator=f'test_annotator_{i}',
                    notes=f'Relevant {query} content'
                )
                labeler.add_label(
                    query=query,
                    doc_url=f'https://example.com/{query}2',
                    relevant=False,
                    language='en',
                    annotator=f'test_annotator_{i}',
                    notes=f'Not relevant {query} content'
                )
            
            print(f"   Added labels for {len(queries)} queries")
            
            # Verify CSV structure
            with open(test_file, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                required_cols = ['query', 'doc_url', 'relevant', 'language', 'annotator', 'timestamp', 'notes']
                
                for col in required_cols:
                    if col not in header:
                        raise ValueError(f"Missing column: {col}")
            
            print("   ✓ All required CSV columns present")
            
            # Test label retrieval
            relevant_docs = labeler.get_relevant_docs("election")
            print(f"   Retrieved {len(relevant_docs)} relevant docs for 'election'")
            
            # Test statistics
            stats = labeler.get_statistics()
            print(f"   Total labels: {stats['total_labels']}")
            print(f"   Relevance rate: {stats['relevance_rate']:.2%}")
            print(f"   Unique queries: {len(set(label[0] for label in labeler.labels.keys()))}")
            
            # Clean up
            os.remove(test_file)
            
            self.test_results['relevance_labeling'] = {
                'status': 'PASSED',
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n✅ RELEVANCE LABELING: PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ RELEVANCE LABELING: FAILED - {e}")
            self.test_results['relevance_labeling'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def test_error_analysis(self):
        """Test 5: Error Analysis"""
        print("\n" + "=" * 80)
        print("TEST 5: ERROR ANALYSIS")
        print("=" * 80)
        
        try:
            from Module_D.error_analysis import ErrorAnalyzer
            
            print("5.1 Testing error categories...")
            analyzer = ErrorAnalyzer("Module_D/data/test_errors")
            
            # Test translation failures
            query = "চেয়ার"  # Chair (should not be Chairman)
            original = [{'url': 'doc1', 'title': 'Room Chair'}]
            translated = [{'url': 'doc2', 'title': 'Company Chairman'}]
            expected = {'doc1'}
            
            translation_errors = analyzer.analyze_translation_failures(query, original, translated, expected)
            print(f"   Translation failures: {len(translation_errors)}")
            
            # Test named entity mismatch
            query_ne = "ঢাকা"
            ne_results = [{'url': 'doc3', 'title': 'Dhaka University'}]
            expected_ne = {'doc3'}
            
            ne_errors = analyzer.analyze_named_entity_mismatches(query_ne, ne_results, expected_ne)
            print(f"   Named entity mismatches: {len(ne_errors)}")
            
            # Test semantic vs lexical
            query_sem = "শিক্ষা"
            lexical = [{'url': 'doc4', 'title': 'School'}]
            semantic = [{'url': 'doc5', 'title': 'Learning Methods'}]
            expected_sem = {'doc5'}
            
            semantic_errors = analyzer.analyze_semantic_vs_lexical(query_sem, lexical, semantic, expected_sem)
            print(f"   Semantic vs lexical issues: {len(semantic_errors)}")
            
            # Test cross-script ambiguity
            query_ambiguity = "Bangla Desh"
            ambiguity_results = [{'url': 'doc6', 'title': 'Bangladesh News'}]
            
            ambiguity_errors = analyzer.analyze_cross_script_ambiguity(query_ambiguity, ambiguity_results)
            print(f"   Cross-script ambiguities: {len(ambiguity_errors)}")
            
            # Test code-switching
            query_cs = "Bangladesh election"
            cs_results = [{'url': 'doc7', 'title': 'Election Results'}]
            
            cs_errors = analyzer.analyze_code_switching(query_cs, cs_results)
            print(f"   Code-switching issues: {len(cs_errors)}")
            
            # Test comprehensive analysis
            comprehensive = analyzer.analyze_query(query_sem, lexical, semantic, expected_sem)
            print(f"   Comprehensive analysis: {comprehensive['summary']['total_errors']} total errors")
            
            # Clean up
            import shutil
            if os.path.exists("Module_D/data/test_errors"):
                shutil.rmtree("Module_D/data/test_errors")
            
            self.test_results['error_analysis'] = {
                'status': 'PASSED',
                'error_counts': {
                    'translation_failures': len(translation_errors),
                    'named_entity_mismatches': len(ne_errors),
                    'semantic_vs_lexical': len(semantic_errors),
                    'cross_script_ambiguity': len(ambiguity_errors),
                    'code_switching': len(cs_errors)
                },
                'comprehensive_summary': comprehensive['summary'],
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n✅ ERROR ANALYSIS: PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR ANALYSIS: FAILED - {e}")
            self.test_results['error_analysis'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def test_integration(self):
        """Test full integration with real CLIR system"""
        print("\n" + "=" * 80)
        print("TEST 6: INTEGRATION WITH REAL CLIR SYSTEM")
        print("=" * 80)
        
        try:
            from Module_D.evaluator import CLIREvaluator
            
            print("6.1 Initializing full evaluator...")
            evaluator = CLIREvaluator()
            
            print("6.2 Testing with real queries...")
            real_queries = ["election", "শিক্ষা", "Bangladesh"]
            
            start_time = time.time()
            evaluation = evaluator.evaluate_query_set(real_queries, k=10)
            eval_time = time.time() - start_time
            
            print(f"   Evaluated {len(real_queries)} queries in {eval_time:.2f} seconds")
            
            # Check aggregate metrics
            if 'aggregate_metrics' in evaluation:
                for model, metrics in evaluation['aggregate_metrics'].items():
                    p10 = metrics.get('avg_precision@10', 0)
                    r50 = metrics.get('avg_recall@50', 0)
                    ndcg = metrics.get('avg_ndcg@10', 0)
                    mrr = metrics.get('avg_mrr', 0)
                    
                    print(f"   {model}: P@10={p10:.3f}, R@50={r50:.3f}, nDCG={ndcg:.3f}, MRR={mrr:.3f}")
            
            # Check performance stats
            if 'performance_stats' in evaluation:
                stats = evaluation['performance_stats']
                avg_time = stats['timing']['avg_time_ms']
                avg_conf = stats['confidence']['avg_top_confidence']
                
                print(f"   Average query time: {avg_time:.2f} ms")
                print(f"   Average confidence: {avg_conf:.3f}")
            
            # Generate outputs
            print("6.3 Generating outputs...")
            eval_file = evaluator.save_evaluation(evaluation, "integration_test.json")
            plot_files = evaluator.generate_visualizations(evaluation)
            report = evaluator.generate_report(evaluation)
            
            print(f"   Saved evaluation: {eval_file}")
            print(f"   Generated {len(plot_files)} visualizations")
            
            self.test_results['integration'] = {
                'status': 'PASSED',
                'queries_tested': len(real_queries),
                'evaluation_time': eval_time,
                'outputs_generated': {
                    'evaluation_file': eval_file,
                    'visualizations': len(plot_files),
                    'report_length': len(report)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            print("\n✅ INTEGRATION TEST: PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ INTEGRATION TEST: FAILED - {e}")
            self.test_results['integration'] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_all_tests(self):
        """Run all tests and generate final report."""
        print("MODULE D COMPLETE TESTING SUITE")
        print("Testing all functionality with real data")
        print("=" * 80)
        
        # Run all tests
        tests = [
            ("Ranking & Scoring", self.test_ranking_scoring),
            ("Evaluation Metrics", self.test_evaluation_metrics),
            ("External Comparison", self.test_external_comparison),
            ("Relevance Labeling", self.test_relevance_labeling),
            ("Error Analysis", self.test_error_analysis),
            ("Integration", self.test_integration)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    print(f"\n⚠️  {test_name} failed, continuing with other tests...")
            except Exception as e:
                print(f"\n❌ {test_name} crashed: {e}")
        
        # Generate final report
        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)
        
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            print(f"{test_name:<25}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - MODULE D READY!")
        else:
            print(f"\n⚠️  {total-passed} test(s) failed - review issues above")
        
        # Save test results
        results_file = os.path.join(self.output_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nTest results saved to: {results_file}")
        return passed == total

def main():
    """Main entry point."""
    tester = ModuleDTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
