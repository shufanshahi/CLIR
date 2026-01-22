"""
Error Analysis Framework for CLIR System
Analyzes retrieval failures across different categories with specific examples.
"""

import os
import json
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict, Counter
import re
from datetime import datetime

class ErrorAnalyzer:
    """
    Class for analyzing and categorizing retrieval failures.
    """
    
    def __init__(self, output_dir: str = "Module_D/data/error_analysis"):
        """
        Initialize the error analyzer.
        
        Args:
            output_dir: Directory to save error analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Error categories from assignment
        self.error_categories = {
            'translation_failure': {
                'description': 'Query translation errors leading to wrong retrieval',
                'examples': []
            },
            'named_entity_mismatch': {
                'description': 'Named entities not matching across languages',
                'examples': []
            },
            'semantic_vs_lexical': {
                'description': 'Cases where semantic succeeds but lexical fails (or vice versa)',
                'examples': []
            },
            'cross_script_ambiguity': {
                'description': 'Ambiguity in transliteration across scripts',
                'examples': []
            },
            'code_switching': {
                'description': 'Queries mixing Bangla and English words',
                'examples': []
            }
        }
    
    def analyze_translation_failures(self, query: str, original_results: List[Dict], 
                                    translated_results: List[Dict], 
                                    expected_docs: Set[str]) -> List[Dict]:
        """
        Analyze translation-related failures.
        
        Args:
            query: Original query
            original_results: Results before translation
            translated_results: Results after translation
            expected_docs: Set of expected relevant document URLs
            
        Returns:
            List of translation failure examples
        """
        failures = []
        
        # Check if translation made results worse
        original_relevant = len([r for r in original_results if r['url'] in expected_docs])
        translated_relevant = len([r for r in translated_results if r['url'] in expected_docs])
        
        if translated_relevant < original_relevant:
            failures.append({
                'query': query,
                'type': 'translation_degradation',
                'original_relevant': original_relevant,
                'translated_relevant': translated_relevant,
                'original_top3': [r['title'][:50] for r in original_results[:3]],
                'translated_top3': [r['title'][:50] for r in translated_results[:3]],
                'analysis': f"Translation reduced relevant results from {original_relevant} to {translated_relevant}"
            })
        
        # Check for obvious mistranslations
        if self._is_obvious_mistranslation(query):
            failures.append({
                'query': query,
                'type': 'obvious_mistranslation',
                'analysis': f"Query '{query}' appears to be mistranslated or contains errors"
            })
        
        return failures
    
    def analyze_named_entity_mismatches(self, query: str, results: List[Dict], 
                                      expected_docs: Set[str]) -> List[Dict]:
        """
        Analyze named entity matching failures.
        
        Args:
            query: Query string
            results: Retrieval results
            expected_docs: Expected relevant documents
            
        Returns:
            List of named entity mismatch examples
        """
        failures = []
        
        # Extract potential named entities from query
        entities = self._extract_named_entities(query)
        
        if entities:
            # Check if documents containing these entities were missed
            for entity in entities:
                entity_in_expected = any(entity.lower() in doc.lower() for doc in expected_docs)
                entity_in_results = any(entity.lower() in r['title'].lower() or 
                                       entity.lower() in r.get('body', '').lower() 
                                       for r in results)
                
                if entity_in_expected and not entity_in_results:
                    failures.append({
                        'query': query,
                        'entity': entity,
                        'type': 'named_entity_missed',
                        'analysis': f"Named entity '{entity}' from query not found in results"
                    })
        
        return failures
    
    def analyze_semantic_vs_lexical(self, query: str, lexical_results: List[Dict], 
                                  semantic_results: List[Dict], 
                                  expected_docs: Set[str]) -> List[Dict]:
        """
        Analyze cases where semantic and lexical models differ significantly.
        
        Args:
            query: Query string
            lexical_results: Results from lexical model
            semantic_results: Results from semantic model
            expected_docs: Expected relevant documents
            
        Returns:
            List of semantic vs lexical examples
        """
        failures = []
        
        # Count relevant results for each model
        lexical_relevant = len([r for r in lexical_results if r['url'] in expected_docs])
        semantic_relevant = len([r for r in semantic_results if r['url'] in expected_docs])
        
        # Find cases where one model significantly outperforms the other
        if semantic_relevant > lexical_relevant * 2:  # Semantic much better
            failures.append({
                'query': query,
                'type': 'semantic_win',
                'lexical_relevant': lexical_relevant,
                'semantic_relevant': semantic_relevant,
                'lexical_top3': [r['title'][:50] for r in lexical_results[:3]],
                'semantic_top3': [r['title'][:50] for r in semantic_results[:3]],
                'analysis': f"Semantic model found {semantic_relevant} relevant docs vs {lexical_relevant} for lexical"
            })
        
        elif lexical_relevant > semantic_relevant * 2:  # Lexical much better
            failures.append({
                'query': query,
                'type': 'lexical_win',
                'lexical_relevant': lexical_relevant,
                'semantic_relevant': semantic_relevant,
                'lexical_top3': [r['title'][:50] for r in lexical_results[:3]],
                'semantic_top3': [r['title'][:50] for r in semantic_results[:3]],
                'analysis': f"Lexical model found {lexical_relevant} relevant docs vs {semantic_relevant} for semantic"
            })
        
        return failures
    
    def analyze_cross_script_ambiguity(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Analyze cross-script transliteration ambiguities.
        
        Args:
            query: Query string
            results: Retrieval results
            
        Returns:
            List of cross-script ambiguity examples
        """
        failures = []
        
        # Check for mixed script queries
        if self._has_mixed_script(query):
            failures.append({
                'query': query,
                'type': 'mixed_script_query',
                'analysis': f"Query contains mixed scripts: {query}"
            })
        
        # Check for common transliteration variations
        transliteration_variations = self._get_transliteration_variations(query)
        if len(transliteration_variations) > 1:
            # Check if different variations would give different results
            failures.append({
                'query': query,
                'type': 'transliteration_ambiguity',
                'variations': transliteration_variations,
                'analysis': f"Query has multiple transliteration variations: {transliteration_variations}"
            })
        
        return failures
    
    def analyze_code_switching(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Analyze code-switching in queries.
        
        Args:
            query: Query string
            results: Retrieval results
            
        Returns:
            List of code-switching examples
        """
        failures = []
        
        if self._is_code_switched(query):
            failures.append({
                'query': query,
                'type': 'code_switching',
                'languages_detected': self._detect_languages(query),
                'analysis': f"Query mixes multiple languages: {query}"
            })
        
        return failures
    
    def _extract_named_entities(self, text: str) -> List[str]:
        """Extract potential named entities from text."""
        # Simple heuristic: capitalized words and proper nouns
        # In a real implementation, you'd use NER models
        entities = []
        
        # Look for patterns that might be named entities
        # This is a simplified version - real NER would be more sophisticated
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(words)
        
        # Look for Bangla proper nouns (simplified)
        bangla_words = re.findall(r'[\u0980-\u09FF]+', text)
        entities.extend(bangla_words)
        
        return list(set(entities))
    
    def _is_obvious_mistranslation(self, query: str) -> bool:
        """Check for obvious mistranslation patterns."""
        # Simple heuristics for obvious mistranslations
        mistranslation_patterns = [
            r'চয়ার',  # Chair mistranslated to Chairman
            r'স্কুল.*education',  # Mixed terms that might indicate translation issues
        ]
        
        for pattern in mistranslation_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def _has_mixed_script(self, text: str) -> bool:
        """Check if text contains mixed scripts."""
        has_bangla = bool(re.search(r'[\u0980-\u09FF]', text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        return has_bangla and has_english
    
    def _get_transliteration_variations(self, text: str) -> List[str]:
        """Get common transliteration variations for text."""
        variations = [text]
        
        # Common transliteration patterns
        transliteration_map = {
            'Bangladesh': ['বাংলাদেশ', 'Bangla Desh'],
            'Dhaka': ['ঢাকা', 'Daka'],
            'Chittagong': ['চট্টগ্রাম', 'Chattogram'],
        }
        
        for english, bangla_variants in transliteration_map.items():
            if english.lower() in text.lower():
                for variant in bangla_variants:
                    variations.append(text.lower().replace(english.lower(), variant))
        
        return list(set(variations))
    
    def _is_code_switched(self, text: str) -> bool:
        """Check if text exhibits code-switching."""
        return self._has_mixed_script(text)
    
    def _detect_languages(self, text: str) -> List[str]:
        """Detect languages present in text."""
        languages = []
        if re.search(r'[\u0980-\u09FF]', text):
            languages.append('Bangla')
        if re.search(r'[a-zA-Z]', text):
            languages.append('English')
        return languages
    
    def analyze_query(self, query: str, lexical_results: List[Dict], 
                     semantic_results: List[Dict], expected_docs: Set[str]) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis for a single query.
        
        Args:
            query: Query string
            lexical_results: Results from lexical model
            semantic_results: Results from semantic model
            expected_docs: Expected relevant documents
            
        Returns:
            Dictionary with error analysis results
        """
        analysis = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'errors_found': defaultdict(list),
            'summary': {}
        }
        
        # Analyze different error categories
        translation_failures = self.analyze_translation_failures(
            query, lexical_results, semantic_results, expected_docs)
        analysis['errors_found']['translation_failure'].extend(translation_failures)
        
        ne_failures = self.analyze_named_entity_mismatches(
            query, lexical_results, expected_docs)
        analysis['errors_found']['named_entity_mismatch'].extend(ne_failures)
        
        semantic_failures = self.analyze_semantic_vs_lexical(
            query, lexical_results, semantic_results, expected_docs)
        analysis['errors_found']['semantic_vs_lexical'].extend(semantic_failures)
        
        cross_script_failures = self.analyze_cross_script_ambiguity(
            query, lexical_results)
        analysis['errors_found']['cross_script_ambiguity'].extend(cross_script_failures)
        
        code_switching_failures = self.analyze_code_switching(
            query, lexical_results)
        analysis['errors_found']['code_switching'].extend(code_switching_failures)
        
        # Create summary
        total_errors = sum(len(errors) for errors in analysis['errors_found'].values())
        analysis['summary'] = {
            'total_errors': total_errors,
            'errors_by_category': {cat: len(errors) for cat, errors in analysis['errors_found'].items()},
            'has_errors': total_errors > 0
        }
        
        return analysis
    
    def analyze_batch(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform error analysis on a batch of query results.
        
        Args:
            batch_results: List of query analysis results
            
        Returns:
            Comprehensive error analysis report
        """
        batch_analysis = {
            'total_queries': len(batch_results),
            'timestamp': datetime.now().isoformat(),
            'overall_summary': {},
            'category_summaries': {},
            'detailed_examples': defaultdict(list)
        }
        
        # Aggregate errors across all queries
        all_errors = defaultdict(list)
        total_errors = 0
        
        for result in batch_results:
            if 'errors_found' in result:
                for category, errors in result['errors_found'].items():
                    all_errors[category].extend(errors)
                    total_errors += len(errors)
        
        # Create category summaries
        for category, errors in all_errors.items():
            batch_analysis['category_summaries'][category] = {
                'count': len(errors),
                'description': self.error_categories.get(category, {}).get('description', ''),
                'percentage': (len(errors) / total_errors * 100) if total_errors > 0 else 0
            }
            
            # Add detailed examples (up to 3 per category)
            batch_analysis['detailed_examples'][category] = errors[:3]
        
        # Overall summary
        batch_analysis['overall_summary'] = {
            'total_errors': total_errors,
            'queries_with_errors': len([r for r in batch_results if r.get('summary', {}).get('has_errors', False)]),
            'most_common_error': max(all_errors.keys(), key=lambda k: len(all_errors[k])) if all_errors else None,
            'error_rate': len([r for r in batch_results if r.get('summary', {}).get('has_errors', False)]) / len(batch_results)
        }
        
        return batch_analysis
    
    def save_analysis(self, analysis: Dict[str, Any], filename: str = None) -> str:
        """
        Save error analysis to file.
        
        Args:
            analysis: Analysis results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_analysis_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            print(f"Error analysis saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return ""
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """
        Generate a human-readable error analysis report.
        
        Args:
            analysis: Error analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("# CLIR Error Analysis Report")
        report.append(f"Generated: {analysis.get('timestamp', 'Unknown')}")
        report.append("")
        
        if 'overall_summary' in analysis:
            # Batch analysis report
            summary = analysis['overall_summary']
            report.append("## Overall Summary")
            report.append(f"- Total Queries Analyzed: {analysis.get('total_queries', 0)}")
            report.append(f"- Total Errors Found: {summary.get('total_errors', 0)}")
            report.append(f"- Queries with Errors: {summary.get('queries_with_errors', 0)}")
            report.append(f"- Error Rate: {summary.get('error_rate', 0):.2%}")
            report.append(f"- Most Common Error: {summary.get('most_common_error', 'None')}")
            report.append("")
            
            # Category breakdown
            report.append("## Error Categories")
            for category, info in analysis.get('category_summaries', {}).items():
                report.append(f"### {category.replace('_', ' ').title()}")
                report.append(f"- Count: {info['count']}")
                report.append(f"- Percentage: {info['percentage']:.1f}%")
                report.append(f"- Description: {info['description']}")
                report.append("")
                
                # Add examples
                examples = analysis.get('detailed_examples', {}).get(category, [])
                if examples:
                    report.append("#### Examples:")
                    for i, example in enumerate(examples[:3], 1):
                        report.append(f"{i}. Query: '{example.get('query', 'Unknown')}'")
                        report.append(f"   Analysis: {example.get('analysis', 'No analysis')}")
                        report.append("")
        
        return "\n".join(report)

if __name__ == "__main__":
    # Test the error analyzer
    print("Testing Error Analyzer...")
    
    analyzer = ErrorAnalyzer()
    
    # Sample data for testing
    query = "চয়ারম্যান election"
    lexical_results = [
        {'url': 'doc1', 'title': 'Meeting Chairman', 'body': ''},
        {'url': 'doc2', 'title': 'Election Results', 'body': ''}
    ]
    semantic_results = [
        {'url': 'doc3', 'title': 'School Chair', 'body': ''},
        {'url': 'doc4', 'title': 'Election News', 'body': ''}
    ]
    expected_docs = {'doc1', 'doc4'}
    
    # Analyze single query
    analysis = analyzer.analyze_query(query, lexical_results, semantic_results, expected_docs)
    print(f"Single query analysis: {analysis['summary']}")
    
    # Test batch analysis
    batch_results = [analysis]
    batch_analysis = analyzer.analyze_batch(batch_results)
    print(f"Batch analysis: {batch_analysis['overall_summary']}")
    
    # Generate report
    report = analyzer.generate_report(batch_analysis)
    print(f"\nSample Report:\n{report[:500]}...")
