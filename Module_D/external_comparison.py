"""
External Search Engine Comparison for CLIR System
Compares CLIR results with Google, Bing, DuckDuckGo, and AI-powered search engines.
"""

import requests
import time
import json
import os
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import logging
from datetime import datetime
from collections import defaultdict

class ExternalSearchComparator:
    """
    Class for comparing CLIR results with external search engines.
    """
    
    def __init__(self, output_dir: str = "Module_D/data/external_comparison"):
        """
        Initialize the external search comparator.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # User-Agent to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.request_delay = 2.0  # seconds between requests
        
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo (no API key required).
        
        Args:
            query: Search query
            num_results: Number of results to retrieve
            
        Returns:
            List of search results
        """
        try:
            # DuckDuckGo instant answer API
            url = "https://duckduckgo.com/html/"
            params = {
                'q': query,
                'kl': 'us-en'  # Language setting
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse results
            for result in soup.find_all('div', class_='result')[:num_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__a')
                url_elem = result.find('a', class_='result__a')
                
                if title_elem and url_elem:
                    title = title_elem.get_text(strip=True)
                    url = url_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'duckduckgo',
                        'rank': len(results) + 1
                    })
            
            time.sleep(self.request_delay)
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching DuckDuckGo for '{query}': {e}")
            return []
    
    def search_bing_api(self, query: str, api_key: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using Bing Web Search API.
        
        Args:
            query: Search query
            api_key: Bing API key
            num_results: Number of results to retrieve
            
        Returns:
            List of search results
        """
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            params = {
                'q': query,
                'count': num_results,
                'mkt': 'en-US',
                'safesearch': 'Moderate'
            }
            
            headers = {
                'Ocp-Apim-Subscription-Key': api_key,
                **self.headers
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'webPages' in data and 'value' in data['webPages']:
                for item in data['webPages']['value']:
                    results.append({
                        'title': item.get('name', ''),
                        'url': item.get('url', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'bing',
                        'rank': len(results) + 1
                    })
            
            time.sleep(self.request_delay)
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching Bing API for '{query}': {e}")
            return []
    
    def search_google_custom_search(self, query: str, api_key: str, 
                                   search_engine_id: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search using Google Custom Search API.
        
        Args:
            query: Search query
            api_key: Google API key
            search_engine_id: Custom Search Engine ID
            num_results: Number of results to retrieve
            
        Returns:
            List of search results
        """
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': search_engine_id,
                'q': query,
                'num': min(num_results, 10)  # Google API limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'items' in data:
                for item in data['items']:
                    results.append({
                        'title': item.get('title', ''),
                        'url': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'google',
                        'rank': len(results) + 1
                    })
            
            time.sleep(self.request_delay)
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching Google API for '{query}': {e}")
            return []
    
    def simulate_ai_search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform AI-powered search using real APIs.
        Uses Perplexity API for AI-powered search results.
        """
        try:
            # Try Perplexity API (requires API key)
            perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
            if perplexity_api_key:
                return self._search_perplexity(query, perplexity_api_key, num_results)
            
            # Fallback: Use Brave Search API for comprehensive results
            brave_api_key = os.getenv('BRAVE_API_KEY')
            if brave_api_key:
                return self._search_brave(query, brave_api_key, num_results)
            
            # If no API keys available, return empty list
            self.logger.warning("No AI search API keys available. Please set PERPLEXITY_API_KEY or BRAVE_API_KEY environment variables.")
            return []
            
        except Exception as e:
            self.logger.error(f"Error in AI search for '{query}': {e}")
            return []
    
    def _search_perplexity(self, query: str, api_key: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search using Perplexity API."""
        try:
            url = "https://api.perplexity.ai/search"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': 'llama-3.1-sonar-small-128k-online',
                'query': query,
                'max_results': num_results,
                'search_mode': 'web'
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=15)
            response.raise_for_status()
            
            result_data = response.json()
            results = []
            
            if 'results' in result_data:
                for i, item in enumerate(result_data['results'][:num_results]):
                    results.append({
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'perplexity',
                        'rank': i + 1
                    })
            
            time.sleep(self.request_delay)
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching Perplexity for '{query}': {e}")
            return []
    
    def _search_brave(self, query: str, api_key: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search using Brave Search API."""
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': api_key
            }
            params = {
                'q': query,
                'count': num_results,
                'text_decorations': 'false',
                'search_lang': 'en',
                'ui_lang': 'en'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'web' in data and 'results' in data['web']:
                for i, item in enumerate(data['web']['results'][:num_results]):
                    results.append({
                        'title': item.get('title', ''),
                        'url': item.get('url', ''),
                        'snippet': item.get('description', ''),
                        'source': 'brave',
                        'rank': i + 1
                    })
            
            time.sleep(self.request_delay)
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching Brave for '{query}': {e}")
            return []
    
    def compare_with_external(self, query: str, clir_results: List[Dict], 
                            external_configs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare CLIR results with external search engines.
        
        Args:
            query: Search query
            clir_results: Results from CLIR system
            external_configs: Configuration for external APIs
            
        Returns:
            Comparison results
        """
        if external_configs is None:
            external_configs = {}
        
        comparison = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'clir_results': clir_results,
            'external_results': {},
            'overlap_analysis': {},
            'summary': {}
        }
        
        # Get external search results
        external_engines = ['duckduckgo', 'ai_search']
        
        if 'bing_api_key' in external_configs:
            external_engines.append('bing')
        
        if 'google_api_key' in external_configs and 'google_search_engine_id' in external_configs:
            external_engines.append('google')
        
        for engine in external_engines:
            try:
                if engine == 'duckduckgo':
                    results = self.search_duckduckgo(query)
                elif engine == 'bing':
                    results = self.search_bing_api(query, external_configs['bing_api_key'])
                elif engine == 'google':
                    results = self.search_google_custom_search(
                        query, external_configs['google_api_key'], external_configs['google_search_engine_id'])
                elif engine == 'ai_search':
                    results = self.simulate_ai_search(query)
                else:
                    continue
                
                comparison['external_results'][engine] = results
                
            except Exception as e:
                self.logger.error(f"Error getting results from {engine}: {e}")
                comparison['external_results'][engine] = []
        
        # Analyze overlap between CLIR and external results
        clir_urls = {r['url'] for r in clir_results}
        
        for engine, external_results in comparison['external_results'].items():
            external_urls = {r['url'] for r in external_results}
            
            overlap = clir_urls.intersection(external_urls)
            overlap_analysis = {
                'total_clir': len(clir_urls),
                'total_external': len(external_urls),
                'overlap_count': len(overlap),
                'overlap_percentage': len(overlap) / len(clir_urls) * 100 if clir_urls else 0,
                'overlapping_urls': list(overlap)
            }
            
            comparison['overlap_analysis'][engine] = overlap_analysis
        
        # Generate summary
        total_external_engines = len([r for r in comparison['external_results'].values() if r])
        avg_overlap = 0
        
        if comparison['overlap_analysis']:
            overlaps = [analysis['overlap_percentage'] for analysis in comparison['overlap_analysis'].values()]
            avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        
        comparison['summary'] = {
            'total_external_engines': total_external_engines,
            'average_overlap_percentage': avg_overlap,
            'clir_unique_results': len(clir_urls),
            'has_external_data': total_external_engines > 0
        }
        
        return comparison
    
    def batch_compare(self, queries: List[str], clir_results_dict: Dict[str, List[Dict]], 
                     external_configs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Compare multiple queries with external search engines.
        
        Args:
            queries: List of queries
            clir_results_dict: Dictionary mapping query to CLIR results
            external_configs: Configuration for external APIs
            
        Returns:
            Batch comparison results
        """
        batch_comparison = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(queries),
            'individual_comparisons': [],
            'aggregate_analysis': {}
        }
        
        print(f"Comparing {len(queries)} queries with external search engines...")
        
        for i, query in enumerate(queries):
            print(f"Progress: {i+1}/{len(queries)} - Comparing: '{query}'")
            
            clir_results = clir_results_dict.get(query, [])
            comparison = self.compare_with_external(query, clir_results, external_configs)
            batch_comparison['individual_comparisons'].append(comparison)
        
        # Aggregate analysis
        all_overlaps = []
        engine_stats = defaultdict(list)
        
        for comp in batch_comparison['individual_comparisons']:
            for engine, analysis in comp['overlap_analysis'].items():
                engine_stats[engine].append(analysis['overlap_percentage'])
                all_overlaps.append(analysis['overlap_percentage'])
        
        batch_comparison['aggregate_analysis'] = {
            'average_overlap_all': sum(all_overlaps) / len(all_overlaps) if all_overlaps else 0,
            'engine_performance': {
                engine: {
                    'avg_overlap': sum(overlaps) / len(overlaps) if overlaps else 0,
                    'min_overlap': min(overlaps) if overlaps else 0,
                    'max_overlap': max(overlaps) if overlaps else 0,
                    'num_queries': len(overlaps)
                }
                for engine, overlaps in engine_stats.items()
            }
        }
        
        return batch_comparison
    
    def save_comparison(self, comparison: Dict[str, Any], filename: str = None) -> str:
        """
        Save comparison results to file.
        
        Args:
            comparison: Comparison results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"external_comparison_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            
            print(f"External comparison saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving comparison: {e}")
            return ""
    
    def generate_comparison_report(self, comparison: Dict[str, Any]) -> str:
        """
        Generate a human-readable comparison report.
        
        Args:
            comparison: Comparison results
            
        Returns:
            Formatted report string
        """
        report_lines = []
        
        # Header
        report_lines.append("# External Search Engine Comparison Report")
        report_lines.append(f"Generated: {comparison.get('timestamp', 'Unknown')}")
        report_lines.append("")
        
        if 'individual_comparisons' in comparison:
            # Batch comparison report
            report_lines.append("## Batch Comparison Summary")
            aggregate = comparison.get('aggregate_analysis', {})
            
            report_lines.append(f"- **Total Queries Compared**: {comparison.get('total_queries', 0)}")
            report_lines.append(f"- **Average Overlap (All Engines)**: {aggregate.get('average_overlap_all', 0):.1f}%")
            report_lines.append("")
            
            # Engine performance
            if 'engine_performance' in aggregate:
                report_lines.append("### Performance by Engine")
                for engine, stats in aggregate['engine_performance'].items():
                    report_lines.append(f"#### {engine.title()}")
                    report_lines.append(f"- Average Overlap: {stats['avg_overlap']:.1f}%")
                    report_lines.append(f"- Range: {stats['min_overlap']:.1f}% - {stats['max_overlap']:.1f}%")
                    report_lines.append(f"- Queries Compared: {stats['num_queries']}")
                    report_lines.append("")
            
            # Individual query details
            report_lines.append("## Individual Query Comparisons")
            
            for i, comp in enumerate(comparison['individual_comparisons']):
                report_lines.append(f"### Query {i+1}: '{comp['query']}'")
                
                summary = comp.get('summary', {})
                report_lines.append(f"- CLIR Results: {summary.get('clir_unique_results', 0)}")
                report_lines.append(f"- External Engines: {summary.get('total_external_engines', 0)}")
                report_lines.append(f"- Average Overlap: {summary.get('average_overlap_percentage', 0):.1f}%")
                
                # Overlap details
                if comp.get('overlap_analysis'):
                    report_lines.append("#### Overlap Details:")
                    for engine, analysis in comp['overlap_analysis'].items():
                        report_lines.append(f"- **{engine.title()}**: {analysis['overlap_count']} URLs ({analysis['overlap_percentage']:.1f}%)")
                
                report_lines.append("")
        
        elif 'query' in comparison:
            # Single query comparison report
            report_lines.append("## Single Query Comparison")
            report_lines.append(f"**Query**: '{comparison['query']}'")
            report_lines.append("")
            
            summary = comparison.get('summary', {})
            report_lines.append("### Summary")
            report_lines.append(f"- CLIR Results: {summary.get('clir_unique_results', 0)}")
            report_lines.append(f"- External Engines: {summary.get('total_external_engines', 0)}")
            report_lines.append(f"- Average Overlap: {summary.get('average_overlap_percentage', 0):.1f}%")
            report_lines.append("")
            
            # Detailed overlap analysis
            if comparison.get('overlap_analysis'):
                report_lines.append("### Overlap Analysis")
                for engine, analysis in comparison['overlap_analysis'].items():
                    report_lines.append(f"#### {engine.title()}")
                    report_lines.append(f"- Total CLIR URLs: {analysis['total_clir']}")
                    report_lines.append(f"- Total External URLs: {analysis['total_external']}")
                    report_lines.append(f"- Overlapping URLs: {analysis['overlap_count']}")
                    report_lines.append(f"- Overlap Percentage: {analysis['overlap_percentage']:.1f}%")
                    
                    if analysis.get('overlapping_urls'):
                        report_lines.append("- Overlapping URLs:")
                        for url in analysis['overlapping_urls'][:3]:  # Show first 3
                            report_lines.append(f"  - {url}")
                    
                    report_lines.append("")
        
        return "\n".join(report_lines)

if __name__ == "__main__":
    # Test the external search comparator
    print("Testing External Search Comparator...")
    
    comparator = ExternalSearchComparator()
    
    # Sample data
    query = "election"
    clir_results = [
        {'url': 'https://example.com/election1', 'title': 'Election Results 2024'},
        {'url': 'https://example.com/election2', 'title': 'Election News'},
        {'url': 'https://example.com/election3', 'title': 'Voting Information'}
    ]
    
    # Test single comparison
    print("\nTesting single query comparison...")
    comparison = comparator.compare_with_external(query, clir_results)
    print(f"Comparison completed for: '{comparison['query']}'")
    print(f"Average overlap: {comparison['summary']['average_overlap_percentage']:.1f}%")
    
    # Test batch comparison
    print("\nTesting batch comparison...")
    queries = ["election", "education"]
    clir_results_dict = {
        "election": clir_results,
        "education": [
            {'url': 'https://example.com/edu1', 'title': 'Education System'},
            {'url': 'https://example.com/edu2', 'title': 'Learning Resources'}
        ]
    }
    
    batch_comparison = comparator.batch_compare(queries, clir_results_dict)
    print(f"Batch comparison completed for {len(queries)} queries")
    print(f"Overall average overlap: {batch_comparison['aggregate_analysis']['average_overlap_all']:.1f}%")
    
    # Generate report
    report = comparator.generate_comparison_report(batch_comparison)
    print(f"\nGenerated comparison report ({len(report)} characters)")
