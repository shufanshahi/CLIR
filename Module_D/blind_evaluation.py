"""
Blind Evaluation & Inter-Model Comparison
Performs relative comparison between retrieval methods without requiring ground truth labels.
Calculates Overlap@K, Jaccard Similarity, and Performance metrics.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Any
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_C.retriever import CLIRRetriever
from Module_D.ranker import CLIRRanker

class BlindEvaluator:
    def __init__(self):
        print("Initializing components for blind evaluation...")
        self.retriever = CLIRRetriever()
        self.ranker = CLIRRanker(self.retriever)
        self.models = ['lexical', 'semantic', 'fuzzy', 'hybrid']
        
    def compare_models(self, queries: List[str], k: int = 10):
        """
        Compare all models against each other for a set of queries.
        """
        results = {model: [] for model in self.models}
        timings = {model: [] for model in self.models}
        per_query_stats = []
        
        print(f"\nRunning retrieval for {len(queries)} queries across {len(self.models)} models...")
        
        # 1. Collect Results
        for q_idx, query in enumerate(queries):
            print(f"Processing query {q_idx+1}/{len(queries)}: '{query}'")
            
            query_data = {
                'query': query,
                'metrics': {},
                'overlaps': {}
            }
            
            # Run for each model
            current_query_urls = {}
            
            for model in self.models:
                start_t = time.time()
                res = self.ranker.rank_query(query, k=k, model=model)
                elapsed = (time.time() - start_t) * 1000 # ms
                
                urls = set([r['url'] for r in res['results']])
                
                results[model].append(urls)
                timings[model].append(elapsed)
                current_query_urls[model] = urls
                
                query_data['metrics'][model] = {
                    'count': len(urls),
                    'time_ms': elapsed,
                    'top_score': res['results'][0]['confidence_score'] if res['results'] else 0.0
                }
            
            # Calculate overlap for this specific query
            for m1 in self.models:
                for m2 in self.models:
                    if m1 >= m2: continue
                    
                    set1 = current_query_urls[m1]
                    set2 = current_query_urls[m2]
                    
                    if not set1 and not set2:
                        jaccard = 1.0
                    elif not set1 or not set2:
                        jaccard = 0.0
                    else:
                        jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                    
                    query_data['overlaps'][f"{m1} vs {m2}"] = jaccard

            per_query_stats.append(query_data)
                
        # 2. Calculate Inter-Model Metrics
        print("\nCalculating inter-model agreement (Jaccard Similarity)...")
        # ... (rest of aggregate calculation remains similar or uses collected data)
        # Initialize comparison matrix
        agreement_matrix = pd.DataFrame(index=self.models, columns=self.models, dtype=float)
        
        for m1 in self.models:
            for m2 in self.models:
                if m1 == m2:
                    agreement_matrix.loc[m1, m2] = 1.0
                    continue
                
                overlaps = []
                for i in range(len(queries)):
                    set1 = results[m1][i]
                    set2 = results[m2][i]
                    if not set1 and not set2: jaccard = 1.0
                    elif not set1 or not set2: jaccard = 0.0
                    else: jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                    overlaps.append(jaccard)
                
                agreement_matrix.loc[m1, m2] = np.mean(overlaps)

        # 3. Calculate Performance Metrics
        print("\nCalculating performance metrics...")
        perf_stats = {}
        for model in self.models:
            perf_stats[model] = {
                'avg_time_ms': np.mean(timings[model]),
                'avg_res_count': np.mean([len(s) for s in results[model]])
            }
            
        return agreement_matrix, perf_stats, results, per_query_stats

    def print_report(self, agreement_matrix, perf_stats, k, per_query_stats=None):
        print("\n" + "="*80)
        print(f"BLIND EVALUATION REPORT (Top-{k} Results)")
        print("="*80)
        
        print("\n1. MODEL PERFORMANCE (Speed & Returns)")
        print("-" * 80)
        print(f"{'Model':<15} | {'Avg Time (ms)':<15} | {'Avg Doc Count':<15}")
        print("-" * 80)
        for model, stats in perf_stats.items():
            print(f"{model:<15} | {stats['avg_time_ms']:<15.2f} | {stats['avg_res_count']:<15.2f}")
            
        print("\n2. INTER-MODEL AGREEMENT (Avg Jaccard Similarity)")
        print("-" * 80)
        print(agreement_matrix.round(3))
        
        if per_query_stats:
            print("\n" + "="*80)
            print("DETAILED PER-QUERY METRICS")
            print("="*80)
            
            for item in per_query_stats:
                print(f"\nQuery: '{item['query']}'")
                print("-" * 60)
                print(f"{'Model':<10} | {'Results':<8} | {'Time (ms)':<10} | {'Top Conf':<10}")
                print("-" * 60)
                for model in self.models:
                    m = item['metrics'][model]
                    print(f"{model:<10} | {m['count']:<8} | {m['time_ms']:<10.2f} | {m['top_score']:<10.3f}")
                
                print("\n  Overlap Highlights:")
                # Print just the key comparisons to keep it readable
                key_pairs = ['lexical vs semantic', 'hybrid vs lexical', 'hybrid vs semantic']
                for pair in key_pairs:
                    score = item['overlaps'].get(pair, 0.0)
                    print(f"  * {pair:<20}: {score:.1%}")

    def generate_visualizations(self, agreement_matrix, perf_stats, per_query_stats, output_dir="Module_D/results/plots"):
        """Generate visualizations for blind evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Performance Comparison (Time)
        plt.figure(figsize=(10, 6))
        models = list(perf_stats.keys())
        times = [perf_stats[m]['avg_time_ms'] for m in models]
        sns.barplot(x=models, y=times)
        plt.title('Average Retrieval Time by Model')
        plt.ylabel('Time (ms)')
        plt.xlabel('Model')
        plt.savefig(f"{output_dir}/avg_time_{timestamp}.png")
        plt.close()
        print(f"Saved time comparison plot to {output_dir}/avg_time_{timestamp}.png")
        
        # 2. Inter-Model Agreement Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(agreement_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
        plt.title('Inter-Model Agreement (Jaccard Similarity)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/agreement_matrix_{timestamp}.png")
        plt.close()
        print(f"Saved agreement matrix plot to {output_dir}/agreement_matrix_{timestamp}.png")

        # 3. Hybrid Overlap Analysis per Query
        # Truncate queries for better visualization
        queries_short = []
        for item in per_query_stats:
            q = item['query']
            # if mixed script, take first few words
            if len(q) > 20: 
                queries_short.append(q[:20] + "...")
            else:
                queries_short.append(q)
                
        hybrid_sem = [item['overlaps'].get('hybrid vs semantic', 0) for item in per_query_stats]
        hybrid_lex = [item['overlaps'].get('hybrid vs lexical', 0) for item in per_query_stats]
        
        df_overlap = pd.DataFrame({
            'Query': queries_short,
            'Hybrid vs Semantic': hybrid_sem,
            'Hybrid vs Lexical': hybrid_lex
        })
        
        df_melted = df_overlap.melt('Query', var_name='Comparison', value_name='Jaccard Index')
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_melted, x='Jaccard Index', y='Query', hue='Comparison')
        plt.title('Hybrid Model Bias: Semantic vs Lexical Overlap per Query')
        plt.xlim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hybrid_bias_{timestamp}.png")
        plt.close()
        print(f"Saved hybrid bias plot to {output_dir}/hybrid_bias_{timestamp}.png")

if __name__ == "__main__":
    evaluator = BlindEvaluator()
    
    # 15+ Queries (Bangla and English)
    queries = [
        # English Queries
        "election results 2024",
        "politics in Bangladesh",
        "education system reform", 
        "latest technology trends",
        "Dhaka Metro Rail",
        "inflation and economy",
        "cricket world cup",
        "climate change impact",
        
        # Bangla Queries
        "দ্বাদশ জাতীয় সংসদ নির্বাচন", # 12th National Parliament Election
        "বাংলাদেশের রাজনীতি", # Politics of Bangladesh
        "শিক্ষা ব্যবস্থা", # Education system
        "শেয়ার বাজার পরিস্থিতি", # Share market situation
        "পদ্মা সেতু", # Padma Bridge
        "দ্রব্যমূল্যের ঊর্ধ্বগতি", # Commodity price hike
        "রোহিঙ্গা সংকট", # Rohingya crisis
        "বাংলাদেশের আবহাওয়া", # Weather of Bangladesh
        "সাইবার নিরাপত্তা" # Cyber security
    ]
    
    # Run comparison with Top-10 results
    matrix, stats, raw_results, query_stats = evaluator.compare_models(queries, k=10)
    evaluator.print_report(matrix, stats, k=10, per_query_stats=query_stats)
    
    # Generate visualizations
    evaluator.generate_visualizations(matrix, stats, query_stats)
    evaluator.generate_visualizations(matrix, stats, query_stats)
