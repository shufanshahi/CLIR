"""
Full Evaluation Script (Blind & Pseudo-Ground Truth)
Generates all required visualizations and metrics.
"""

import sys
import os
import shutil
from typing import List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_D.blind_evaluation import BlindEvaluator
from Module_D.evaluator import CLIREvaluator
from Module_D.labeling import RelevanceLabeler
from Module_D.ranker import CLIRRanker
from Module_C.retriever import CLIRRetriever

def run_blind_evaluation(queries: List[str]):
    print("\n" + "="*80)
    print("PHASE 1: BLIND EVALUATION (No Labels)")
    print("="*80)
    evaluator = BlindEvaluator()
    matrix, stats, raw_results, query_stats = evaluator.compare_models(queries, k=10)
    evaluator.print_report(matrix, stats, k=10, per_query_stats=query_stats)
    evaluator.generate_visualizations(matrix, stats, query_stats, output_dir="Module_D/results/plots/blind")

def run_ground_truth_evaluation(queries: List[str]):
    print("\n" + "="*80)
    print("PHASE 2: GROUND TRUTH EVALUATION (Pseudo-Relevance)")
    print("="*80)
    print("Generating pseudo-labels based on Hybrid Top-3 results (for demonstration)...")
    
    # 1. Generate Pseudo-Labels
    pseudo_label_file = "Module_D/data/pseudo_labels.csv"
    if os.path.exists(pseudo_label_file):
        os.remove(pseudo_label_file)
        
    retriever = CLIRRetriever()
    ranker = CLIRRanker(retriever)
    labeler = RelevanceLabeler(pseudo_label_file)
    
    for query in queries:
        # Get Top 3 Hybrid results
        res = ranker.rank_query(query, k=3, model='hybrid')
        for doc in res['results']:
            labeler.add_label(
                query=query,
                doc_url=doc['url'],
                relevant=True,
                language=doc['language'],
                annotator='pseudo_auto',
                notes='Auto-generated top-3 hybrid'
            )
            
    print(f"Generated pseudo-labels in {pseudo_label_file}")
    
    # 2. Run Evaluator
    # We use the pseudo label file to compute metrics
    evaluator = CLIREvaluator(label_file=pseudo_label_file, output_dir="Module_D/results/plots/ground_truth")
    
    evaluation = evaluator.evaluate_query_set(queries, k=10)
    
    # 3. Generate Visualizations
    plot_files = evaluator.generate_visualizations(evaluation)
    print("\nGenerated Ground Truth Plots:")
    for p in plot_files:
        print(f" - {p}")
        
    # 4. Print Report to Console
    report = evaluator.generate_report(evaluation)
    print("\n" + report)

if __name__ == "__main__":
    queries = [
        # English
        "election results 2024",
        "politics in Bangladesh",
        "education system reform", 
        "latest technology trends",
        "Dhaka Metro Rail",
        "inflation and economy",
        "cricket world cup",
        "climate change impact",
        
        # Bangla
        "দ্বাদশ জাতীয় সংসদ নির্বাচন",
        "বাংলাদেশের রাজনীতি",
        "শিক্ষা ব্যবস্থা",
        "শেয়ার বাজার পরিস্থিতি",
        "পদ্মা সেতু",
        "দ্রব্যমূল্যের ঊর্ধ্বগতি",
        "রোহিঙ্গা সংকট",
        "বাংলাদেশের আবহাওয়া",
        "সাইবার নিরাপত্তা"
    ]
    
    # Run both phases
    run_blind_evaluation(queries)
    run_ground_truth_evaluation(queries)
