"""
Comprehensive CLIR Evaluation System
Integrates ranking, metrics, labeling, and error analysis for complete evaluation.
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_C.retriever import CLIRRetriever
from .ranker import CLIRRanker
from .metrics import IRMetrics
from .labeling import RelevanceLabeler
from .error_analysis import ErrorAnalyzer

class CLIREvaluator:
    """
    Main evaluation class that orchestrates all evaluation components.
    """
    
    def __init__(self, index_dir: str = "Module_A/indexed_data", 
                 embedding_dir: str = "Module_C/data",
                 label_file: str = "Module_D/data/relevance_labels.csv",
                 output_dir: str = "Module_D/results"):
        """
        Initialize the evaluator with all necessary components.
        
        Args:
            index_dir: Directory containing indexed data
            embedding_dir: Directory containing embeddings
            label_file: Path to relevance labels file
            output_dir: Directory for evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        print("Initializing CLIR Evaluator...")
        self.retriever = CLIRRetriever(index_dir, embedding_dir)
        self.ranker = CLIRRanker(self.retriever)
        self.labeler = RelevanceLabeler(label_file)
        self.error_analyzer = ErrorAnalyzer(f"{output_dir}/error_analysis")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("✓ CLIR Evaluator initialized successfully")
    
    def evaluate_single_query(self, query: str, k: int = 10, 
                            models: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single query across multiple models.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            models: List of models to evaluate (default: all)
            
        Returns:
            Dictionary with evaluation results
        """
        if models is None:
            models = ['lexical', 'semantic', 'fuzzy', 'hybrid']
        
        evaluation = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'k': k,
            'models': {},
            'comparison': {}
        }
        
        # Get results from each model
        model_results = {}
        for model in models:
            try:
                result = self.ranker.rank_query(query, k=k, model=model)
                evaluation['models'][model] = result
                model_results[model] = result['results']
            except Exception as e:
                self.logger.error(f"Error evaluating {model} for query '{query}': {e}")
                evaluation['models'][model] = {'error': str(e)}
        
        # Get expected relevant documents from labels
        relevant_docs = self.labeler.get_relevant_docs(query)
        
        # Calculate metrics for each model
        metrics_comparison = {}
        for model, results in model_results.items():
            if results:
                doc_urls = [r['url'] for r in results]
                relevance_scores = {doc_id: 1.0 for doc_id in relevant_docs}
                
                metrics = {
                    'precision@10': IRMetrics.precision_at_k(relevant_docs, doc_urls, 10),
                    'recall@50': IRMetrics.recall_at_k(relevant_docs, doc_urls, 50),
                    'ndcg@10': IRMetrics.ndcg_at_k(relevance_scores, doc_urls, 10),
                    'mrr': IRMetrics.mean_reciprocal_rank([relevant_docs], [doc_urls])
                }
                metrics_comparison[model] = metrics
        
        evaluation['comparison'] = metrics_comparison
        
        # Perform error analysis
        if 'lexical' in model_results and 'semantic' in model_results:
            error_analysis = self.error_analyzer.analyze_query(
                query, model_results['lexical'], model_results['semantic'], relevant_docs)
            evaluation['error_analysis'] = error_analysis
        
        return evaluation
    
    def evaluate_query_set(self, queries: List[str], k: int = 10, 
                          models: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a set of queries comprehensively.
        
        Args:
            queries: List of query strings
            k: Number of results to retrieve
            models: List of models to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        if models is None:
            models = ['lexical', 'semantic', 'fuzzy', 'hybrid']
        
        print(f"Evaluating {len(queries)} queries across {len(models)} models...")
        start_time = time.time()
        
        batch_evaluation = {
            'timestamp': datetime.now().isoformat(),
            'queries': queries,
            'k': k,
            'models': models,
            'individual_results': [],
            'aggregate_metrics': {},
            'performance_stats': {},
            'error_analysis': {},
            'target_status': {}
        }
        
        # Evaluate each query
        individual_results = []
        for i, query in enumerate(queries):
            print(f"Progress: {i+1}/{len(queries)} - Evaluating: '{query}'")
            result = self.evaluate_single_query(query, k, models)
            individual_results.append(result)
        
        batch_evaluation['individual_results'] = individual_results
        
        # Aggregate metrics across all queries
        aggregate_metrics = self._aggregate_metrics(individual_results, models)
        batch_evaluation['aggregate_metrics'] = aggregate_metrics
        
        # Calculate performance statistics
        performance_stats = self._calculate_performance_stats(individual_results)
        batch_evaluation['performance_stats'] = performance_stats
        
        # Check target metrics
        target_status = self._check_target_metrics(aggregate_metrics)
        batch_evaluation['target_status'] = target_status
        
        # Perform batch error analysis
        error_analyses = [r.get('error_analysis', {}) for r in individual_results if 'error_analysis' in r]
        if error_analyses:
            batch_error_analysis = self.error_analyzer.analyze_batch(error_analyses)
            batch_evaluation['error_analysis'] = batch_error_analysis
        
        # Calculate total evaluation time
        total_time = time.time() - start_time
        batch_evaluation['total_evaluation_time'] = total_time
        
        print(f"Evaluation completed in {total_time:.2f} seconds")
        
        return batch_evaluation
    
    def _aggregate_metrics(self, individual_results: List[Dict], models: List[str]) -> Dict[str, Any]:
        """Aggregate metrics across all queries."""
        aggregated = {}
        
        for model in models:
            model_metrics = defaultdict(list)
            
            for result in individual_results:
                if model in result.get('comparison', {}):
                    for metric, value in result['comparison'][model].items():
                        model_metrics[metric].append(value)
            
            # Calculate averages
            aggregated[model] = {}
            for metric, values in model_metrics.items():
                if values:
                    aggregated[model][f'avg_{metric}'] = sum(values) / len(values)
                    aggregated[model][f'std_{metric}'] = (sum((x - aggregated[model][f'avg_{metric}'])**2 for x in values) / len(values))**0.5
                    aggregated[model][f'min_{metric}'] = min(values)
                    aggregated[model][f'max_{metric}'] = max(values)
        
        return aggregated
    
    def _calculate_performance_stats(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Calculate performance statistics."""
        all_times = []
        all_confidences = []
        low_confidence_count = 0
        
        for result in individual_results:
            for model, model_result in result.get('models', {}).items():
                if 'total_time_ms' in model_result:
                    all_times.append(model_result['total_time_ms'])
                
                if model_result.get('results') and model_result['results']:
                    top_confidence = model_result['results'][0].get('confidence_score', 0)
                    all_confidences.append(top_confidence)
                    
                    if top_confidence < 0.2:
                        low_confidence_count += 1
        
        stats = {
            'timing': {
                'avg_time_ms': sum(all_times) / len(all_times) if all_times else 0,
                'min_time_ms': min(all_times) if all_times else 0,
                'max_time_ms': max(all_times) if all_times else 0
            },
            'confidence': {
                'avg_top_confidence': sum(all_confidences) / len(all_confidences) if all_confidences else 0,
                'min_top_confidence': min(all_confidences) if all_confidences else 0,
                'max_top_confidence': max(all_confidences) if all_confidences else 0
            },
            'low_confidence_queries': low_confidence_count,
            'total_queries_evaluated': len(individual_results)
        }
        
        return stats
    
    def _check_target_metrics(self, aggregate_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check if metrics meet assignment targets."""
        targets = {
            'precision@10': 0.6,
            'recall@50': 0.5,
            'ndcg@10': 0.5,
            'mrr': 0.4
        }
        
        status = {}
        
        for model, metrics in aggregate_metrics.items():
            model_status = {}
            overall_passed = True
            
            for metric, target_value in targets.items():
                avg_key = f'avg_{metric}'
                if avg_key in metrics:
                    computed_value = metrics[avg_key]
                    passed = computed_value >= target_value
                    model_status[metric] = {
                        'computed': computed_value,
                        'target': target_value,
                        'passed': passed,
                        'gap': computed_value - target_value
                    }
                    
                    if not passed:
                        overall_passed = False
            
            model_status['overall_passed'] = overall_passed
            status[model] = model_status
        
        return status
    
    def create_annotation_templates(self, queries: List[str], docs_per_query: int = 20) -> List[str]:
        """
        Create annotation templates for manual labeling.
        
        Args:
            queries: List of queries to create templates for
            docs_per_query: Number of documents per query in template
            
        Returns:
            List of template file paths
        """
        template_files = []
        
        for query in queries:
            # Get candidate documents for this query
            try:
                result = self.ranker.rank_query(query, k=docs_per_query, model='hybrid')
                candidate_docs = result['results']
                
                # Create template
                template_file = self.labeler.create_annotation_template(query, candidate_docs)
                if template_file:
                    template_files.append(template_file)
                    
            except Exception as e:
                self.logger.error(f"Error creating template for query '{query}': {e}")
        
        return template_files
    
    def save_evaluation(self, evaluation: Dict[str, Any], filename: str = None) -> str:
        """Save evaluation results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clir_evaluation_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"Evaluation saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving evaluation: {e}")
            return ""
    
    def generate_visualizations(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generate visualization plots for evaluation results based on IR metrics."""
        plot_files = []
        
        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_context("notebook", font_scale=1.1)
            sns.set_style("whitegrid")
            
            if 'aggregate_metrics' not in evaluation:
                return []

            models = list(evaluation['aggregate_metrics'].keys())
            # Define metrics and their targets based on Module D requirements
            metrics_config = {
                'precision@10': {'target': 0.6, 'title': 'Precision@10 (Target ≥ 0.6)'},
                'recall@50': {'target': 0.5, 'title': 'Recall@50 (Target ≥ 0.5)'},
                'ndcg@10': {'target': 0.5, 'title': 'nDCG@10 (Target ≥ 0.5)'},
                'mrr': {'target': 0.4, 'title': 'Mean Reciprocal Rank (Target ≥ 0.4)'}
            }

            # Prepare data frame for plotting
            data = []
            for model in models:
                vals = evaluation['aggregate_metrics'][model]
                for metric_key, config in metrics_config.items():
                    avg_key = f'avg_{metric_key}'
                    if avg_key in vals:
                        data.append({
                            'Model': model,
                            'Metric': metric_key,
                            'Score': vals[avg_key],
                            'Target': config['target']
                        })
            
            if not data:
                return []
                
            df = pd.DataFrame(data)

            # 1. Individual Metric Charts with Target Lines
            for metric_key, config in metrics_config.items():
                subset = df[df['Metric'] == metric_key]
                if subset.empty:
                    continue
                
                plt.figure(figsize=(10, 6))
                # Bar plot for scores
                ax = sns.barplot(x='Model', y='Score', data=subset, palette='viridis', hue='Model', legend=False)
                
                # Add target line
                plt.axhline(y=config['target'], color='r', linestyle='--', linewidth=2, label=f"Target ({config['target']})")
                
                # Add score labels on top of bars
                for p in ax.patches:
                    height = p.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.3f}', 
                                (p.get_x() + p.get_width() / 2., height), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points',
                                fontweight='bold')
                
                plt.title(config['title'], fontsize=14, pad=20)
                plt.xlabel('Retrieval Model', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.ylim(0, 1.1) # IR metrics are usually 0-1
                plt.legend(loc='upper right')
                plt.tight_layout()
                
                # Save plot
                filename = f"{metric_key.split('@')[0]}_comparison.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300)
                plt.close()
                plot_files.append(filepath)

            # 2. Combined Summary Chart (Grouped Bar Chart)
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x='Metric', y='Score', hue='Model', data=df, palette='viridis')
            
            plt.title('Overall Retrieval Performance Comparison', fontsize=16, pad=20)
            plt.xlabel('Metric', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1.1)
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, 'overall_metrics_summary.png')
            plt.savefig(filepath, dpi=300)
            plt.close()
            plot_files.append(filepath)

            print(f"Generated {len(plot_files)} metric visualization plots based on Module D requirements")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        return plot_files
    
    def generate_report(self, evaluation: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = []
        
        # Header
        report_lines.append("# CLIR System Evaluation Report")
        report_lines.append(f"Generated: {evaluation.get('timestamp', 'Unknown')}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        if 'aggregate_metrics' in evaluation:
            best_model = None
            best_score = 0
            
            for model, metrics in evaluation['aggregate_metrics'].items():
                avg_precision = metrics.get('avg_precision@10', 0)
                if avg_precision > best_score:
                    best_score = avg_precision
                    best_model = model
            
            report_lines.append(f"- **Best Performing Model**: {best_model or 'N/A'}")
            report_lines.append(f"- **Queries Evaluated**: {evaluation.get('queries', ['N/A']).__len__()}")
            report_lines.append(f"- **Evaluation Time**: {evaluation.get('total_evaluation_time', 0):.2f} seconds")
        report_lines.append("")
        
        # Performance Metrics
        if 'aggregate_metrics' in evaluation:
            report_lines.append("## Performance Metrics")
            
            # Create metrics table
            report_lines.append("| Model | Precision@10 | Recall@50 | nDCG@10 | MRR | Target Met |")
            report_lines.append("|-------|-------------|-----------|----------|-----|------------|")
            
            for model, metrics in evaluation['aggregate_metrics'].items():
                precision = metrics.get('avg_precision@10', 0)
                recall = metrics.get('avg_recall@50', 0)
                ndcg = metrics.get('avg_ndcg@10', 0)
                mrr = metrics.get('avg_mrr', 0)
                
                # Check if targets are met
                targets_met = (precision >= 0.6 and recall >= 0.5 and ndcg >= 0.5 and mrr >= 0.4)
                target_status = "✅" if targets_met else "❌"
                
                report_lines.append(f"| {model} | {precision:.3f} | {recall:.3f} | {ndcg:.3f} | {mrr:.3f} | {target_status} |")
            
            report_lines.append("")
        
        # Performance Statistics
        if 'performance_stats' in evaluation:
            stats = evaluation['performance_stats']
            report_lines.append("## Performance Statistics")
            report_lines.append(f"- **Average Query Time**: {stats.get('timing', {}).get('avg_time_ms', 0):.2f} ms")
            report_lines.append(f"- **Average Top Confidence**: {stats.get('confidence', {}).get('avg_top_confidence', 0):.3f}")
            report_lines.append(f"- **Low Confidence Queries**: {stats.get('low_confidence_queries', 0)}")
            report_lines.append("")
        
        # Error Analysis
        if 'error_analysis' in evaluation:
            error_analysis = evaluation['error_analysis']
            report_lines.append("## Error Analysis")
            
            if 'overall_summary' in error_analysis:
                summary = error_analysis['overall_summary']
                report_lines.append(f"- **Total Errors Found**: {summary.get('total_errors', 0)}")
                report_lines.append(f"- **Queries with Errors**: {summary.get('queries_with_errors', 0)}")
                report_lines.append(f"- **Error Rate**: {summary.get('error_rate', 0):.2%}")
                report_lines.append(f"- **Most Common Error**: {summary.get('most_common_error', 'None')}")
                report_lines.append("")
            
            if 'category_summaries' in error_analysis:
                report_lines.append("### Error Categories")
                for category, info in error_analysis['category_summaries'].items():
                    if info['count'] > 0:
                        report_lines.append(f"- **{category.replace('_', ' ').title()}**: {info['count']} occurrences ({info['percentage']:.1f}%)")
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        
        # Generate recommendations based on results
        recommendations = []
        
        if 'target_status' in evaluation:
            failing_models = [model for model, status in evaluation['target_status'].items() 
                            if not status.get('overall_passed', True)]
            if failing_models:
                recommendations.append(f"- Consider improving models: {', '.join(failing_models)}")
        
        if 'performance_stats' in evaluation:
            avg_time = evaluation['performance_stats'].get('timing', {}).get('avg_time_ms', 0)
            if avg_time > 1000:  # If average time is over 1 second
                recommendations.append("- Optimize query processing for better performance")
        
        if 'error_analysis' in evaluation:
            most_common = evaluation['error_analysis'].get('overall_summary', {}).get('most_common_error')
            if most_common:
                recommendations.append(f"- Address {most_common.replace('_', ' ')} issues")
        
        if not recommendations:
            recommendations.append("- System is performing well within target thresholds")
        
        for rec in recommendations:
            report_lines.append(rec)
        
        report_lines.append("")
        
        return "\n".join(report_lines)

if __name__ == "__main__":
    # Test the evaluator
    print("Testing CLIR Evaluator...")
    
    try:
        evaluator = CLIREvaluator()
        
        # Sample queries for testing
        test_queries = ["election", "শিক্ষা", "Bangladesh"]
        
        # Evaluate single query
        print("\nTesting single query evaluation...")
        single_result = evaluator.evaluate_single_query(test_queries[0], k=5)
        print(f"Single query evaluation completed for: '{single_result['query']}'")
        
        # Evaluate query set
        print("\nTesting batch evaluation...")
        batch_result = evaluator.evaluate_query_set(test_queries[:2], k=5)
        print(f"Batch evaluation completed for {len(test_queries[:2])} queries")
        
        # Generate report
        report = evaluator.generate_report(batch_result)
        print(f"\nGenerated report ({len(report)} characters)")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure Module A, B, and C are properly set up with indexed data.")
