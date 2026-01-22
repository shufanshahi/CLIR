"""
Relevance Labeling System for CLIR Evaluation
Handles manual relevance labeling and CSV format management.
"""

import csv
import json
import os
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import random
from datetime import datetime

class RelevanceLabeler:
    """
    Class for managing relevance labeling of query-document pairs.
    """
    
    def __init__(self, label_file: str = "Module_D/data/relevance_labels.csv"):
        """
        Initialize the labeler with a CSV file path.
        
        Args:
            label_file: Path to the CSV file for storing labels
        """
        self.label_file = label_file
        self.labels = {}
        self.annotators = set()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        
        # Load existing labels
        self._load_labels()
    
    def _load_labels(self):
        """Load existing labels from CSV file."""
        if os.path.exists(self.label_file):
            try:
                with open(self.label_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        query = row['query']
                        doc_url = row['doc_url']
                        key = (query, doc_url)
                        
                        self.labels[key] = {
                            'relevant': row['relevant'].lower() == 'yes',
                            'language': row['language'],
                            'annotator': row['annotator'],
                            'timestamp': row.get('timestamp', ''),
                            'notes': row.get('notes', '')
                        }
                        
                        self.annotators.add(row['annotator'])
                
                print(f"Loaded {len(self.labels)} existing relevance labels")
            except Exception as e:
                print(f"Error loading labels: {e}")
                self.labels = {}
        else:
            print("No existing label file found. Starting fresh.")
    
    def add_label(self, query: str, doc_url: str, relevant: bool, 
                  language: str, annotator: str, notes: str = "") -> bool:
        """
        Add or update a relevance label.
        
        Args:
            query: Query string
            doc_url: Document URL
            relevant: Whether the document is relevant to the query
            language: Document language
            annotator: Person who made the annotation
            notes: Optional notes about the labeling decision
            
        Returns:
            True if label was added successfully
        """
        key = (query, doc_url)
        
        self.labels[key] = {
            'relevant': relevant,
            'language': language,
            'annotator': annotator,
            'timestamp': datetime.now().isoformat(),
            'notes': notes
        }
        
        self.annotators.add(annotator)
        return self._save_labels()
    
    def get_label(self, query: str, doc_url: str) -> Optional[Dict]:
        """
        Get the relevance label for a specific query-document pair.
        
        Args:
            query: Query string
            doc_url: Document URL
            
        Returns:
            Label dictionary or None if not found
        """
        key = (query, doc_url)
        return self.labels.get(key)
    
    def get_relevant_docs(self, query: str) -> Set[str]:
        """
        Get all relevant documents for a query.
        
        Args:
            query: Query string
            
        Returns:
            Set of relevant document URLs
        """
        relevant_docs = set()
        for (q, doc_url), label_info in self.labels.items():
            if q == query and label_info['relevant']:
                relevant_docs.add(doc_url)
        return relevant_docs
    
    def get_query_labels(self, query: str) -> Dict[str, Dict]:
        """
        Get all labels for a specific query.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary mapping doc_url to label info
        """
        query_labels = {}
        for (q, doc_url), label_info in self.labels.items():
            if q == query:
                query_labels[doc_url] = label_info
        return query_labels
    
    def get_unlabeled_pairs(self, query: str, candidate_docs: List[Dict]) -> List[Dict]:
        """
        Get unlabeled query-document pairs for annotation.
        
        Args:
            query: Query string
            candidate_docs: List of candidate documents with 'url' field
            
        Returns:
            List of unlabeled documents
        """
        unlabeled = []
        for doc in candidate_docs:
            if self.get_label(query, doc['url']) is None:
                unlabeled.append(doc)
        return unlabeled
    
    def sample_for_annotation(self, query: str, candidate_docs: List[Dict], 
                            sample_size: int = 10, random_seed: int = None) -> List[Dict]:
        """
        Sample documents for annotation, prioritizing unlabeled ones.
        
        Args:
            query: Query string
            candidate_docs: List of candidate documents
            sample_size: Number of documents to sample
            random_seed: Random seed for reproducibility
            
        Returns:
            List of documents to annotate
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        unlabeled = self.get_unlabeled_pairs(query, candidate_docs)
        
        # If we have enough unlabeled docs, sample from them
        if len(unlabeled) >= sample_size:
            return random.sample(unlabeled, sample_size)
        
        # Otherwise, take all unlabeled and fill with random labeled docs
        labeled = [doc for doc in candidate_docs if doc not in unlabeled]
        remaining_needed = sample_size - len(unlabeled)
        
        if labeled:
            additional = random.sample(labeled, min(remaining_needed, len(labeled)))
            return unlabeled + additional
        
        return unlabeled
    
    def _save_labels(self) -> bool:
        """Save labels to CSV file."""
        try:
            with open(self.label_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['query', 'doc_url', 'relevant', 'language', 'annotator', 'timestamp', 'notes']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for (query, doc_url), label_info in self.labels.items():
                    writer.writerow({
                        'query': query,
                        'doc_url': doc_url,
                        'relevant': 'yes' if label_info['relevant'] else 'no',
                        'language': label_info['language'],
                        'annotator': label_info['annotator'],
                        'timestamp': label_info['timestamp'],
                        'notes': label_info['notes']
                    })
            
            print(f"Saved {len(self.labels)} labels to {self.label_file}")
            return True
        except Exception as e:
            print(f"Error saving labels: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get labeling statistics.
        
        Returns:
            Dictionary with labeling statistics
        """
        if not self.labels:
            return {'total_labels': 0}
        
        # Count by relevance
        relevant_count = sum(1 for label in self.labels.values() if label['relevant'])
        irrelevant_count = len(self.labels) - relevant_count
        
        # Count by language
        language_counts = defaultdict(int)
        for label_info in self.labels.values():
            language_counts[label_info['language']] += 1
        
        # Count by annotator
        annotator_counts = defaultdict(int)
        for label_info in self.labels.values():
            annotator_counts[label_info['annotator']] += 1
        
        # Count unique queries
        unique_queries = len(set(q for (q, _) in self.labels.keys()))
        
        return {
            'total_labels': len(self.labels),
            'relevant_labels': relevant_count,
            'irrelevant_labels': irrelevant_count,
            'relevance_rate': relevant_count / len(self.labels) if self.labels else 0,
            'unique_queries': unique_queries,
            'language_distribution': dict(language_counts),
            'annotator_distribution': dict(annotator_counts),
            'annotators': list(self.annotators)
        }
    
    def export_for_evaluation(self) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]]]:
        """
        Export labels in format suitable for evaluation.
        
        Returns:
            Tuple of (relevant_docs_dict, all_retrieved_docs_dict)
        """
        relevant_docs_dict = defaultdict(set)
        all_retrieved_docs_dict = defaultdict(list)
        
        # Group by query
        query_docs = defaultdict(list)
        for (query, doc_url), label_info in self.labels.items():
            query_docs[query].append(doc_url)
            
            if label_info['relevant']:
                relevant_docs_dict[query].add(doc_url)
        
        # For evaluation, we need retrieved lists (all docs for each query)
        for query, docs in query_docs.items():
            all_retrieved_docs_dict[query] = docs
        
        return dict(relevant_docs_dict), dict(all_retrieved_docs_dict)
    
    def create_annotation_template(self, query: str, candidate_docs: List[Dict], 
                                 output_file: str = None) -> str:
        """
        Create a CSV template for manual annotation.
        
        Args:
            query: Query string
            candidate_docs: List of candidate documents
            output_file: Output file path (auto-generated if None)
            
        Returns:
            Path to the created template file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_query = safe_query.replace(' ', '_')[:50]
            output_file = f"Module_D/data/annotation_template_{safe_query}_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['query', 'doc_url', 'title', 'language', 'relevant', 'annotator', 'notes']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for doc in candidate_docs:
                    writer.writerow({
                        'query': query,
                        'doc_url': doc['url'],
                        'title': doc.get('title', ''),
                        'language': doc.get('language', ''),
                        'relevant': '',  # To be filled by annotator
                        'annotator': '',  # To be filled by annotator
                        'notes': ''       # Optional notes
                    })
            
            print(f"Created annotation template: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error creating template: {e}")
            return ""

if __name__ == "__main__":
    # Test the relevance labeler
    print("Testing Relevance Labeler...")
    
    # Create sample documents
    sample_docs = [
        {'url': 'http://example.com/doc1', 'title': 'Election Results 2024', 'language': 'en'},
        {'url': 'http://example.com/doc2', 'title': 'Sports News', 'language': 'en'},
        {'url': 'http://example.com/doc3', 'title': 'নির্বাচন ফলাফল', 'language': 'bn'},
        {'url': 'http://example.com/doc4', 'title': 'Technology Update', 'language': 'en'},
        {'url': 'http://example.com/doc5', 'title': 'শিক্ষা ব্যবস্থা', 'language': 'bn'}
    ]
    
    # Initialize labeler
    labeler = RelevanceLabeler("test_labels.csv")
    
    # Add some sample labels
    query = "election"
    labeler.add_label(query, sample_docs[0]['url'], True, 'en', 'annotator1', 'Clearly about election')
    labeler.add_label(query, sample_docs[1]['url'], False, 'en', 'annotator1', 'Sports, not election')
    labeler.add_label(query, sample_docs[2]['url'], True, 'bn', 'annotator1', 'Bangla election content')
    
    # Test retrieval
    relevant_docs = labeler.get_relevant_docs(query)
    print(f"Relevant docs for '{query}': {relevant_docs}")
    
    # Get statistics
    stats = labeler.get_statistics()
    print(f"Labeling stats: {stats}")
    
    # Create annotation template
    template_file = labeler.create_annotation_template(query, sample_docs)
    print(f"Template created: {template_file}")
    
    # Clean up test file
    if os.path.exists("test_labels.csv"):
        os.remove("test_labels.csv")
    if template_file and os.path.exists(template_file):
        os.remove(template_file)
