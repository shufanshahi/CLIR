"""
Module A - Document Indexing System
Creates inverted index with metadata for multilingual documents
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
import pickle
import math


class DocumentIndexer:
    """
    Creates and manages inverted index for document retrieval
    Supports TF-IDF and BM25 scoring
    """
    
    def __init__(self):
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.documents = []
        self.doc_metadata = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0
        self.vocabulary = set()
        
    def tokenize(self, text):
        """Simple tokenization - splits on whitespace and removes punctuation"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b[\w]+\b', text)
        return tokens
    
    def build_index_from_json_files(self, bangla_folders, english_folders):
        """
        Build index from all JSON article files
        
        Args:
            bangla_folders: List of paths to Bangla article folders
            english_folders: List of paths to English article folders
        """
        doc_id = 0
        
        # Process Bangla documents
        for folder in bangla_folders:
            articles_file = os.path.join(folder, 'articles.json')
            if os.path.exists(articles_file):
                with open(articles_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                    
                for article in articles:
                    # Create document
                    doc = {
                        'id': doc_id,
                        'title': article.get('title', ''),
                        'body': article.get('body', ''),
                        'url': article.get('url', ''),
                        'date': article.get('date', ''),
                        'language': 'bn'
                    }
                    
                    # Store document
                    self.documents.append(doc)
                    self.doc_metadata[doc_id] = doc
                    
                    # Tokenize and index
                    text = doc['title'] + ' ' + doc['body']
                    tokens = self.tokenize(text)
                    
                    # Calculate term frequencies
                    term_freq = defaultdict(int)
                    for token in tokens:
                        term_freq[token] += 1
                        self.vocabulary.add(token)
                    
                    # Store doc length
                    self.doc_lengths[doc_id] = len(tokens)
                    
                    # Build inverted index
                    for term, freq in term_freq.items():
                        self.inverted_index[term][doc_id] = freq
                    
                    doc_id += 1
        
        # Process English documents
        for folder in english_folders:
            articles_file = os.path.join(folder, 'articles.json')
            if os.path.exists(articles_file):
                with open(articles_file, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                    
                for article in articles:
                    # Create document
                    doc = {
                        'id': doc_id,
                        'title': article.get('title', ''),
                        'body': article.get('body', ''),
                        'url': article.get('url', ''),
                        'date': article.get('date', ''),
                        'language': 'en'
                    }
                    
                    # Store document
                    self.documents.append(doc)
                    self.doc_metadata[doc_id] = doc
                    
                    # Tokenize and index
                    text = doc['title'] + ' ' + doc['body']
                    tokens = self.tokenize(text)
                    
                    # Calculate term frequencies
                    term_freq = defaultdict(int)
                    for token in tokens:
                        term_freq[token] += 1
                        self.vocabulary.add(token)
                    
                    # Store doc length
                    self.doc_lengths[doc_id] = len(tokens)
                    
                    # Build inverted index
                    for term, freq in term_freq.items():
                        self.inverted_index[term][doc_id] = freq
                    
                    doc_id += 1
        
        # Calculate average document length
        self.total_docs = len(self.documents)
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        
        print(f"Index built successfully!")
        print(f"Total documents: {self.total_docs}")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Average document length: {self.avg_doc_length:.2f}")
    
    def calculate_tf_idf(self, term, doc_id):
        """Calculate TF-IDF score for a term in a document"""
        if term not in self.inverted_index or doc_id not in self.inverted_index[term]:
            return 0.0
        
        # Term frequency
        tf = self.inverted_index[term][doc_id] / self.doc_lengths.get(doc_id, 1)
        
        # Inverse document frequency
        df = len(self.inverted_index[term])
        idf = math.log((self.total_docs + 1) / (df + 1))
        
        return tf * idf
    
    def calculate_bm25(self, term, doc_id, k1=1.5, b=0.75):
        """Calculate BM25 score for a term in a document"""
        if term not in self.inverted_index or doc_id not in self.inverted_index[term]:
            return 0.0
        
        # Term frequency in document
        tf = self.inverted_index[term][doc_id]
        
        # Document length normalization
        doc_len = self.doc_lengths.get(doc_id, 1)
        
        # Document frequency
        df = len(self.inverted_index[term])
        
        # IDF component
        idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
        
        # BM25 formula
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def save_index(self, filepath='index.pkl'):
        """Save the index to a file"""
        data = {
            'inverted_index': dict(self.inverted_index),
            'documents': self.documents,
            'doc_metadata': self.doc_metadata,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'total_docs': self.total_docs,
            'vocabulary': self.vocabulary
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath='index.pkl'):
        """Load the index from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.inverted_index = defaultdict(lambda: defaultdict(list), data['inverted_index'])
        self.documents = data['documents']
        self.doc_metadata = data['doc_metadata']
        self.doc_lengths = data['doc_lengths']
        self.avg_doc_length = data['avg_doc_length']
        self.total_docs = data['total_docs']
        self.vocabulary = data['vocabulary']
        
        print(f"Index loaded from {filepath}")
        print(f"Total documents: {self.total_docs}")
        print(f"Vocabulary size: {len(self.vocabulary)}")


def main():
    """Build index from all collected articles"""
    
    # Define folders
    bangla_folders = [
        'Bangla_DB/BanglaTribune',
        'Bangla_DB/DhakaPost',
        'Bangla_DB/ProthomAlo'
    ]
    
    english_folders = [
        'English_DB/newagebd'
    ]
    
    # Create indexer
    indexer = DocumentIndexer()
    
    # Build index
    indexer.build_index_from_json_files(bangla_folders, english_folders)
    
    # Save index
    indexer.save_index('document_index.pkl')
    
    # Print statistics
    print("\n=== Index Statistics ===")
    print(f"Bangla documents: {sum(1 for doc in indexer.documents if doc['language'] == 'bn')}")
    print(f"English documents: {sum(1 for doc in indexer.documents if doc['language'] == 'en')}")


if __name__ == "__main__":
    main()
