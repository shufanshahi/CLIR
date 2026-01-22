"""
Indexing Module for CLIR System
Builds an inverted index with document metadata for Bangla and English documents.
Supports tokenization, metadata storage, and statistics for TF-IDF/BM25 retrieval.
"""

import json
import os
import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re
from pathlib import Path

# Import tokenizers (BanglaBERT for Bangla, XLM-RoBERTa for English)
try:
    from .bert_tokenizer import get_tokenizer as get_bert_tokenizer
    from .xlm_tokenizer import get_tokenizer as get_xlm_tokenizer
except ImportError:
    from bert_tokenizer import get_tokenizer as get_bert_tokenizer
    from xlm_tokenizer import get_tokenizer as get_xlm_tokenizer


class DocumentIndexer:
    """
    Indexes documents and builds an inverted index for cross-lingual retrieval.
    """
    
    def __init__(self):
        """
        Initialize DocumentIndexer.
        """
        # Inverted index: term -> {doc_id: [positions]}
        self.inverted_index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        
        # Document metadata: doc_id -> {title, body, url, date, language, tokens, doc_length}
        self.document_metadata: Dict[int, Dict] = {}
        
        # Document frequency: term -> number of documents containing this term
        self.document_frequency: Dict[str, int] = defaultdict(int)
        
        # Total number of documents
        self.total_documents = 0
        
        # Document ID counter
        self.doc_id_counter = 0
        
        # Load stopwords
        self.bangla_stopwords = self._load_bangla_stopwords()
        self.english_stopwords = self._load_english_stopwords()
        
        # Initialize tokenizers
        print("Initializing tokenizers...")
        try:
            # BanglaBERT for Bangla
            self.bangla_tokenizer = get_bert_tokenizer(bangla_model="sagorsarker/bangla-bert-base")
            # XLM-RoBERTa for English
            self.english_tokenizer = get_xlm_tokenizer(model_name="xlm-roberta-base")
        except Exception as e:
            print(f"Error loading tokenizers: {e}")
            raise e
    
    def _load_bangla_stopwords(self) -> Set[str]:
        """Load Bangla stopwords from bangla_stopwords.csv."""
        stopwords_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bangla_stopwords.csv')
        stopwords = set()
        
        try:
            import csv
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    if len(row) >= 2:  # Make sure row has at least 2 columns
                        word = row[1].strip()  # word_list is in second column
                        if word:  # Skip empty entries
                            stopwords.add(word)
            print(f"Loaded {len(stopwords)} Bangla stopwords from {stopwords_file}")
        except FileNotFoundError:
            print(f"Warning: {stopwords_file} not found. Using default stopwords.")
            stopwords = {
                'এবং', 'ও', 'কিন্তু', 'যা', 'যে', 'হবে', 'থাকবে', 'থাকে',
                'এই', 'একটি', 'একই', 'তিনি', 'সে', 'তারা', 'আমি', 'আমরা', 'তুমি'
            }
        except Exception as e:
            print(f"Warning: Error loading {stopwords_file}: {e}. Using default stopwords.")
            stopwords = {
                'এবং', 'ও', 'কিন্তু', 'যা', 'যে', 'হবে', 'থাকবে', 'থাকে',
                'এই', 'একটি', 'একই', 'তিনি', 'সে', 'তারা', 'আমি', 'আমরা', 'তুমি'
            }
        
        return stopwords
    
    def _load_english_stopwords(self) -> Set[str]:
        """Load English stopwords from english_stopwords.txt."""
        stopwords_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'english_stopwords.txt')
        stopwords = set()
        
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:  # Skip empty lines
                        stopwords.add(word)
            print(f"Loaded {len(stopwords)} English stopwords from {stopwords_file}")
        except FileNotFoundError:
            print(f"Warning: {stopwords_file} not found. Using default stopwords.")
            stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for',
                'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that',
                'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
                'have', 'had', 'what', 'said', 'each', 'which', 'their', 'if'
            }
        except Exception as e:
            print(f"Warning: Error loading {stopwords_file}: {e}. Using default stopwords.")
            stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for',
                'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that',
                'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
                'have', 'had', 'what', 'said', 'each', 'which', 'their', 'if'
            }
        
        return stopwords
    
    def tokenize_bangla(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """
        Tokenize Bangla text using BanglaBERT.
        """
        # Use BanglaBERT tokenizer
        # Note: tokenize_and_normalize handles normalization internally if needed, 
        # but here we rely on the tokenizer's split mostly. 
        # We pass language='bn' to the BERT wrapper.
        tokens = self.bangla_tokenizer.tokenize_and_normalize(text, language='bn', remove_special_tokens=True)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.bangla_stopwords]
        
        return tokens
    
    def tokenize_english(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """
        Tokenize English text using XLM-RoBERTa.
        """
        # Use XLM-RoBERTa tokenizer
        tokens = self.english_tokenizer.tokenize_and_normalize(text, remove_special_tokens=True)
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.english_stopwords]
        
        return tokens
    
    def _clean_tokens(self, tokens: List[str], language: str = "en") -> List[str]:
        """
        Clean tokens by removing single characters, punctuation, and unintelligible tokens.
        
        Args:
            tokens: List of token strings
            language: Language code ('bn' for Bangla, 'en' for English)
        
        Returns:
            List of cleaned token strings
        """
        cleaned = []
        
        for token in tokens:
            # Skip empty tokens
            if not token or not token.strip():
                continue
            
            # Remove all single character tokens (including digits, letters, punctuation)
            if len(token) == 1:
                continue
            
            # Remove tokens that are only punctuation or special characters
            if language == 'en':
                # For English: remove if token is only punctuation or special chars
                if not re.match(r'^[\w\d]+$', token):
                    continue
                # Remove tokens that are very short (1-2 chars) and not meaningful
                # Keep if it's all digits
                if len(token) <= 2 and not token.isdigit() and not token.isalpha():
                    continue
            else:
                # For Bangla: remove if token contains only punctuation
                # Check if token has any meaningful Bangla/English/numeric characters
                has_meaningful = False
                for char in token:
                    # Check for Bangla Unicode range (0x0980-0x09FF), English letters, or digits
                    code = ord(char)
                    if (0x0980 <= code <= 0x09FF) or char.isalnum():
                        has_meaningful = True
                        break
                
                if not has_meaningful:
                    continue
            
            # Remove very short tokens (1-2 chars) - all single and double character tokens
            if len(token) <= 2:
                continue
            
            cleaned.append(token)
        
        return cleaned
    
    def tokenize(self, text: str, language: str, remove_stopwords: bool = False) -> List[str]:
        """
        Tokenize text based on language using XLM-RoBERTa or simple tokenization.
        """
        if language == 'bn':
            return self.tokenize_bangla(text, remove_stopwords)
        elif language == 'en':
            return self.tokenize_english(text, remove_stopwords)
        else:
            # Default to English tokenization
            return self.tokenize_english(text, remove_stopwords)
    
    def add_document(self, document: Dict) -> int:
        """
        Add a document to the index.
        
        Flow:
        1. Normalize text using BERT (BanglaBERT for Bangla, BERT for English)
        2. Remove stopwords from normalized text
        3. Tokenize using XLM-RoBERTa for indexing (for both languages)
        
        Args:
            document: Dictionary with keys: title, body, url, date, language
            
        Returns:
            doc_id: Assigned document ID
        """
        doc_id = self.doc_id_counter
        self.doc_id_counter += 1
        
        # Extract document fields
        title = document.get('title', '')
        body = document.get('body', '')
        url = document.get('url', '')
        date = document.get('date', '')
        language = document.get('language', 'en')
        
        # Combine title and body for indexing (title is weighted more)
        # Repeat title 3 times to give it more weight
        full_text = f"{title} {title} {title} {body}"
        
        # Step 1: Normalize using basic normalization (without BERT)
        # As per user request: simple whitespace cleanup and lowercasing for English
        normalized_text = re.sub(r'\s+', ' ', full_text.strip())
        if language == 'en':
            normalized_text = normalized_text.lower()
        
        # Step 2: Remove stopwords from normalized text
        normalized_text = self._remove_stopwords(normalized_text, language)
        
        # Step 3: Tokenize using specific models
        if language == 'bn':
            tokens = self.tokenize_bangla(normalized_text, remove_stopwords=False)
        else:
            tokens = self.tokenize_english(normalized_text, remove_stopwords=False)
        
        # Store document metadata
        self.document_metadata[doc_id] = {
            'doc_id': doc_id,
            'title': title,
            'body': body,
            'url': url,
            'date': date,
            'language': language,
            'tokens': len(tokens),
            'doc_length': len(tokens),
            'unique_tokens': len(set(tokens))
        }
        
        # Build inverted index with term positions
        term_positions = defaultdict(list)
        for position, term in enumerate(tokens):
            term_positions[term].append(position)
        
        # Update inverted index
        for term, positions in term_positions.items():
            self.inverted_index[term][doc_id] = positions
            
            # Document frequency will be calculated in finalize_index()
        
        self.total_documents += 1
        
        return doc_id
    
    def _remove_stopwords(self, text: str, language: str) -> str:
        """
        Remove stopwords from normalized text.
        
        Args:
            text: Normalized text string
            language: Language code ('bn' for Bangla, 'en' for English)
        
        Returns:
            Text with stopwords removed
        """
        # Split text into words/tokens
        words = text.split()
        
        # Filter stopwords based on language
        if language == 'bn':
            filtered_words = [w for w in words if w not in self.bangla_stopwords]
        else:
            filtered_words = [w for w in words if w.lower() not in self.english_stopwords]
        
        # Join back to text
        return ' '.join(filtered_words)
    
    def finalize_index(self):
        """
        Finalize the index by calculating document frequencies.
        Should be called after all documents are added.
        """
        # Calculate document frequency for each term
        self.document_frequency = {
            term: len(doc_ids) 
            for term, doc_ids in self.inverted_index.items()
        }
        
        print(f"Index finalized:")
        print(f"  Total documents: {self.total_documents}")
        print(f"  Total unique terms: {len(self.inverted_index)}")
        print(f"  Average doc length: {sum(m['doc_length'] for m in self.document_metadata.values()) / self.total_documents:.2f}")
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """
        Get term frequency (TF) in a document.
        """
        if term in self.inverted_index and doc_id in self.inverted_index[term]:
            return len(self.inverted_index[term][doc_id])
        return 0
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get document frequency (DF) - number of documents containing the term.
        """
        return self.document_frequency.get(term, 0)
    
    def get_inverse_document_frequency(self, term: str) -> float:
        """
        Calculate IDF = log(N / df), where N is total documents and df is document frequency.
        """
        df = self.get_document_frequency(term)
        if df == 0:
            return 0.0
        import math
        return math.log(self.total_documents / df)
    
    def save_index(self, output_dir: str = "Module_A/indexed_data"):
        """
        Save the index and metadata to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save inverted index
        # Convert defaultdict to regular dict for JSON serialization
        index_dict = {
            term: {str(doc_id): positions for doc_id, positions in doc_dict.items()}
            for term, doc_dict in self.inverted_index.items()
        }
        
        with open(os.path.join(output_dir, 'inverted_index.json'), 'w', encoding='utf-8') as f:
            json.dump(index_dict, f, indent=2, ensure_ascii=False)
        
        # Save document metadata
        # Convert doc_id keys to strings for JSON
        metadata_dict = {str(k): v for k, v in self.document_metadata.items()}
        with open(os.path.join(output_dir, 'document_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        # Save document frequency
        with open(os.path.join(output_dir, 'document_frequency.json'), 'w', encoding='utf-8') as f:
            json.dump(self.document_frequency, f, indent=2, ensure_ascii=False)
        
        # Save index statistics
        stats = {
            'total_documents': self.total_documents,
            'total_unique_terms': len(self.inverted_index),
            'average_doc_length': sum(m['doc_length'] for m in self.document_metadata.values()) / self.total_documents if self.total_documents > 0 else 0,
            'languages': {
                lang: sum(1 for m in self.document_metadata.values() if m['language'] == lang)
                for lang in ['bn', 'en']
            }
        }
        
        with open(os.path.join(output_dir, 'index_stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nIndex saved to {output_dir}/")
        print(f"  - inverted_index.json")
        print(f"  - document_metadata.json")
        print(f"  - document_frequency.json")
        print(f"  - index_stats.json")
    
    def load_index(self, index_dir: str = "Module_A/indexed_data"):
        """
        Load the index from disk.
        """
        # Load inverted index
        with open(os.path.join(index_dir, 'inverted_index.json'), 'r', encoding='utf-8') as f:
            index_dict = json.load(f)
            self.inverted_index = {
                term: {int(doc_id): positions for doc_id, positions in doc_dict.items()}
                for term, doc_dict in index_dict.items()
            }
        
        # Load document metadata
        with open(os.path.join(index_dir, 'document_metadata.json'), 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
            self.document_metadata = {int(k): v for k, v in metadata_dict.items()}
        
        # Load document frequency
        with open(os.path.join(index_dir, 'document_frequency.json'), 'r', encoding='utf-8') as f:
            self.document_frequency = json.load(f)
        
        # Load statistics
        with open(os.path.join(index_dir, 'index_stats.json'), 'r', encoding='utf-8') as f:
            stats = json.load(f)
            self.total_documents = stats['total_documents']
            self.doc_id_counter = max(self.document_metadata.keys()) + 1 if self.document_metadata else 0
        
        print(f"Index loaded from {index_dir}/")


def load_articles_from_json(file_path: str) -> List[Dict]:
    """Load articles from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return articles


def main():
    """
    Main function to index all articles from different sources.
    """
    print("=" * 60)
    print("CLIR Indexing System")
    print("=" * 60)
    
    # Initialize indexer
    indexer = DocumentIndexer()
    
    # Define paths to article JSON files
    article_sources = [
        ("Module_A/Bangla_DB/ProthomAlo/articles.json", "ProthomAlo (Bangla)"),
        ("Module_A/Bangla_DB/BangladeshProtidin/articles.json", "BangladeshProtidin (Bangla)"),
        ("Module_A/Bangla_DB/BanglaTribune/articles.json", "BanglaTribune (Bangla)"),
        ("Module_A/Bangla_DB/newsbangla24/articles.json", "NewsBangla24 (Bangla)"),
        ("Module_A/Bangla_DB/DhakaPost/articles.json", "DhakaPost (Bangla)"),
        
        ("Module_A/English_DB/newagebd/articles.json", "NewAge (English)"),
        ("Module_A/English_DB/BangladeshProtidin/articles.json", "BangladeshProtidin (English)"),
        ("Module_A/English_DB/DailyAsianAge/articles.json", "DailyAsianAge (English)"),
        ("Module_A/English_DB/DhakaTribune/articles.json", "DhakaTribune (English)"),
        ("Module_A/English_DB/ProthomAlo/articles.json", "ProthomAlo (English)")
        
    ]
    
    total_articles = 0
    
    # Load and index articles from each source
    for file_path, source_name in article_sources:
        if os.path.exists(file_path):
            print(f"\nLoading articles from {source_name}...")
            articles = load_articles_from_json(file_path)
            print(f"  Found {len(articles)} articles")
            
            indexed_count = 0
            for article in articles:
                try:
                    doc_id = indexer.add_document(article)
                    indexed_count += 1
                except Exception as e:
                    print(f"  Error indexing article {article.get('url', 'unknown')}: {e}")
                    continue
            
            print(f"  Successfully indexed: {indexed_count} articles")
            total_articles += indexed_count
        else:
            print(f"\nWarning: {file_path} not found, skipping...")
    
    print(f"\n{'='*60}")
    print(f"Total articles indexed: {total_articles}")
    
    # Finalize index
    print("\nFinalizing index...")
    indexer.finalize_index()
    
    # Save index
    print("\nSaving index...")
    indexer.save_index()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Indexing Summary:")
    print("=" * 60)
    
    lang_stats = {}
    for doc_id, metadata in indexer.document_metadata.items():
        lang = metadata['language']
        if lang not in lang_stats:
            lang_stats[lang] = {'count': 0, 'total_tokens': 0}
        lang_stats[lang]['count'] += 1
        lang_stats[lang]['total_tokens'] += metadata['tokens']
    
    for lang, stats in lang_stats.items():
        lang_name = "Bangla" if lang == "bn" else "English"
        avg_tokens = stats['total_tokens'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{lang_name}: {stats['count']} documents, avg {avg_tokens:.1f} tokens/doc")
    
    print("=" * 60)
    print("Indexing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

