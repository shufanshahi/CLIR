"""
Module B - Query Processing & Cross-Lingual Handling
Implements complete query processing pipeline:
1. Language Detection
2. Normalization
3. Query Translation
4. Query Expansion (optional)
5. Named Entity Mapping (optional)
"""

import re
from typing import List, Dict, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class QueryProcessor:
    """
    Complete query processing pipeline for cross-lingual information retrieval
    """
    
    def __init__(self):
        # Common Bangla stopwords
        self.bangla_stopwords = {
            'এবং', 'কিন্তু', 'বা', 'যদি', 'তবে', 'সে', 'তিনি', 'এই', 'যে', 'তার',
            'করে', 'করা', 'হয়', 'হওয়া', 'থেকে', 'সঙ্গে', 'জন্য', 'দ্বারা'
        }
        
        # Common English stopwords
        self.english_stopwords = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'a', 'an', 'and', 'or', 'but', 'if', 'then', 'of', 'at',
            'by', 'for', 'with', 'to', 'from', 'in', 'on', 'this', 'that'
        }
        
        # Named entity mappings (Bangla <-> English)
        self.ne_mappings = {
            'bangladesh': 'বাংলাদেশ',
            'বাংলাদেশ': 'bangladesh',
            'dhaka': 'ঢাকা',
            'ঢাকা': 'dhaka',
            'chittagong': 'চট্টগ্রাম',
            'চট্টগ্রাম': 'chittagong',
            'sylhet': 'সিলেট',
            'সিলেট': 'sylhet',
            'rajshahi': 'রাজশাহী',
            'রাজশাহী': 'rajshahi',
            'khulna': 'খুলনা',
            'খুলনা': 'khulna',
            'barisal': 'বরিশাল',
            'বরিশাল': 'barisal',
            'rangpur': 'রংপুর',
            'রংপুর': 'rangpur',
            'mymensingh': 'ময়মনসিংহ',
            'ময়মনসিংহ': 'mymensingh'
        }
        
        # Simple synonym dictionary for query expansion
        self.synonyms_en = {
            'university': ['college', 'institution', 'school'],
            'student': ['pupil', 'learner'],
            'government': ['administration', 'state'],
            'police': ['law enforcement', 'cops'],
            'hospital': ['clinic', 'medical center']
        }
        
        self.synonyms_bn = {
            'বিশ্ববিদ্যালয়': ['কলেজ', 'শিক্ষাপ্রতিষ্ঠান'],
            'ছাত্র': ['শিক্ষার্থী'],
            'সরকার': ['প্রশাসন', 'রাষ্ট্র'],
            'পুলিশ': ['আইনশৃঙ্খলা'],
            'হাসপাতাল': ['চিকিৎসালয়']
        }
    
    def detect_language(self, query: str) -> str:
        """
        Detect whether query is in Bangla or English
        
        Args:
            query: Input query string
            
        Returns:
            'bn' for Bangla, 'en' for English
        """
        # Count Bangla Unicode characters
        bangla_chars = sum(1 for char in query if '\u0980' <= char <= '\u09FF')
        
        # If more than 30% characters are Bangla, consider it Bangla
        if len(query) > 0 and (bangla_chars / len(query)) > 0.3:
            return 'bn'
        else:
            return 'en'
    
    def normalize_query(self, query: str, language: str, remove_stopwords: bool = False) -> str:
        """
        Normalize query text
        - Lowercase for English
        - Remove extra whitespace
        - Optionally remove stopwords
        
        Args:
            query: Input query
            language: 'bn' or 'en'
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Normalized query string
        """
        # Lowercase (mainly for English)
        if language == 'en':
            query = query.lower()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove stopwords if requested
        if remove_stopwords:
            if language == 'bn':
                words = query.split()
                query = ' '.join([w for w in words if w not in self.bangla_stopwords])
            elif language == 'en':
                words = query.split()
                query = ' '.join([w for w in words if w not in self.english_stopwords])
        
        return query
    
    def translate_query_simple(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Simple translation using named entity mapping
        For production, use Google Translate API or similar
        
        Args:
            query: Query to translate
            source_lang: Source language ('bn' or 'en')
            target_lang: Target language ('bn' or 'en')
            
        Returns:
            Translated query (or original if translation not available)
        """
        # This is a placeholder - in real implementation, use translation API
        # For now, just map named entities
        translated_words = []
        words = query.split()
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.ne_mappings:
                translated_words.append(self.ne_mappings[word_lower])
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def translate_query_api(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Translate query using external API (Google Translate)
        
        This requires: pip install googletrans==4.0.0rc1
        
        Args:
            query: Query to translate
            source_lang: Source language ('bn' or 'en')
            target_lang: Target language ('bn' or 'en')
            
        Returns:
            Translated query
        """
        try:
            from googletrans import Translator
            translator = Translator()
            
            # Map language codes
            lang_map = {'bn': 'bn', 'en': 'en'}
            
            result = translator.translate(
                query,
                src=lang_map[source_lang],
                dest=lang_map[target_lang]
            )
            
            return result.text
        except ImportError:
            print("Warning: googletrans not installed. Using simple translation.")
            return self.translate_query_simple(query, source_lang, target_lang)
        except Exception as e:
            print(f"Translation error: {e}. Using original query.")
            return query
    
    def expand_query(self, query: str, language: str, max_synonyms: int = 2) -> List[str]:
        """
        Expand query with synonyms
        
        Args:
            query: Input query
            language: 'bn' or 'en'
            max_synonyms: Maximum synonyms per term
            
        Returns:
            List of expanded query terms
        """
        words = query.split()
        expanded_terms = []
        
        synonym_dict = self.synonyms_bn if language == 'bn' else self.synonyms_en
        
        for word in words:
            expanded_terms.append(word)
            
            # Add synonyms if available
            if word.lower() in synonym_dict:
                synonyms = synonym_dict[word.lower()][:max_synonyms]
                expanded_terms.extend(synonyms)
        
        return expanded_terms
    
    def extract_named_entities(self, query: str) -> List[str]:
        """
        Simple named entity extraction
        Identifies known entities from mapping dictionary
        
        Args:
            query: Input query
            
        Returns:
            List of identified named entities
        """
        entities = []
        words = query.split()
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.ne_mappings:
                entities.append(word)
        
        return entities
    
    def map_named_entities(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Map named entities to target language
        
        Args:
            query: Input query
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Query with mapped named entities
        """
        words = query.split()
        mapped_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.ne_mappings:
                # Map to target language
                mapped_words.append(self.ne_mappings[word_lower])
            else:
                mapped_words.append(word)
        
        return ' '.join(mapped_words)
    
    def process_query(
        self,
        query: str,
        target_language: str = None,
        remove_stopwords: bool = False,
        expand: bool = False,
        translate: bool = True,
        use_api: bool = False
    ) -> Dict:
        """
        Complete query processing pipeline
        
        Args:
            query: Input query
            target_language: Target language for translation ('bn' or 'en')
                           If None, processes for both languages
            remove_stopwords: Whether to remove stopwords
            expand: Whether to perform query expansion
            translate: Whether to translate query
            use_api: Whether to use translation API (requires googletrans)
            
        Returns:
            Dictionary containing processed queries and metadata
        """
        # Step 1: Detect language
        detected_lang = self.detect_language(query)
        
        # Step 2: Normalize
        normalized_query = self.normalize_query(query, detected_lang, remove_stopwords)
        
        # Step 3: Extract named entities
        named_entities = self.extract_named_entities(normalized_query)
        
        result = {
            'original_query': query,
            'detected_language': detected_lang,
            'normalized_query': normalized_query,
            'named_entities': named_entities,
            'queries': {}
        }
        
        # Process for same language
        same_lang_terms = [normalized_query]
        if expand:
            same_lang_terms = self.expand_query(normalized_query, detected_lang)
        
        result['queries'][detected_lang] = {
            'query': normalized_query,
            'expanded_terms': same_lang_terms,
            'is_translated': False
        }
        
        # Step 4 & 5: Translate and map named entities for cross-lingual search
        if translate:
            other_lang = 'en' if detected_lang == 'bn' else 'bn'
            
            # Translate
            if use_api:
                translated_query = self.translate_query_api(
                    normalized_query, detected_lang, other_lang
                )
            else:
                # Simple translation with NE mapping
                translated_query = self.map_named_entities(
                    normalized_query, detected_lang, other_lang
                )
            
            # Expand translated query
            translated_terms = [translated_query]
            if expand:
                translated_terms = self.expand_query(translated_query, other_lang)
            
            result['queries'][other_lang] = {
                'query': translated_query,
                'expanded_terms': translated_terms,
                'is_translated': True
            }
        
        # If target language specified, return only that
        if target_language and target_language in result['queries']:
            result['target_query'] = result['queries'][target_language]
        
        return result


def main():
    """Demo of query processing"""
    processor = QueryProcessor()
    
    # Test queries
    test_queries = [
        "Bangladesh university student",
        "ঢাকা বিশ্ববিদ্যালয় ছাত্র",
        "What is the capital of Bangladesh?",
        "বাংলাদেশের রাজধানী কোথায়?"
    ]
    
    print("=== Query Processing Demo ===\n")
    
    for query in test_queries:
        print(f"Original Query: {query}")
        result = processor.process_query(
            query,
            remove_stopwords=True,
            expand=True,
            translate=True,
            use_api=False  # Set to True if you have googletrans installed
        )
        
        print(f"Detected Language: {result['detected_language']}")
        print(f"Normalized: {result['normalized_query']}")
        print(f"Named Entities: {result['named_entities']}")
        
        for lang, query_info in result['queries'].items():
            print(f"\n  {lang.upper()} Query:")
            print(f"    Query: {query_info['query']}")
            print(f"    Expanded: {query_info['expanded_terms']}")
            print(f"    Translated: {query_info['is_translated']}")
        
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
