"""
Query Processing & Cross-Lingual Handling Module
Implements language detection, normalization, translation, expansion, and named entity mapping.
"""

import re
import json
import time
import os
import csv
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import requests
from pathlib import Path
import torch
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found.")

# Try to import BERT tokenizer (BanglaBERT for Bangla, BERT for English)
try:
    from ..Module_A.bert_tokenizer import get_tokenizer, BERTTokenizerWrapper
    USE_BERT_TOKENIZER = True
except ImportError:
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from Module_A.bert_tokenizer import get_tokenizer, BERTTokenizerWrapper
        USE_BERT_TOKENIZER = True
    except ImportError:
        USE_BERT_TOKENIZER = False
        print("Warning: BERT tokenizer not available. Using simple normalization.")


class LanguageDetector:
    """Detects the language of a query (Bangla or English)."""
    
    def __init__(self):
        # Unicode ranges for Bangla script
        self.bangla_range = (0x0980, 0x09FF)
        
        # Common words removed - relying on external resources
    
    def contains_bangla_characters(self, text: str) -> bool:
        """Check if text contains Bangla Unicode characters."""
        for char in text:
            code_point = ord(char)
            if self.bangla_range[0] <= code_point <= self.bangla_range[1]:
                return True
        return False
    
    def detect(self, query: str) -> str:
        """
        Detect the language of the query.
        Returns 'bn' for Bangla, 'en' for English, or 'mixed' for code-switched.
        """
        query = query.strip()
        if not query:
            return 'en'  # Default to English
        
        # Count Bangla and English characters
        has_bangla = self.contains_bangla_characters(query)
        
        # Simple heuristic: if contains Bangla characters, likely Bangla
        # If only ASCII/English, likely English
        # If both, code-switched
        
        # Count words
        words = re.findall(r'\b\w+\b', query.lower())
        bangla_words = [w for w in query.split() if self.contains_bangla_characters(w)]
        english_words = [w for w in words if w in self.common_english_words]
        
        # Decision logic
        if has_bangla:
            # Check for code-switching
            english_only_words = [w for w in words if not self.contains_bangla_characters(w) and len(w) > 2]
            if len(english_only_words) > len(bangla_words) * 0.5:
                return 'mixed'  # Code-switched
            return 'bn'
        else:
            return 'en'


class QueryNormalizer:
    """Normalizes queries using XLM-RoBERTa or simple normalization."""
    
    def __init__(self, stopwords_dir: str = ".", use_bert_tokenizer: bool = True):
        """
        Initialize normalizer with stopwords from files.
        
        Args:
            stopwords_dir: Directory containing stopword files (default: current directory)
            use_bert_tokenizer: If True, use BanglaBERT for Bangla and BERT for English (default: True)
        """
        # Load English stopwords from file
        self.english_stopwords = self._load_english_stopwords(stopwords_dir)
        
        # Load Bangla stopwords from file
        self.bangla_stopwords = self._load_bangla_stopwords(stopwords_dir)
        
        # Initialize BERT tokenizers if requested and available
        self.use_bert_tokenizer = use_bert_tokenizer and USE_BERT_TOKENIZER
        self.bert_tokenizer = None
        
        if self.use_bert_tokenizer:
            try:
                self.bert_tokenizer = get_tokenizer(
                    bangla_model="sagorsarker/bangla-bert-base",
                    english_model="bert-base-uncased"
                )
                print("✓ Using BanglaBERT for Bangla and BERT for English normalization")
            except Exception as e:
                print(f"Warning: Failed to load BERT tokenizers: {e}")
                print("Falling back to simple normalization")
                self.use_bert_tokenizer = False
    
    def _load_english_stopwords(self, dir_path: str) -> Set[str]:
        """Load English stopwords from english_stopwords.txt"""
        stopwords_file = os.path.join(dir_path, 'english_stopwords.txt')
        stopwords = set()
        
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word:  # Skip empty lines
                        stopwords.add(word)
            print(f"Loaded {len(stopwords)} English stopwords from {stopwords_file}")
        except FileNotFoundError:
            print(f"Warning: {stopwords_file} not found. No default stopwords loaded.")
            stopwords = set()
        
        return stopwords
    
    def _load_bangla_stopwords(self, dir_path: str) -> Set[str]:
        """Load Bangla stopwords from bangla_stopwords.csv"""
        stopwords_file = os.path.join(dir_path, 'bangla_stopwords.csv')
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
            print(f"Warning: {stopwords_file} not found. No default stopwords loaded.")
            stopwords = set()
        except Exception as e:
            print(f"Warning: Error loading {stopwords_file}: {e}. No default stopwords loaded.")
            stopwords = set()
        
        return stopwords
    
    def normalize(self, query: str, language: str, remove_stopwords: bool = False) -> str:
        """
        Normalize query using BanglaBERT for Bangla and BERT for English, or simple normalization.
        Optionally remove stopwords.
        """
        if self.use_bert_tokenizer and self.bert_tokenizer:
            # Use BERT normalization (BanglaBERT for Bangla, BERT for English)
            normalized = self.bert_tokenizer.normalize(query, language)
            
            # Remove stopwords if requested
            if remove_stopwords:
                # Tokenize to get words for stopword removal
                tokens = self.bert_tokenizer.tokenize_and_normalize(normalized, language, remove_special_tokens=True)
                
                # Filter stopwords based on language
                if language == 'en':
                    tokens = [t for t in tokens if t.lower() not in self.english_stopwords]
                elif language == 'bn':
                    tokens = [t for t in tokens if t not in self.bangla_stopwords]
                elif language == 'mixed':
                    # For mixed, check each token
                    filtered_tokens = []
                    for t in tokens:
                        if self.contains_bangla_characters(t):
                            if t not in self.bangla_stopwords:
                                filtered_tokens.append(t)
                        else:
                            if t.lower() not in self.english_stopwords:
                                filtered_tokens.append(t)
                    tokens = filtered_tokens
                
                # Join tokens back to string
                normalized = ' '.join(tokens)
            
            return normalized
        else:
            # Fallback to simple normalization
            query = re.sub(r'\s+', ' ', query.strip())
            
            # For English: lowercase
            if language == 'en':
                query = query.lower()
                
                if remove_stopwords:
                    words = query.split()
                    words = [w for w in words if w not in self.english_stopwords]
                    query = ' '.join(words)
            
            # For Bangla: keep original case (Bangla doesn't have case)
            elif language == 'bn':
                if remove_stopwords:
                    words = query.split()
                    words = [w for w in words if w not in self.bangla_stopwords]
                    query = ' '.join(words)
            
            # For mixed: lowercase English parts, keep Bangla as is
            elif language == 'mixed':
                # Split and process each word
                words = query.split()
                processed = []
                for word in words:
                    if self.contains_bangla_characters(word):
                        processed.append(word)  # Keep Bangla as is
                    else:
                        processed.append(word.lower())  # Lowercase English
                query = ' '.join(processed)
            
            return query
    
    def contains_bangla_characters(self, text: str) -> bool:
        """Check if text contains Bangla characters."""
        bangla_range = (0x0980, 0x09FF)
        for char in text:
            if bangla_range[0] <= ord(char) <= bangla_range[1]:
                return True
        return False


class QueryTranslator:
    """
    Translates queries between Bangla and English.
    Uses free translation APIs (Google Translate API, DeepL, or others).
    For this implementation, we'll use a simple dictionary-based approach
    with fallback to a free API if available.
    """
    
    def __init__(self):
        # Translation dictionary - relying on NLLB model
        self.translation_dict = {}
        
        # For production, you could use:
        # - googletrans library (free, but rate-limited)
        # - DeepL API (has free tier)
        # - MyMemory Translation API (free)
        # - OPUS-MT models from HuggingFace (local, free)
        
        # Initialize NLLB model
        self.use_nllb = False
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading NLLB-200 translation model...")
                self.model_name = "facebook/nllb-200-distilled-600M"
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model.eval()
                self.use_nllb = True
                print(f"✓ Loaded NLLB-200 model on {self.device}")
            except Exception as e:
                print(f"Warning: Failed to load NLLB model: {e}")
                print("Falling back to dictionary translation.")
        else:
             print("Transformers not available. Using dictionary translation.")
    
    def translate(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Translate query from source_lang to target_lang.
        Returns translated query.
        """
        if source_lang == target_lang:
            return query
        
        # Check dictionary first for common terms (faster and accurate for specific domains)
        if query.lower() in self.translation_dict:
            return self.translation_dict[query.lower()]
            
        # NLLB Translation
        if self.use_nllb:
            try:
                # Map language codes to NLLB codes
                # ben_Beng: Bangla
                # eng_Latn: English
                src_code = "ben_Beng" if source_lang == "bn" else "eng_Latn"
                tgt_code = "ben_Beng" if target_lang == "bn" else "eng_Latn"
                
                # Set source and target languages
                self.tokenizer.src_lang = src_code
                
                # Tokenize
                inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
                
                # Get target language token ID
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
                
                # Generate translation
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=30
                    )
                
                # Decode
                translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                return translation
            except Exception as e:
                print(f"Translation error: {e}")
                # Fallback to dictionary/simple methods below
        
        # Try word-by-word translation for simple queries (Fallback)
        if source_lang == 'en' and target_lang == 'bn':
            return self._translate_en_to_bn(query)
        elif source_lang == 'bn' and target_lang == 'en':
            return self._translate_bn_to_en(query)
        else:
            # Return original if can't translate
            return query
    
    def _translate_en_to_bn(self, query: str) -> str:
        """Simple dictionary-based EN to BN translation."""
        words = query.lower().split()
        translated = []
        
        for word in words:
            if word in self.translation_dict:
                translated.append(self.translation_dict[word])
            else:
                # Keep original word if no translation found
                # In a real system, this would call a translation API
                translated.append(word)
        
        return ' '.join(translated)
    
    def _translate_bn_to_en(self, query: str) -> str:
        """Simple dictionary-based BN to EN translation."""
        words = query.split()
        translated = []
        
        for word in words:
            if word in self.translation_dict:
                translated.append(self.translation_dict[word])
            else:
                # Keep original if no translation found
                translated.append(word)
        
        return ' '.join(translated)
    
    def translate_using_api(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Translate using an external API.
        This is a placeholder - you can implement using googletrans, DeepL, etc.
        """
        # Example using requests (if you have access to a free translation API)
        # For now, fall back to dictionary-based translation
        return self.translate(query, source_lang, target_lang)


class QueryExpander:
    """
    Expands queries with synonyms and morphological variants.
    """
    
    def __init__(self):
        # Synonym dictionary - empty, can be populated from external sources if needed
        self.synonyms = {}
    
    def expand(self, query: str, language: str) -> List[str]:
        """
        Expand query with synonyms.
        Returns list of expanded query terms.
        """
        query_terms = query.split()
        expanded_terms = set(query_terms)  # Start with original terms
        
        for term in query_terms:
            term_lower = term.lower()
            if term_lower in self.synonyms:
                expanded_terms.update(self.synonyms[term_lower])
        
        return list(expanded_terms)


class NamedEntityMapper:
    """
    Extracts named entities and maps them across languages.
    Important for proper noun matching (e.g., "Bangladesh" → "বাংলাদেশ").
    """
    
    def __init__(self):
        # Named entity mapping dictionary - relying on NLLB translation
        self.ne_mappings = {}
    
    def extract_and_map(self, query: str, source_lang: str, target_lang: str) -> Dict[str, str]:
        """
        Extract named entities from query and return mappings.
        Returns dictionary: {original_ne: mapped_ne}
        """
        mappings = {}
        query_lower = query.lower()
        
        # Check for named entities in the mapping dictionary
        for ne, mapped_ne in self.ne_mappings.items():
            if ne in query_lower:
                # Determine if this NE needs mapping
                if source_lang == 'en' and ne.isascii():
                    if mapped_ne not in mappings:
                        mappings[ne] = mapped_ne
                elif source_lang == 'bn':
                    # Check if mapped_ne is in English
                    if mapped_ne.isascii():
                        mappings[ne] = mapped_ne
        
        # Also try word-by-word matching
        words = query.split()
        for word in words:
            word_lower = word.lower()
            if word_lower in self.ne_mappings:
                mapped = self.ne_mappings[word_lower]
                if source_lang == 'en' and not mapped.isascii():
                    mappings[word] = mapped
                elif source_lang == 'bn' and mapped.isascii():
                    mappings[word] = mapped
        
        return mappings
    
    def map_query_nes(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Map named entities in query to target language.
        """
        mappings = self.extract_and_map(query, source_lang, target_lang)
        mapped_query = query
        
        for original, mapped in mappings.items():
            mapped_query = mapped_query.replace(original, mapped)
        
        return mapped_query


class QueryProcessor:
    """
    Main query processing pipeline that integrates all components.
    """
    
    def __init__(self, stopwords_dir: str = None):
        """
        Initialize query processor.
        
        Args:
            stopwords_dir: Directory containing stopword files. 
                          If None, uses parent directory of Module_B.
        """
        if stopwords_dir is None:
            # Default to parent directory (where stopword files are located)
            stopwords_dir = os.path.dirname(os.path.dirname(__file__))
        
        self.language_detector = LanguageDetector()
        self.normalizer = QueryNormalizer(stopwords_dir)
        self.translator = QueryTranslator()
        self.expander = QueryExpander()
        self.ne_mapper = NamedEntityMapper()
    
    def process(self, query: str, 
                target_languages: Optional[List[str]] = None,
                remove_stopwords: bool = False,
                expand: bool = True,
                map_nes: bool = True) -> Dict:
        """
        Process a query through the complete pipeline.
        
        Args:
            query: Input query string
            target_languages: List of target languages to retrieve from (default: both bn and en)
            remove_stopwords: Whether to remove stopwords during normalization
            expand: Whether to expand query with synonyms
            map_nes: Whether to map named entities
        
        Returns:
            Dictionary with processed query information:
            {
                'original_query': str,
                'detected_language': str,
                'normalized_query': str,
                'target_queries': {
                    'bn': str,  # Query for Bangla documents
                    'en': str   # Query for English documents
                },
                'expanded_terms': List[str],
                'named_entities': Dict[str, str],
                'processing_time': float
            }
        """
        start_time = time.time()
        
        # Step 1: Language Detection
        detected_lang = self.language_detector.detect(query)
        
        # Step 2: Normalization
        normalized_query = self.normalizer.normalize(query, detected_lang, remove_stopwords)
        
        # Step 3: Determine target languages
        if target_languages is None:
            target_languages = ['bn', 'en']
        
        # Step 4: Query Translation & Named Entity Mapping
        target_queries = {}
        for target_lang in target_languages:
            if detected_lang == target_lang or detected_lang == 'mixed':
                # Same language or mixed - use normalized query
                target_query = normalized_query
            else:
                # Translate to target language
                # First, map named entities if enabled
                if map_nes:
                    query_with_mapped_nes = self.ne_mapper.map_query_nes(
                        normalized_query, detected_lang, target_lang
                    )
                    target_query = self.translator.translate(
                        query_with_mapped_nes, detected_lang, target_lang
                    )
                else:
                    target_query = self.translator.translate(
                        normalized_query, detected_lang, target_lang
                    )
            
            target_queries[target_lang] = target_query
        
        # Step 5: Query Expansion (optional)
        expanded_terms = []
        if expand:
            expanded_terms = self.expander.expand(normalized_query, detected_lang)
        
        # Step 6: Extract Named Entities
        named_entities = {}
        if map_nes:
            for target_lang in target_languages:
                if target_lang != detected_lang:
                    nes = self.ne_mapper.extract_and_map(query, detected_lang, target_lang)
                    named_entities.update(nes)
        
        processing_time = (time.time() - start_time) * 1000  # in milliseconds
        
        return {
            'original_query': query,
            'detected_language': detected_lang,
            'normalized_query': normalized_query,
            'target_queries': target_queries,
            'expanded_terms': expanded_terms,
            'named_entities': named_entities,
            'processing_time': processing_time
        }
    
    def process_for_retrieval(self, query: str) -> Tuple[str, List[str]]:
        """
        Process query and return language-specific queries for retrieval.
        Returns: (detected_language, [list of queries for each target language])
        """
        result = self.process(query, expand=True, map_nes=True)
        queries = list(result['target_queries'].values())
        return result['detected_language'], queries


def main():
    """Test the query processor with sample queries."""
    print("=" * 60)
    print("Query Processor Test")
    print("=" * 60)
    
    processor = QueryProcessor()
    
    # Test queries
    test_queries = [
        "bangladesh election",
        "বাংলাদেশ নির্বাচন",
        "dhaka government",
        "ঢাকা সরকার",
        "education policy",
        "শিক্ষা নীতি",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"{'='*60}")
        
        result = processor.process(query)
        
        print(f"Detected Language: {result['detected_language']}")
        print(f"Normalized Query: {result['normalized_query']}")
        print(f"Target Queries:")
        for lang, target_query in result['target_queries'].items():
            lang_name = "Bangla" if lang == "bn" else "English"
            print(f"  {lang_name}: {target_query}")
        print(f"Expanded Terms: {result['expanded_terms']}")
        print(f"Named Entities: {result['named_entities']}")
        print(f"Processing Time: {result['processing_time']:.2f} ms")
    
    print("\n" + "=" * 60)
    print("Query Processing Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

