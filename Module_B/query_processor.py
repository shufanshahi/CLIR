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
        words = query.split()
        bangla_words = [w for w in words if self.contains_bangla_characters(w)]
        english_words = [w for w in words if not self.contains_bangla_characters(w) and len(w) > 0]
        
        # Decision logic
        if has_bangla:
            # Check for code-switching (has both Bangla and English words)
            if len(english_words) > 0 and len(bangla_words) > 0:
                if len(english_words) > len(bangla_words) * 0.5:
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
    Uses different models for different translation directions:
    - Bangla to English: facebook/nllb-200-distilled-600M
    - English to Bangla: Helsinki-NLP/opus-mt-en-mt
    """

    def __init__(self):
        # Initialize models
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.opus_model = None
        self.opus_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if TRANSFORMERS_AVAILABLE:
            self._load_models()
        else:
            print("Transformers not available. Using dictionary translation.")

    def _load_models(self):
        """Load the translation models."""
        try:
            # Load NLLB model for both directions (it supports English and Bangla)
            print("Loading NLLB-200 model for bidirectional English-Bangla translation...")
            nllb_model_name = "facebook/nllb-200-distilled-600M"
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to(self.device)
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
            self.nllb_model.eval()
            print(f"✓ Loaded NLLB-200 model on {self.device}")
        except Exception as e:
            print(f"Warning: Failed to load NLLB model: {e}")

        # Note: Using NLLB for both directions since OPUS-MT en-bn model doesn't exist
        # NLLB supports both English and Bangla well
    
    def translate(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Translate query from source_lang to target_lang.
        Uses NLLB-200 model for both directions (English ↔ Bangla).
        """
        if source_lang == target_lang:
            return query

        # Use NLLB for translation
        return self._translate_with_nllb(query, source_lang, target_lang)

    def _translate_with_nllb(self, query: str, source_lang: str, target_lang: str) -> str:
        """Translate using NLLB-200 model (supports both English and Bangla)."""
        if not self.nllb_model or not self.nllb_tokenizer:
            return self._fallback_translation(query, source_lang, target_lang)

        try:
            # Map language codes to NLLB codes
            src_code = "ben_Beng" if source_lang == "bn" else "eng_Latn"
            tgt_code = "ben_Beng" if target_lang == "bn" else "eng_Latn"

            # Set source language
            self.nllb_tokenizer.src_lang = src_code

            # Tokenize
            inputs = self.nllb_tokenizer(query, return_tensors="pt").to(self.device)

            # Get target language token ID
            forced_bos_token_id = self.nllb_tokenizer.convert_tokens_to_ids(tgt_code)

            # Generate translation
            with torch.no_grad():
                generated_tokens = self.nllb_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=30
                )

            # Decode
            translation = self.nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation

        except Exception as e:
            print(f"NLLB translation error: {e}")
            return self._fallback_translation(query, source_lang, target_lang)

    def _fallback_translation(self, query: str, source_lang: str, target_lang: str) -> str:
        """Fallback translation when models are not available."""
        # Try word-by-word translation for simple queries
        if source_lang == 'en' and target_lang == 'bn':
            return self._translate_en_to_bn(query)
        elif source_lang == 'bn' and target_lang == 'en':
            return self._translate_bn_to_en(query)
        else:
            # Return original if can't translate
            return query
    
    def _translate_en_to_bn(self, query: str) -> str:
        """Fallback translation when model is unavailable."""
        return query
    
    def _translate_bn_to_en(self, query: str) -> str:
        """Fallback translation when model is unavailable."""
        return query
    
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
    Expands queries with semantically similar words using language models.
    Uses BanglaBERT for Bangla and BERT for English.
    """

    def __init__(self):
        # Initialize language models for semantic similarity
        self.bangla_model = None
        self.bangla_tokenizer = None
        self.english_model = None
        self.english_tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        self._load_models()

        # Cache for similar words to avoid recomputation
        self.similarity_cache = {}

    def _load_models(self):
        """Load language models for semantic similarity."""
        try:
            from transformers import AutoModel, AutoTokenizer

            # Load BanglaBERT for Bangla semantic similarity
            print("Loading BanglaBERT for Bangla query expansion...")
            bangla_model_name = "sagorsarker/bangla-bert-base"
            self.bangla_model = AutoModel.from_pretrained(bangla_model_name).to(self.device)
            self.bangla_tokenizer = AutoTokenizer.from_pretrained(bangla_model_name)
            self.bangla_model.eval()
            print("✓ Loaded BanglaBERT model")
        except Exception as e:
            print(f"Warning: Failed to load BanglaBERT: {e}")

        try:
            # Load BERT for English semantic similarity
            print("Loading BERT for English query expansion...")
            english_model_name = "bert-base-uncased"
            self.english_model = AutoModel.from_pretrained(english_model_name).to(self.device)
            self.english_tokenizer = AutoTokenizer.from_pretrained(english_model_name)
            self.english_model.eval()
            print("✓ Loaded BERT model")
        except Exception as e:
            print(f"Warning: Failed to load BERT: {e}")

    def _get_word_embedding(self, word: str, language: str) -> torch.Tensor:
        """Get word embedding using appropriate language model."""
        if language == 'bn' and self.bangla_model and self.bangla_tokenizer:
            inputs = self.bangla_tokenizer(word, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.bangla_model(**inputs)
            # Use mean pooling of token embeddings
            return outputs.last_hidden_state.mean(dim=1).squeeze()
        elif language == 'en' and self.english_model and self.english_tokenizer:
            inputs = self.english_tokenizer(word, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.english_model(**inputs)
            # Use mean pooling of token embeddings
            return outputs.last_hidden_state.mean(dim=1).squeeze()
        else:
            return None

    def _find_similar_words(self, word: str, language: str, top_k: int = 5) -> List[str]:
        """Find semantically similar words using language models."""
        # Check cache first
        cache_key = f"{word}_{language}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Get word embedding
        word_embedding = self._get_word_embedding(word, language)
        if word_embedding is None:
            return [word]  # Return original word if no model available

        # For demonstration, we'll use a simple vocabulary expansion approach
        # In a full implementation, you'd have a vocabulary to compare against
        similar_words = [word]  # Start with original word

        # Basic morphological expansion (simplified)
        if language == 'en':
            # English morphological variants
            if word.endswith('tion'):
                similar_words.append(word.replace('tion', 'ing'))
                similar_words.append(word.replace('tion', 'ment'))
            elif word.endswith('ing'):
                similar_words.append(word.replace('ing', 'tion'))
                similar_words.append(word.replace('ing', 'er'))
            elif word.endswith('ment'):
                similar_words.append(word.replace('ment', 'tion'))

            # Common synonyms (basic set)
            basic_synonyms = {
                'election': ['vote', 'poll', 'voting'],
                'education': ['school', 'learning', 'teaching'],
                'government': ['administration', 'authority'],
                'policy': ['strategy', 'plan'],
                'health': ['medical', 'wellness'],
                'development': ['growth', 'progress']
            }
            if word in basic_synonyms:
                similar_words.extend(basic_synonyms[word])

        elif language == 'bn':
            # Bangla morphological variants (simplified)
            # This is a very basic implementation - in practice you'd need more sophisticated rules
            if word.endswith('ন'):
                similar_words.append(word.replace('ন', 'ণী'))  # election -> electoral
            elif word.endswith('া'):
                similar_words.append(word + 'র')  # Add agent suffix

            # Common Bangla synonyms (basic set)
            basic_synonyms_bn = {
                'নির্বাচন': ['ভোট', 'নির্বাচনী'],
                'শিক্ষা': ['পড়াশোনা', 'লেখাপড়া'],
                'সরকার': ['প্রশাসন', 'রাষ্ট্র'],
                'নীতি': ['কৌশল', 'পরিকল্পনা']
            }
            if word in basic_synonyms_bn:
                similar_words.extend(basic_synonyms_bn[word])

        # Remove duplicates and limit to top_k
        similar_words = list(set(similar_words))[:top_k]

        # Cache result
        self.similarity_cache[cache_key] = similar_words

        return similar_words

    def expand(self, query: str, language: str) -> List[str]:
        """
        Expand query with semantically similar words using language models.
        Returns list of expanded query terms.
        """
        query_terms = query.split()
        expanded_terms = set(query_terms)  # Start with original terms

        # Find similar words for each term
        for term in query_terms:
            similar_words = self._find_similar_words(term, language, top_k=3)
            expanded_terms.update(similar_words)

        return list(expanded_terms)


class NamedEntityMapper:
    """
    Extracts named entities and maps them across languages.
    Important for proper noun matching (e.g., "Bangladesh" → "বাংলাদেশ").
    """

    def __init__(self):
        # Initialize NER models for different languages
        self.ner_pipelines = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load language-specific NER models
        self._load_ner_models()

    def _load_ner_models(self):
        """Load NER models for different languages."""
        try:
            from transformers import pipeline

            # English NER model
            print("Loading English NER model...")
            self.ner_pipelines['en'] = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                tokenizer="dslim/bert-base-NER",
                grouped_entities=True,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ Loaded English NER model")
        except Exception as e:
            print(f"Warning: Failed to load English NER model: {e}")

        try:
            # Bangla NER model
            print("Loading Bangla NER model...")
            self.ner_pipelines['bn'] = pipeline(
                "ner",
                model="sagorsarker/mbert-bengali-ner",
                tokenizer="sagorsarker/mbert-bengali-ner",
                grouped_entities=True,
                device=0 if torch.cuda.is_available() else -1
            )
            print("✓ Loaded Bangla NER model")
        except Exception as e:
            print(f"Warning: Failed to load Bangla NER model: {e}")

        if not self.ner_pipelines:
            print("No NER models loaded. Named entity extraction will be limited.")

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text using language-specific NER models.
        """
        # Detect language
        detector = LanguageDetector()
        language = detector.detect(text)
        if language == 'mixed':
            language = 'en'  # Default to English for mixed content

        # Get appropriate NER pipeline
        ner_pipeline = self.ner_pipelines.get(language)
        if not ner_pipeline:
            return []

        try:
            # For English, capitalize the text to improve NER detection
            if language == 'en':
                text = text.title()  # Capitalize first letter of each word

            # Run NER
            entities = ner_pipeline(text)

            # Filter and format entities
            filtered_entities = []
            for entity in entities:
                # Filter criteria:
                # - Reasonable confidence (> 0.7 for English, > 0.5 for Bangla as it might be less accurate)
                confidence_threshold = 0.5 if language == 'bn' else 0.7
                word = entity['word'].strip()

                # For Bangla NER, accept LABEL_X entities (except LABEL_0 which is usually 'O')
                # For English NER, use proper entity types
                entity_label = entity.get('entity_group', entity.get('entity', ''))
                if language == 'bn':
                    # Bangla model uses generic LABEL_X
                    # LABEL_0 is typically 'O' (Outside entity), so we exclude it
                    is_valid_entity = entity_label.startswith('LABEL_') and entity_label != 'LABEL_0'
                else:
                    # English model uses proper BIO tags or aggregated tags
                    # Standard tags: PER, LOC, ORG, MISC
                    is_valid_entity = entity_label in ['PER', 'LOC', 'ORG', 'MISC', 
                                                     'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 
                                                     'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']

                # Check if word contains valid characters (allow spaces for multi-word entities)
                clean_word = word.replace(' ', '')
                is_valid_word = len(clean_word) > 0 and not clean_word.isdigit()

                if (entity['score'] > confidence_threshold and
                    len(word) > 1 and
                    is_valid_word and
                    is_valid_entity):

                    # Map entity types

                    if language == 'bn':
                        # For Bangla, use generic 'ENTITY' type since labels are not specific
                        entity_type = 'ENTITY'
                    else:
                        # For English, extract type from BIO tag
                        entity_type = entity_label.split('-')[-1] if '-' in entity_label else entity_label

                    filtered_entities.append({
                        'entity': word,
                        'type': entity_type,
                        'confidence': entity['score']
                    })

            return filtered_entities

        except Exception as e:
            print(f"NER extraction error: {e}")
            return []

    def extract_and_map(self, query: str, source_lang: str, target_lang: str) -> Dict[str, str]:
        """
        Extract named entities from query and return mappings.
        Returns dictionary: {original_ne: mapped_ne}
        """
        mappings = {}

        # Extract entities using NER model
        entities = self.extract_entities(query)

        # For now, just return the entities found
        # In a full implementation, you would map these entities across languages
        # using translation or knowledge bases
        for entity_info in entities:
            entity = entity_info['entity']
            entity_type = entity_info['type']

            # For demonstration, we'll just note the entities
            # In production, you'd translate entities like:
            # "Bangladesh" -> "বাংলাদেশ" for Bangla queries
            # "Dhaka" -> "ঢাকা" for Bangla queries
            mappings[entity] = entity  # Placeholder - no actual mapping yet

        return mappings

    def map_query_nes(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Map named entities in query to target language.
        Currently returns the original query (placeholder implementation).
        """
        # For now, just return the original query
        # In a full implementation, this would replace entities with their translations
        return query


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

