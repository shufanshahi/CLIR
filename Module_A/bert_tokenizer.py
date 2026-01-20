"""
BERT Tokenizer Wrapper for CLIR System
Provides tokenization and normalization using BanglaBERT for Bangla and BERT for English.
"""

import os
from typing import List, Optional

try:
    from transformers import AutoTokenizer, BertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers torch")


class BERTTokenizerWrapper:
    """
    Wrapper around BERT tokenizers for CLIR system.
    Uses BanglaBERT for Bangla and BERT-base-uncased for English.
    """
    
    def __init__(self, 
                 bangla_model: str = "sagorsarker/bangla-bert-base",
                 english_model: str = "bert-base-uncased",
                 cache_dir: Optional[str] = None):
        """
        Initialize BERT tokenizers for both languages.
        
        Args:
            bangla_model: HuggingFace model name for BanglaBERT (default: sagorsarker/bangla-bert-base)
            english_model: HuggingFace model name for English BERT (default: bert-base-uncased)
            cache_dir: Directory to cache the tokenizers
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. Install with: "
                "conda activate clir && pip install transformers torch"
            )
        
        try:
            # Load BanglaBERT tokenizer
            self.bangla_tokenizer = AutoTokenizer.from_pretrained(
                bangla_model,
                cache_dir=cache_dir
            )
            self.bangla_model_name = bangla_model
            print(f"✓ Loaded BanglaBERT tokenizer: {bangla_model}")
        except Exception as e:
            raise RuntimeError(f"Failed to load BanglaBERT tokenizer: {e}")
        
        try:
            # Load English BERT tokenizer
            self.english_tokenizer = AutoTokenizer.from_pretrained(
                english_model,
                cache_dir=cache_dir
            )
            self.english_model_name = english_model
            print(f"✓ Loaded English BERT tokenizer: {english_model}")
        except Exception as e:
            raise RuntimeError(f"Failed to load English BERT tokenizer: {e}")
    
    def tokenize(self, text: str, language: str = "en", remove_special_tokens: bool = True) -> List[str]:
        """
        Tokenize text using appropriate BERT tokenizer based on language.
        
        Args:
            text: Input text to tokenize
            language: Language code ('bn' for Bangla, 'en' for English)
            remove_special_tokens: Whether to remove special tokens like [CLS], [SEP], etc.
        
        Returns:
            List of token strings
        """
        if not text or not text.strip():
            return []
        
        # Select appropriate tokenizer based on language
        if language == 'bn':
            tokenizer = self.bangla_tokenizer
        else:
            tokenizer = self.english_tokenizer
        
        # Tokenize using the appropriate tokenizer
        tokens = tokenizer.tokenize(text)
        
        # Remove special tokens if requested
        if remove_special_tokens:
            # Remove tokens that start with [ or are in special tokens
            special_tokens = set(tokenizer.all_special_tokens)
            tokens = [t for t in tokens if t not in special_tokens and not t.startswith('[')]
        
        # Remove the '##' prefix that BERT uses for subword tokens
        # This helps with matching full words
        tokens = [t.replace('##', '') if t.startswith('##') else t for t in tokens]
        
        # Remove empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        # Clean up tokens: remove single characters, punctuation, and unintelligible tokens
        tokens = self._clean_tokens(tokens, language)
        
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
        import re
        cleaned = []
        
        for token in tokens:
            # Skip empty tokens
            if not token or not token.strip():
                continue
            
            # Remove single character tokens (except digits which might be meaningful)
            if len(token) == 1:
                # Keep single digits as they might be meaningful
                if token.isdigit():
                    cleaned.append(token)
                # Skip single punctuation/letters
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
            
            # Remove very short tokens (1-2 chars) unless they're digits or common abbreviations
            if len(token) <= 2:
                # Keep if it's all digits
                if token.isdigit():
                    cleaned.append(token)
                    continue
                # Skip very short tokens that are likely fragments
                # (except common ones, but we'll be conservative)
                continue
            
            cleaned.append(token)
        
        return cleaned
    
    def normalize(self, text: str, language: str = "en") -> str:
        """
        Normalize text using BERT tokenizer preprocessing.
        This includes:
        - Unicode normalization
        - Lowercasing (for English BERT)
        - Whitespace cleanup
        
        Args:
            text: Input text to normalize
            language: Language code ('bn' for Bangla, 'en' for English)
        
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Basic normalization
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # For English, BERT-base-uncased automatically lowercases
        # For Bangla, keep original case
        if language == 'en':
            text = text.lower()
        
        return text
    
    def tokenize_and_normalize(self, text: str, language: str = "en", remove_special_tokens: bool = True) -> List[str]:
        """
        Normalize and tokenize text in one step.
        
        Args:
            text: Input text
            language: Language code ('bn' for Bangla, 'en' for English)
            remove_special_tokens: Whether to remove special tokens
        
        Returns:
            List of normalized token strings
        """
        normalized = self.normalize(text, language)
        tokens = self.tokenize(normalized, language, remove_special_tokens)
        return tokens
    
    def decode_tokens(self, tokens: List[str], language: str = "en") -> str:
        """
        Decode tokens back to text string.
        
        Args:
            tokens: List of token strings
            language: Language code ('bn' for Bangla, 'en' for English)
        
        Returns:
            Decoded text string
        """
        # Select appropriate tokenizer based on language
        if language == 'bn':
            tokenizer = self.bangla_tokenizer
        else:
            tokenizer = self.english_tokenizer
        
        # Decode tokens, handling subword prefixes (## in BERT)
        # Reconstruct subword tokens
        reconstructed_tokens = []
        for token in tokens:
            # If this is not the first token and doesn't start with ##, it's a new word
            if reconstructed_tokens and not token.startswith('##'):
                reconstructed_tokens.append('##' + token)
            else:
                reconstructed_tokens.append(token)
        
        text = tokenizer.convert_tokens_to_string(reconstructed_tokens)
        return text


# Global tokenizer instance (lazy loading)
_tokenizer_instance = None


def get_tokenizer(bangla_model: str = "sagorsarker/bangla-bert-base",
                  english_model: str = "bert-base-uncased",
                  cache_dir: Optional[str] = None) -> BERTTokenizerWrapper:
    """
    Get or create a global BERT tokenizer instance.
    
    Args:
        bangla_model: HuggingFace model name for BanglaBERT
        english_model: HuggingFace model name for English BERT
        cache_dir: Directory to cache the tokenizers
    
    Returns:
        BERTTokenizerWrapper instance
    """
    global _tokenizer_instance
    
    if _tokenizer_instance is None:
        _tokenizer_instance = BERTTokenizerWrapper(bangla_model, english_model, cache_dir)
    
    return _tokenizer_instance

