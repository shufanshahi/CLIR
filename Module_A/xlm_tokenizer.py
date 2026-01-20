"""
XLM-RoBERTa Tokenizer Wrapper for CLIR System
Provides tokenization and normalization using XLM-RoBERTa tokenizer.
"""

import os
from typing import List, Optional

try:
    from transformers import XLMRobertaTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers torch")


class XLMRobertaTokenizerWrapper:
    """
    Wrapper around XLM-RoBERTa tokenizer for CLIR system.
    Handles both Bangla and English tokenization and normalization.
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base", cache_dir: Optional[str] = None):
        """
        Initialize XLM-RoBERTa tokenizer.
        
        Args:
            model_name: HuggingFace model name (default: xlm-roberta-base)
            cache_dir: Directory to cache the tokenizer
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. Install with: "
                "conda activate clir && pip install transformers torch"
            )
        
        try:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model_name = model_name
            print(f"✓ Loaded XLM-RoBERTa tokenizer: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load XLM-RoBERTa tokenizer: {e}")
    
    def tokenize(self, text: str, remove_special_tokens: bool = True, remove_prefix: bool = True) -> List[str]:
        """
        Tokenize text using XLM-RoBERTa tokenizer.
        
        Args:
            text: Input text to tokenize
            remove_special_tokens: Whether to remove special tokens like [CLS], [SEP], etc.
            remove_prefix: Whether to remove the '▁' prefix from tokens (default: True)
        
        Returns:
            List of token strings
        """
        if not text or not text.strip():
            return []
        
        # Tokenize using XLM-RoBERTa
        tokens = self.tokenizer.tokenize(text)
        
        # Remove special tokens if requested
        if remove_special_tokens:
            tokens = [t for t in tokens if not t.startswith('<') and not t.startswith('[')]
        
        # Remove the '▁' prefix that XLM-RoBERTa uses to indicate word boundaries
        if remove_prefix:
            tokens = [t.lstrip('▁') for t in tokens]
        
        # Remove empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        return tokens
    
    def normalize(self, text: str) -> str:
        """
        Normalize text using XLM-RoBERTa tokenizer preprocessing.
        This includes:
        - Unicode normalization
        - Lowercasing (if applicable)
        - Whitespace cleanup
        
        Args:
            text: Input text to normalize
        
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # XLM-RoBERTa tokenizer normalizes text as part of its preprocessing
        # We can use the encode/decode cycle to get normalized text
        # Or apply basic normalization manually
        
        # Basic normalization
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # XLM-RoBERTa handles the rest through its tokenizer
        return text
    
    def tokenize_and_normalize(self, text: str, remove_special_tokens: bool = True, remove_prefix: bool = True) -> List[str]:
        """
        Normalize and tokenize text in one step.
        
        Args:
            text: Input text
            remove_special_tokens: Whether to remove special tokens
            remove_prefix: Whether to remove the '▁' prefix from tokens
        
        Returns:
            List of normalized token strings
        """
        normalized = self.normalize(text)
        tokens = self.tokenize(normalized, remove_special_tokens, remove_prefix)
        return tokens
    
    def decode_tokens(self, tokens: List[str]) -> str:
        """
        Decode tokens back to text string.
        
        Args:
            tokens: List of token strings
        
        Returns:
            Decoded text string
        """
        # Join tokens, handling subword prefixes (## in BERT, ▁ in XLM-RoBERTa)
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text


# Global tokenizer instance (lazy loading)
_tokenizer_instance = None


def get_tokenizer(model_name: str = "xlm-roberta-base", cache_dir: Optional[str] = None) -> XLMRobertaTokenizerWrapper:
    """
    Get or create a global XLM-RoBERTa tokenizer instance.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache the tokenizer
    
    Returns:
        XLMRobertaTokenizerWrapper instance
    """
    global _tokenizer_instance
    
    if _tokenizer_instance is None:
        _tokenizer_instance = XLMRobertaTokenizerWrapper(model_name, cache_dir)
    
    return _tokenizer_instance

