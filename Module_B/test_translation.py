#!/usr/bin/env python3
"""
Translation Testing Script
Tests the QueryTranslator class with Bangla ↔ English translation examples.
"""

import sys
import os

# Add the parent directory to the path so we can resolve imports relative to root if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from query_processor import QueryTranslator
except ImportError:
    # If running from root, try module import
    from Module_B.query_processor import QueryTranslator

def test_translation():
    """Test translation functionality with various examples."""

    print("=" * 70)
    print("TRANSLATION TESTING SCRIPT")
    print("=" * 70)

    # Initialize translator
    translator = QueryTranslator()

    # Test cases: Bangla to English
    bangla_words = [
        "বাংলাদেশ", "ঢাকা", "নির্বাচন", "সরকার", "শিক্ষা",
        "স্বাস্থ্য", "উন্নয়ন", "মন্ত্রী", "রাষ্ট্রপতি",
        "খেলা", "খাবার", "বই", "স্কুল", "হাসপাতাল",
        "ব্যবসা", "কাজ", "সময়", "দিন", "রাত"
    ]

    # Test cases: English to Bangla
    english_words = [
        "bangladesh", "dhaka", "election", "government", "education",
        "health", "development", "minister", "president",
        "sports", "food", "book", "school", "hospital",
        "business", "work", "time", "day", "night"
    ]

    print("\nBANGLA → ENGLISH TRANSLATION")
    print("-" * 40)

    for word in bangla_words:
        translation = translator.translate(word, "bn", "en")
        print(f"{word:20} → {translation}")

    print("\nENGLISH → BANGLA TRANSLATION")
    print("-" * 40)

    for word in english_words:
        translation = translator.translate(word, "en", "bn")
        print(f"{word:20} → {translation}")

    print("\nPHRASE TRANSLATION")
    print("-" * 40)

    phrases = [
        ("বাংলাদেশ নির্বাচন", "bn", "en"),
        ("bangladesh election", "en", "bn"),
        ("ঢাকা সরকার", "bn", "en"),
        ("dhaka government", "en", "bn"),
        ("আমি স্কুলে যাই", "bn", "en"),
        ("I go to school", "en", "bn"),
        ("সরকার শিক্ষা উন্নয়ন করছে", "bn", "en"),
        ("Government is developing education", "en", "bn"),
        ("খেলা খুব ভালো", "bn", "en"),
        ("Sports is very good", "en", "bn")
    ]

    for phrase, src_lang, tgt_lang in phrases:
        translation = translator.translate(phrase, src_lang, tgt_lang)
        direction = "BN→EN" if src_lang == "bn" else "EN→BN"
        print(f"[{direction}] {phrase:35} → {translation}")

    print("\n" + "=" * 70)
    print("TRANSLATION TESTING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_translation()