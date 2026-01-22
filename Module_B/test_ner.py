#!/usr/bin/env python3
"""
NER Testing Script
Tests the NamedEntityMapper class with Bangla and English sentences.
"""

import sys
import os

# Add the parent directory to the path so we can resolve imports relative to root if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from query_processor import NamedEntityMapper
except ImportError:
    # If running from root, try module import
    from Module_B.query_processor import NamedEntityMapper

def test_ner():
    """Test NER functionality with various examples."""

    print("=" * 70)
    print("NAMED ENTITY RECOGNITION TESTING SCRIPT")
    print("=" * 70)

    # Initialize NER mapper
    print("Loading NER models...")
    ner_mapper = NamedEntityMapper()
    print("Models loaded.")

    # Test cases: Bangla sentences
    bangla_sentences = [
        "শেখ হাসিনা বাংলাদেশের প্রধানমন্ত্রী ছিলেন।",
        "আমি ঢাকায় থাকি।",
        "রহিম সাহেব গ্রামীণ ব্যাংকে কাজ করেন।",
        "কাজী নজরুল ইসলাম আমাদের জাতীয় কবি।",
        "শাকিব আল হাসান ক্রিকেট খেলেন।",
        "বাংলাদেশ একটি সুন্দর দেশ।"
    ]

    # Test cases: English sentences
    english_sentences = [
        "Joe Biden is the president of USA.",
        "I live in New York City.",
        "Elon Musk is the CEO of Tesla and SpaceX.",
        "Google has its headquarters in Mountain View.",
        "Lionel Messi plays for Inter Miami.",
        "The United Nations was established in 1945."
    ]

    print("\n" + "=" * 70)
    print("BANGLA NER TESTS")
    print("=" * 70)

    for sentence in bangla_sentences:
        print(f"\nSentence: {sentence}")
        try:
            entities = ner_mapper.extract_entities(sentence)
            if entities:
                print(f"Found {len(entities)} entities:")
                for ent in entities:
                    print(f"  - Entity: {ent['entity']:20} Type: {ent['type']:10} Confidence: {ent['confidence']:.4f}")
            else:
                print("  No entities found.")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("ENGLISH NER TESTS")
    print("=" * 70)

    for sentence in english_sentences:
        print(f"\nSentence: {sentence}")
        try:
            entities = ner_mapper.extract_entities(sentence)
            if entities:
                print(f"Found {len(entities)} entities:")
                for ent in entities:
                    print(f"  - Entity: {ent['entity']:20} Type: {ent['type']:10} Confidence: {ent['confidence']:.4f}")
            else:
                print("  No entities found.")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("NER TESTING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_ner()