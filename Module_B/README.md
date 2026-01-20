# Module B: Query Processing & Cross-Lingual Handling

This module implements the query processing pipeline for the CLIR system as required by Module B of the assignment.

## Components

### 1. Language Detection (`LanguageDetector`)
- **Purpose**: Identifies whether the query is in Bangla, English, or mixed (code-switched)
- **Method**: Unicode range detection for Bangla script (0x0980-0x09FF)
- **Features**:
  - Detects Bangla characters
  - Detects English text
  - Handles code-switched queries (mix of Bangla and English)

### 2. Query Normalization (`QueryNormalizer`)
- **Purpose**: Normalizes queries for consistent processing
- **Features**:
  - Lowercases English text
  - Removes extra whitespace
  - Optional stopword removal
  - Language-aware normalization

### 3. Query Translation (`QueryTranslator`)
- **Purpose**: Translates queries from one language to another (Required)
- **Features**:
  - Dictionary-based translation for common terms
  - Word-by-word translation
  - Supports EN ↔ BN translation
  - Extensible to use external translation APIs (Google Translate, DeepL, OPUS-MT)

### 4. Query Expansion (`QueryExpander`)
- **Purpose**: Expands queries with synonyms and related terms (Recommended)
- **Features**:
  - Synonym dictionary for common terms
  - Expands both English and Bangla queries
  - Adds morphological variants

### 5. Named Entity Mapping (`NamedEntityMapper`)
- **Purpose**: Maps named entities across languages (Recommended)
- **Features**:
  - Extracts named entities from queries
  - Maps proper nouns across languages (e.g., "Bangladesh" ↔ "বাংলাদেশ")
  - Important for matching place names, person names, organization names

### 6. Query Processor (`QueryProcessor`)
- **Purpose**: Main pipeline that integrates all components
- **Features**:
  - Complete query processing workflow
  - Returns processed queries for each target language
  - Includes timing information
  - Returns expanded terms and named entity mappings

## Usage

### Basic Usage

```python
from query_processor import QueryProcessor

processor = QueryProcessor()

# Process a query
result = processor.process("bangladesh election")

print(f"Detected Language: {result['detected_language']}")
print(f"Target Queries:")
for lang, query in result['target_queries'].items():
    print(f"  {lang}: {query}")
```

### Advanced Usage

```python
# Process with options
result = processor.process(
    query="বাংলাদেশ নির্বাচন",
    target_languages=['bn', 'en'],
    remove_stopwords=False,
    expand=True,
    map_nes=True
)

# Get processed queries for retrieval
detected_lang, queries = processor.process_for_retrieval("bangladesh election")
```

## Output Format

The `process()` method returns a dictionary with:

```python
{
    'original_query': str,           # Original input query
    'detected_language': str,        # 'bn', 'en', or 'mixed'
    'normalized_query': str,         # Normalized version
    'target_queries': {              # Queries for each target language
        'bn': str,                   # Query for Bangla documents
        'en': str                    # Query for English documents
    },
    'expanded_terms': List[str],     # List of expanded terms (synonyms)
    'named_entities': Dict[str, str], # Named entity mappings
    'processing_time': float         # Processing time in milliseconds
}
```

## Testing

Run the comprehensive test suite:

```bash
python3 Module_B/test_query_processor.py
```

The test suite verifies:
- Language detection accuracy
- Normalization correctness
- Translation quality
- Query expansion
- Named entity mapping
- Full pipeline integration

## Test Results

All 6 test suites passed:
- ✓ Language Detection (7/7 tests passed)
- ✓ Normalization (3/3 tests passed)
- ✓ Translation (3/3 tests passed)
- ✓ Query Expansion (2/2 tests passed)
- ✓ Named Entity Mapping (3/3 tests passed)
- ✓ Full Pipeline (4/4 tests passed)

## Integration with Retrieval

The processed queries can be directly used with the retrieval models:

```python
from query_processor import QueryProcessor
from indexer import DocumentIndexer

processor = QueryProcessor()
indexer = DocumentIndexer()
indexer.load_index("Module_A/indexed_data")

# Process query
result = processor.process("bangladesh election")

# Use target queries for retrieval
for lang, query in result['target_queries'].items():
    # Tokenize query
    tokens = indexer.tokenize(query, lang)
    
    # Retrieve documents using tokens
    # (This will be implemented in Module C)
```

## Requirements

- Python 3.6+
- Standard library only (no external dependencies required)
- For production use, consider integrating:
  - `googletrans` or `deep-translator` for better translation
  - `spacy` or `nltk` for better NER
  - `sentence-transformers` for semantic query expansion

## Notes

1. **Translation**: Currently uses a dictionary-based approach. For better translation quality, integrate with:
   - Google Translate API (free tier available)
   - DeepL API (free tier available)
   - OPUS-MT models from HuggingFace (local, free)

2. **Named Entity Recognition**: Uses a dictionary-based approach. For production, consider:
   - spaCy with Bangla model
   - HuggingFace Transformers (multilingual NER models)

3. **Code-Switching**: The system detects and handles code-switched queries, but translation may be partial for mixed queries.

4. **Processing Time**: Typical processing time is < 1ms per query, making it suitable for real-time retrieval.

