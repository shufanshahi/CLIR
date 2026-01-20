# Module B - Query Processing & Cross-Lingual Handling

## Overview
This module implements a complete query processing pipeline for Cross-Lingual Information Retrieval (CLIR) between Bangla and English.

## Components

### 1. `query_processor.py`
Implements the query processing pipeline:
- **Language Detection**: Automatically detects if query is in Bangla or English
- **Normalization**: Lowercase, whitespace removal, optional stopword removal
- **Query Translation**: Translates queries between Bangla and English
- **Query Expansion**: Adds synonyms to improve recall
- **Named Entity Mapping**: Maps named entities across languages

### 2. `retrieval_system.py`
Complete CLIR system that:
- Integrates query processing with document indexing
- Implements TF-IDF and BM25 retrieval methods
- Performs cross-lingual search across both languages
- Ranks and returns relevant documents

## Installation

### Required Packages
```bash
pip install googletrans==4.0.0rc1  # Optional, for translation API
```

### Basic Setup (No external API)
The system works out-of-the-box with:
- Named entity mapping
- Simple translation via entity mapping
- Synonym-based query expansion

### Advanced Setup (With Translation API)
For better translation quality:
```bash
pip install googletrans==4.0.0rc1
```
Then set `use_translation_api=True` in search function.

## Usage

### 1. First Build the Index (Module A)
```bash
cd Module_A
python indexer.py
```

This creates `document_index.pkl` containing the inverted index.

### 2. Test Query Processing
```python
from Module_B.query_processor import QueryProcessor

processor = QueryProcessor()

# Process a query
result = processor.process_query(
    query="Bangladesh university",
    remove_stopwords=True,
    expand=True,
    translate=True,
    use_api=False
)

print(f"Detected Language: {result['detected_language']}")
print(f"Normalized: {result['normalized_query']}")
print(f"English Query: {result['queries']['en']['query']}")
print(f"Bangla Query: {result['queries']['bn']['query']}")
```

### 3. Perform Cross-Lingual Search
```python
from Module_B.retrieval_system import CrossLingualRetrievalSystem

# Initialize system
system = CrossLingualRetrievalSystem(index_path='Module_A/document_index.pkl')

# Search
results = system.search(
    query="Bangladesh university student",
    method='bm25',           # or 'tfidf'
    cross_lingual=True,      # Search both languages
    expand_query=True,       # Use query expansion
    remove_stopwords=True,
    use_translation_api=False,
    top_k=10                 # Number of results
)

# Display results
system.print_results(results, show_body=False)
```

### 4. Run Demo
```bash
cd Module_B
python retrieval_system.py
```

## Module B Pipeline Steps

### Step 1: Language Detection
```python
query = "ঢাকা বিশ্ববিদ্যালয়"
detected_lang = processor.detect_language(query)
# Output: 'bn'
```

### Step 2: Normalization
```python
normalized = processor.normalize_query(
    query="  Bangladesh   University  ",
    language='en',
    remove_stopwords=True
)
# Output: "bangladesh university"
```

### Step 3: Query Translation
```python
# Simple translation (NE mapping)
translated = processor.translate_query_simple(
    query="Bangladesh university",
    source_lang='en',
    target_lang='bn'
)

# API translation (if available)
translated = processor.translate_query_api(
    query="Bangladesh university",
    source_lang='en',
    target_lang='bn'
)
```

### Step 4: Query Expansion (Recommended)
```python
expanded = processor.expand_query(
    query="university student",
    language='en',
    max_synonyms=2
)
# Output: ['university', 'college', 'institution', 'student', 'pupil', 'learner']
```

### Step 5: Named Entity Mapping (Recommended)
```python
entities = processor.extract_named_entities("Bangladesh Dhaka university")
# Output: ['Bangladesh', 'Dhaka']

mapped = processor.map_named_entities(
    query="Bangladesh Dhaka",
    source_lang='en',
    target_lang='bn'
)
# Output: "বাংলাদেশ ঢাকা"
```

## Retrieval Methods

### TF-IDF Retrieval
```python
results = system.search(query="test", method='tfidf', top_k=10)
```

### BM25 Retrieval (Recommended)
```python
results = system.search(query="test", method='bm25', top_k=10)
```

BM25 parameters:
- `k1` (default=1.5): Term frequency saturation
- `b` (default=0.75): Document length normalization

## Cross-Lingual Search Modes

### 1. Monolingual Search
Search only in the query language:
```python
results = system.search(query="Bangladesh", cross_lingual=False)
```

### 2. Cross-Lingual Search
Search in both Bangla and English:
```python
results = system.search(query="Bangladesh", cross_lingual=True)
# Returns results from both languages
```

## Customization

### Add More Named Entity Mappings
Edit `query_processor.py`:
```python
self.ne_mappings = {
    'new_entity_en': 'নতুন_সত্তা',
    'নতুন_সত্তা': 'new_entity_en',
    # ... add more
}
```

### Add More Synonyms
Edit `query_processor.py`:
```python
self.synonyms_en = {
    'your_word': ['synonym1', 'synonym2'],
    # ... add more
}

self.synonyms_bn = {
    'আপনার_শব্দ': ['সমার্থক১', 'সমার্থক২'],
    # ... add more
}
```

### Adjust BM25 Parameters
```python
results = system.retrieve_bm25(
    query_terms=['test'],
    target_language='en',
    top_k=10,
    k1=2.0,    # Adjust term frequency impact
    b=0.5      # Adjust length normalization
)
```

## Example Queries

### English Queries
- "Bangladesh university student"
- "Dhaka government policy"
- "tourist places Bangladesh"
- "education system"

### Bangla Queries
- "বাংলাদেশ বিশ্ববিদ্যালয় ছাত্র"
- "ঢাকা সরকারি নীতি"
- "পর্যটন স্থান বাংলাদেশ"
- "শিক্ষা ব্যবস্থা"

## Output Format

Results dictionary structure:
```python
{
    'query_info': {
        'original_query': str,
        'detected_language': str,
        'normalized_query': str,
        'named_entities': List[str],
        'queries': {
            'bn': {...},
            'en': {...}
        }
    },
    'results': {
        'bn': {
            'query': str,
            'expanded_terms': List[str],
            'is_translated': bool,
            'documents': List[Tuple[int, float, Dict]],
            'count': int
        },
        'en': {...}
    }
}
```

## Performance Tips

1. **Enable Query Expansion**: Improves recall
2. **Use BM25 over TF-IDF**: Generally better ranking
3. **Remove Stopwords**: Reduces noise
4. **Use Translation API**: Better translation quality (requires internet)
5. **Adjust top_k**: Balance between coverage and speed

## Troubleshooting

### Issue: No results found
- Check if index is built (`Module_A/document_index.pkl` exists)
- Verify query language detection is correct
- Try without stopword removal
- Enable query expansion

### Issue: Translation not working
- Check if translation API is installed
- Fall back to simple NE mapping: `use_translation_api=False`
- Add more named entity mappings manually

### Issue: Poor ranking quality
- Try different retrieval method (BM25 vs TF-IDF)
- Adjust BM25 parameters (k1, b)
- Enable query expansion
- Check if stopword removal is helping or hurting

## Future Enhancements

1. **Better Translation**: Integrate Google Translate API or similar
2. **Advanced NER**: Use spaCy or similar for better NE extraction
3. **Multilingual Embeddings**: Use BERT or similar for semantic search
4. **Query Spelling Correction**: Handle typos
5. **Relevance Feedback**: Learn from user clicks
6. **Context-Aware Translation**: Consider query context
