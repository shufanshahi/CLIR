# Module A - Document Indexing

## What is Indexing?

**Indexing** is the process of organizing documents in a way that enables fast and efficient retrieval. Think of it like an index in a textbook - instead of reading every page to find information about "Bangladesh", you look up "Bangladesh" in the index which tells you exactly which pages contain that word.

In our system, we create an **Inverted Index**:
- Maps each word (term) to the list of documents containing that word
- Stores term frequencies for ranking
- Enables fast lookup during search

### Example of Inverted Index:

```
Term: "bangladesh"
├── Doc 1: appears 5 times
├── Doc 3: appears 2 times
└── Doc 7: appears 3 times

Term: "university"
├── Doc 1: appears 2 times
├── Doc 2: appears 4 times
└── Doc 5: appears 1 time
```

## Is Indexing Done in Current Codebase?

**NO**, indexing is NOT implemented in the current codebase. The current code only:
- Scrapes articles from websites
- Saves articles to JSON files

We have now added the indexing system in `indexer.py`.

## Components

### 1. `indexer.py`
Creates and manages the inverted index:
- Reads all article JSON files
- Tokenizes document text
- Builds inverted index with term frequencies
- Calculates document statistics
- Supports TF-IDF and BM25 scoring
- Saves/loads index to/from disk

## How to Build the Index

### Step 1: Ensure Articles are Scraped
Make sure you have articles in:
- `Module_A/Bangla_DB/BanglaTribune/articles.json`
- `Module_A/Bangla_DB/DhakaPost/articles.json`
- `Module_A/Bangla_DB/ProthomAlo/articles.json`
- `Module_A/English_DB/newagebd/articles.json`

### Step 2: Run the Indexer
```bash
cd Module_A
python indexer.py
```

This will:
1. Read all articles from JSON files
2. Build inverted index
3. Calculate statistics
4. Save index to `document_index.pkl`

Output:
```
Index built successfully!
Total documents: 2500
Vocabulary size: 50000
Average document length: 450.23

=== Index Statistics ===
Bangla documents: 1800
English documents: 700
```

## Index Structure

### Data Stored:

1. **Inverted Index**: `{term: {doc_id: frequency}}`
   ```python
   {
       'bangladesh': {
           0: 5,      # Document 0 contains 'bangladesh' 5 times
           15: 2,     # Document 15 contains 'bangladesh' 2 times
           ...
       },
       'university': {
           0: 2,
           3: 4,
           ...
       }
   }
   ```

2. **Document Metadata**: `{doc_id: {title, body, url, date, language}}`
   ```python
   {
       0: {
           'id': 0,
           'title': 'Article Title',
           'body': 'Article content...',
           'url': 'https://...',
           'date': '2025-04-01',
           'language': 'bn'
       },
       ...
   }
   ```

3. **Document Lengths**: `{doc_id: length}`
   - Used for length normalization in BM25

4. **Statistics**:
   - Total documents
   - Average document length
   - Vocabulary size

## Usage

### Load Existing Index
```python
from Module_A.indexer import DocumentIndexer

indexer = DocumentIndexer()
indexer.load_index('document_index.pkl')

print(f"Total documents: {indexer.total_docs}")
print(f"Vocabulary size: {len(indexer.vocabulary)}")
```

### Calculate TF-IDF Score
```python
# For term 'bangladesh' in document 0
score = indexer.calculate_tf_idf('bangladesh', 0)
print(f"TF-IDF Score: {score:.4f}")
```

### Calculate BM25 Score
```python
# For term 'bangladesh' in document 0
score = indexer.calculate_bm25('bangladesh', 0, k1=1.5, b=0.75)
print(f"BM25 Score: {score:.4f}")
```

### Search for Documents Containing a Term
```python
term = 'bangladesh'
if term in indexer.inverted_index:
    docs = indexer.inverted_index[term]
    print(f"'{term}' appears in {len(docs)} documents")
    
    for doc_id, freq in list(docs.items())[:5]:  # Show first 5
        print(f"  Doc {doc_id}: {freq} times")
        print(f"    Title: {indexer.doc_metadata[doc_id]['title']}")
```

## Ranking Methods

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**Formula**: `TF-IDF = TF × IDF`

Where:
- **TF (Term Frequency)**: `frequency of term in document / total terms in document`
- **IDF (Inverse Document Frequency)**: `log(total documents / documents containing term)`

**Intuition**:
- Terms appearing many times in a document get higher weight (TF)
- Terms appearing in many documents get lower weight (IDF)
- Balances term importance at document and collection level

**Example**:
```python
# Word "bangladesh" appears 5 times in a 100-word document
# Word "bangladesh" appears in 50 out of 1000 documents

TF = 5/100 = 0.05
IDF = log(1000/50) = log(20) ≈ 3.0
TF-IDF = 0.05 × 3.0 = 0.15
```

### 2. BM25 (Best Matching 25)

**Formula**: More complex than TF-IDF, considers:
- Term frequency with saturation
- Document length normalization
- Inverse document frequency

**Parameters**:
- `k1` (default=1.5): Controls term frequency saturation
  - Higher k1 = more weight to repeated terms
  - Lower k1 = diminishing returns for repetition
- `b` (default=0.75): Controls length normalization
  - b=1: Full length normalization
  - b=0: No length normalization

**Advantages over TF-IDF**:
- Handles term frequency saturation better
- Better length normalization
- Generally produces better rankings

## Index Statistics

After building index, check statistics:

```python
indexer = DocumentIndexer()
indexer.load_index('document_index.pkl')

print(f"Total Documents: {indexer.total_docs}")
print(f"Vocabulary Size: {len(indexer.vocabulary)}")
print(f"Average Doc Length: {indexer.avg_doc_length:.2f}")

# Language distribution
bn_docs = sum(1 for doc in indexer.documents if doc['language'] == 'bn')
en_docs = sum(1 for doc in indexer.documents if doc['language'] == 'en')
print(f"Bangla Documents: {bn_docs}")
print(f"English Documents: {en_docs}")

# Most common terms
from collections import Counter
term_doc_counts = {term: len(docs) for term, docs in indexer.inverted_index.items()}
most_common = Counter(term_doc_counts).most_common(10)
print("\nMost Common Terms:")
for term, count in most_common:
    print(f"  {term}: appears in {count} documents")
```

## Minimum Metadata Requirements

Each document must have:
- ✅ **title**: Document title
- ✅ **body**: Document content/text
- ✅ **url**: Source URL
- ✅ **date**: Publication date
- ✅ **language**: 'bn' for Bangla, 'en' for English

Optional (for future enhancements):
- **tokens**: Word count
- **word_embeddings**: Vector representations
- **named_entities**: Extracted entities

## Rebuilding the Index

When to rebuild:
1. New articles are scraped
2. Articles are updated/modified
3. Want to change tokenization
4. Want to experiment with different preprocessing

To rebuild:
```bash
cd Module_A
python indexer.py
```

Old index is overwritten with new one.

## Performance Considerations

### Index Size
- Depends on vocabulary size and number of documents
- ~50MB for 2500 documents with 50K vocabulary
- Use pickle format for fast loading

### Loading Time
- ~1-2 seconds for typical index
- Index is loaded once at system startup

### Query Time
- Very fast (milliseconds)
- Lookup in inverted index is O(1)
- Ranking all matching documents is O(n) where n = matching docs

## Troubleshooting

### Issue: Index file not found
```
Solution: Run `python indexer.py` first
```

### Issue: Articles not found
```
Solution: Check JSON files exist in correct folders
```

### Issue: Memory error
```
Solution: Process articles in batches
Modify indexer.py to read files one at a time
```

### Issue: Index too large
```
Solution:
1. Remove very common terms (stopwords)
2. Use stemming/lemmatization
3. Limit vocabulary size
```

## Advanced: Custom Tokenization

Modify `tokenize()` method for better tokenization:

```python
def tokenize(self, text):
    # Add Bangla-specific rules
    # Add stemming
    # Add stopword removal
    # etc.
    pass
```

## Next Steps

After building index:
1. ✅ Index is ready
2. Go to Module_B for query processing
3. Use `retrieval_system.py` for complete CLIR

## Questions?

Common questions:

**Q: Do I need to rebuild index every time?**
A: No! Index is saved to disk. Load it using `load_index()`.

**Q: Can I add more documents later?**
A: Yes, but need to rebuild entire index. For incremental updates, modify `build_index_from_json_files()`.

**Q: Which is better: TF-IDF or BM25?**
A: BM25 is generally better. It's the default in most search engines.

**Q: How to see what's in the index?**
A: Use the statistics code above or explore `inverted_index` dictionary.
