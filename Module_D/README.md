# Module D - Ranking, Scoring, & Evaluation

This module implements the complete evaluation framework for the CLIR system, including ranking functions, confidence scoring, evaluation metrics, relevance labeling, error analysis, and external search engine comparison.

## Components

### 1. Ranking and Scoring (`ranker.py`)
- **CLIRRanker**: Main ranking class that implements confidence scoring
- Features:
  - Confidence scores normalized to [0,1] range
  - Low-confidence warnings (threshold: 0.20)
  - Query execution time tracking with breakdown
  - Support for multiple retrieval models (lexical, semantic, fuzzy, hybrid)
  - Batch processing capabilities

### 2. Evaluation Metrics (`metrics.py`)
- **IRMetrics**: Standard Information Retrieval evaluation metrics
- Implemented metrics:
  - Precision@K (target: ≥0.6 for K=10)
  - Recall@K (target: ≥0.5 for K=50)
  - nDCG@K (target: ≥0.5 for K=10)
  - Mean Reciprocal Rank (MRR) (target: ≥0.4)
  - Mean Average Precision (MAP)
  - Hit Rate@K

### 3. Relevance Labeling (`labeling.py`)
- **RelevanceLabeler**: Manual relevance labeling system
- Features:
  - CSV-based label storage with metadata
  - Support for multiple annotators
  - Annotation template generation
  - Label statistics and export for evaluation
  - Sample format: `query, doc_url, relevant, language, annotator, timestamp, notes`

### 4. Error Analysis (`error_analysis.py`)
- **ErrorAnalyzer**: Comprehensive error analysis framework
- Error categories:
  - **Translation Failures**: Query translation errors
  - **Named Entity Mismatch**: Cross-lingual NE matching issues
  - **Semantic vs Lexical**: Cases where one model outperforms the other
  - **Cross-Script Ambiguity**: Transliteration variations
  - **Code-Switching**: Mixed-language queries

### 5. Comprehensive Evaluator (`evaluator.py`)
- **CLIREvaluator**: Main evaluation orchestrator
- Features:
  - Single and batch query evaluation
  - Automatic metric calculation and target checking
  - Performance statistics and visualization generation
  - Comprehensive report generation
  - Integration with all other components

### 6. External Search Comparison (`external_comparison.py`)
- **ExternalSearchComparator**: Compare CLIR with external search engines
- Supported engines:
  - DuckDuckGo (free, no API key required)
  - Bing Web Search API (requires API key)
  - Google Custom Search API (requires API key and CSE ID)
  - AI-powered search simulation
- Features:
  - Overlap analysis between CLIR and external results
  - Batch comparison capabilities
  - Detailed reporting

## Usage Examples

### Basic Evaluation

```python
from Module_D.evaluator import CLIREvaluator

# Initialize evaluator
evaluator = CLIREvaluator()

# Evaluate single query
result = evaluator.evaluate_single_query("election", k=10)
print(f"Results: {result['comparison']}")

# Evaluate multiple queries
queries = ["election", "শিক্ষা", "Bangladesh"]
batch_result = evaluator.evaluate_query_set(queries, k=10)

# Generate and save report
report = evaluator.generate_report(batch_result)
evaluator.save_evaluation(batch_result)
```

### Relevance Labeling

```python
from Module_D.labeling import RelevanceLabeler

# Initialize labeler
labeler = RelevanceLabeler()

# Add manual labels
labeler.add_label(
    query="election",
    doc_url="https://example.com/election-news",
    relevant=True,
    language="en",
    annotator="annotator1",
    notes="Clearly about election results"
)

# Get statistics
stats = labeler.get_statistics()
print(f"Total labels: {stats['total_labels']}")
```

### External Search Comparison

```python
from Module_D.external_comparison import ExternalSearchComparator

# Initialize comparator
comparator = ExternalSearchComparator()

# Compare single query
clir_results = [{'url': 'https://example.com/doc1', 'title': 'Election News'}]
comparison = comparator.compare_with_external("election", clir_results)

# Batch comparison
queries = ["election", "education"]
clir_results_dict = {"election": clir_results, "education": []}
batch_comparison = comparator.batch_compare(queries, clir_results_dict)
```

## File Structure

```
Module_D/
├── __init__.py              # Module initialization
├── ranker.py                # Ranking and confidence scoring
├── metrics.py               # IR evaluation metrics
├── labeling.py              # Relevance labeling system
├── error_analysis.py        # Error analysis framework
├── evaluator.py             # Main evaluation orchestrator
├── external_comparison.py   # External search engine comparison
├── README.md               # This file
└── data/                   # Data directory
    ├── relevance_labels.csv
    ├── error_analysis/
    └── external_comparison/
```

## Evaluation Targets

The system aims to meet these assignment targets:

| Metric | Target | Description |
|--------|--------|-------------|
| Precision@10 | ≥ 0.6 | At least 6 relevant documents in top 10 |
| Recall@50 | ≥ 0.5 | At least 50% of relevant documents retrieved |
| nDCG@10 | ≥ 0.5 | Good ranking quality with position penalties |
| MRR | ≥ 0.4 | First relevant document appears early |

## Error Analysis Categories

### 1. Translation Failures
- Example: Query "চয়ার" (chair) mistranslated to "Chairman"
- Detection: Compare results before/after translation
- Impact: Wrong document retrieval

### 2. Named Entity Mismatch
- Example: "ঢাকা" vs "Dhaka" not matching
- Detection: Entity extraction and cross-lingual matching
- Impact: Missing relevant documents

### 3. Semantic vs Lexical
- Example: "শিক্ষা" (education) - BM25 fails, embeddings succeed
- Detection: Compare model performance differences
- Impact: Understanding model strengths/weaknesses

### 4. Cross-Script Ambiguity
- Example: "Bangladesh" vs "বাংলাদেশ" vs "Bangla Desh"
- Detection: Transliteration variation analysis
- Impact: Inconsistent matching

### 5. Code-Switching
- Example: Mixed Bangla-English queries
- Detection: Script mixing detection
- Impact: Query processing challenges

## Configuration

### External Search APIs (Optional)

For external search comparison, you can configure API keys:

```python
external_configs = {
    'bing_api_key': 'your-bing-api-key',
    'google_api_key': 'your-google-api-key',
    'google_search_engine_id': 'your-cse-id'
}

comparison = comparator.compare_with_external("election", clir_results, external_configs)
```

### Confidence Threshold

Adjust the low-confidence warning threshold:

```python
ranker = CLIRRanker(retriever, confidence_threshold=0.15)  # Default: 0.20
```

## Output Formats

### Evaluation Results (JSON)
```json
{
  "timestamp": "2024-01-22T...",
  "queries": ["election", "education"],
  "aggregate_metrics": {
    "hybrid": {
      "avg_precision@10": 0.75,
      "avg_recall@50": 0.60,
      "avg_ndcg@10": 0.68,
      "avg_mrr": 0.72
    }
  },
  "target_status": {
    "hybrid": {
      "overall_passed": true
    }
  }
}
```

### Relevance Labels (CSV)
```csv
query,doc_url,relevant,language,annotator,timestamp,notes
election,https://example.com/election1,yes,en,annotator1,2024-01-22T...,Relevant content
education,https://example.com/edu2,no,bn,annotator1,2024-01-22...,Not relevant
```

## Dependencies

Install required packages:

```bash
conda activate clir
pip install pandas matplotlib seaborn requests beautifulsoup4
```

## Testing

Run individual component tests:

```bash
# Test metrics
python Module_D/metrics.py

# Test ranker
python Module_D/ranker.py

# Test evaluator (requires Modules A, B, C)
python Module_D/evaluator.py

# Test external comparison
python Module_D/external_comparison.py
```

## Integration with Other Modules

- **Module A**: Uses indexed documents and metadata
- **Module B**: Uses query processing for cross-lingual handling
- **Module C**: Uses retrieval models (BM25, semantic, fuzzy, hybrid)

The evaluator automatically initializes and integrates with all required modules.
