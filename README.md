# CLIR System

## Description

This project implements a Cross-Lingual Information Retrieval (CLIR) system capable of retrieving and ranking multilingual documents in Bangla and English. It is built as part of a Data Mining course assignment, focusing on search pipelines, indexing, query processing, retrieval models, and evaluation.

## Features

- Multilingual dataset construction from news websites
- Inverted index with document metadata
- Query processing with language detection, normalization, translation, expansion, and named entity mapping
- Multiple retrieval models: BM25, TF-IDF, semantic embeddings
- Evaluation metrics and error analysis
- Support for Bangla and English languages

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd CLIR
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Collection and Indexing (Module A)

Run the indexing process to build the dataset and inverted index:

```
python Module_A/indexer.py
```

### Query Processing (Module B)

Process queries using the query processor:

```
python Module_B/query_processor.py
```

### Retrieval (Module C)

Perform retrieval using different models:

```
python Module_C/retriever.py
```

### Evaluation (Module D)

Run evaluations and generate reports:

```
python Module_D/run_full_evaluation.py
```

## Project Structure

- `Module_A/`: Dataset construction and indexing
- `Module_B/`: Query processing and cross-lingual handling
- `Module_C/`: Retrieval models
- `Module_D/`: Evaluation and metrics
- `Bangla_DB/` and `English_DB/`: Scraped data from news sites
- `requirements.txt`: Python dependencies

## Modules Overview

### Module A: Dataset Construction & Indexing

Constructs a multilingual dataset by crawling 5 Bangla and 5 English news sites. Builds an inverted index with metadata including title, body, URL, date, language, tokens, and named entities.

### Module B: Query Processing & Cross-Lingual Handling

Implements a query pipeline with language detection, normalization, translation, expansion, and named entity mapping to enable cross-lingual retrieval.

### Module C: Retrieval Models

Implements lexical (BM25, TF-IDF) and semantic retrieval models using embeddings for ranking documents.

### Module D: Evaluation

Provides evaluation metrics, ranking, error analysis, and comparison with external systems.

## Evaluation

The system is evaluated using metrics such as precision, recall, F1-score, and NDCG. Results are stored in `Module_D/results/`.

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome. Please follow standard practices for pull requests and issues.