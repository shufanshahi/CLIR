"""
Document Embedding Generator for CLIR Module C.
Loads documents from Module A's index and generates dense vector embeddings
using a multilingual Transformer model (LaBSE).
"""

import sys
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

# Ensure we can import from Module_A
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Module_A.indexer import DocumentIndexer

# Configuration
# MODEL_NAME = "sentence-transformers/LaBSE" # Too large (1.88GB), failing to download
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Smaller (~470MB) and effective
BATCH_SIZE = 32
OUTPUT_DIR = "Module_C/data"
EMBEDDING_FILE = "doc_embeddings.pt"
MAPPING_FILE = "doc_id_mapping.json"

def load_documents_from_index() -> List[Dict[str, Any]]:
    """Load all documents from Module A's index."""
    print("Loading Module A index...")
    indexer = DocumentIndexer()
    indexer.load_index("Module_A/indexed_data")
    
    documents = []
    # We want to process documents in order of their IDs to match embedding indices
    sorted_doc_ids = sorted(indexer.document_metadata.keys())
    
    print(f"preparing {len(sorted_doc_ids)} documents for embedding...")
    
    for doc_id in sorted_doc_ids:
        meta = indexer.document_metadata[doc_id]
        # Combine title and body for better context
        # Title is repeated to give it more weight
        text = f"{meta.get('title', '')} {meta.get('title', '')} {meta.get('body', '')}"
        
        documents.append({
            'doc_id': doc_id,
            'text': text,
            'language': meta.get('language', 'unknown')
        })
        
    return documents

def generate_embeddings():
    """Generate and save document embeddings."""
    print("=" * 60)
    print("CLIR Document Embedding Generator")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Documents
    documents = load_documents_from_index()
    if not documents:
        print("Error: No documents found!")
        return

    # 2. Initialize Model
    print(f"\nInitializing model: {MODEL_NAME}...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model loaded on {device}")
        
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Please install it: pip install sentence-transformers")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Generate Embeddings
    texts = [doc['text'] for doc in documents]
    doc_ids = [doc['doc_id'] for doc in documents]
    
    print(f"\nGenerating embeddings for {len(texts)} documents...")
    print(f"Batch size: {BATCH_SIZE}")
    
    start_time = time.time()
    
    # Encode all texts
    # show_progress_bar=True gives a nice progress bar
    embeddings = model.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        device=device
    )
    
    elapsed = time.time() - start_time
    print(f"\nEmbedding generation complete!")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Shape: {embeddings.shape}")

    # 4. Save Embeddings & Mapping
    print(f"\nSaving data to {OUTPUT_DIR}/...")
    
    # Save embeddings as Torch tensor
    emb_path = os.path.join(OUTPUT_DIR, EMBEDDING_FILE)
    torch.save(embeddings.cpu(), emb_path)
    
    # Save doc_id mapping (index -> doc_id)
    # This ensures we know which row corresponds to which document ID
    mapping_path = os.path.join(OUTPUT_DIR, MAPPING_FILE)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(doc_ids, f)
        
    print(f"✓ Embeddings saved to: {emb_path}")
    print(f"✓ ID Mapping saved to: {mapping_path}")
    print("\nDone.")

if __name__ == "__main__":
    generate_embeddings()
