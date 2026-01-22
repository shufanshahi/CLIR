"""
Script to inspect the generated document embeddings for Module C.
This helps verify that the embeddings were generated correctly and helps understand their structure.
"""

import torch
import json
import os
import sys
import numpy as np

# Ensure we can import from Module_A
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Module_A.indexer import DocumentIndexer

# Paths
EMBEDDING_FILE = "Module_C/data/doc_embeddings.pt"
MAPPING_FILE = "Module_C/data/doc_id_mapping.json"

def inspect_embeddings():
    print("=" * 60)
    print("INSPECTING DOCUMENT EMBEDDINGS")
    print("=" * 60)
    
    # 1. Load Embeddings
    if not os.path.exists(EMBEDDING_FILE):
        print(f"Error: Embedding file {EMBEDDING_FILE} not found!")
        return

    print(f"\n1. Loading Embeddings from {EMBEDDING_FILE}...")
    try:
        # Load torch tensor
        embeddings = torch.load(EMBEDDING_FILE)
        print(f"✓ Embeddings loaded successfully.")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Type: {embeddings.dtype}")
        
        num_docs, dim = embeddings.shape
        print(f"   Number of Documents: {num_docs}")
        print(f"   Vector Dimension: {dim}")
        
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # 2. Load Mapping
    print(f"\n2. Loading ID Mapping from {MAPPING_FILE}...")
    try:
        with open(MAPPING_FILE, 'r') as f:
            doc_ids = json.load(f)
        print(f"✓ Mapping loaded successfully.")
        print(f"   Total Mapped IDs: {len(doc_ids)}")
        
        if len(doc_ids) != num_docs:
            print(f"⚠ WARNING: Mismatch! Embeddings have {num_docs} rows but mapping has {len(doc_ids)} IDs.")
    except Exception as e:
        print(f"Error loading mapping: {e}")
        return

    # 3. Load Real Document Metadata for Context
    print(f"\n3. Loading Document Metadata for Context...")
    try:
        indexer = DocumentIndexer()
        indexer.load_index("Module_A/indexed_data")
        print(f"✓ Index loaded.")
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # 4. Statistical Analysis
    print("\n4. Statistical Analysis of Embeddings:")
    
    # Convert to numpy for easier stats
    emb_np = embeddings.numpy()
    
    print(f"   Min Value: {np.min(emb_np):.4f}")
    print(f"   Max Value: {np.max(emb_np):.4f}")
    print(f"   Mean Value: {np.mean(emb_np):.4f}")
    print(f"   Std Dev: {np.std(emb_np):.4f}")
    
    norm_vectors = np.linalg.norm(emb_np, axis=1)
    print(f"   Average Vector Norm: {np.mean(norm_vectors):.4f}")

    # 5. Sample Inspection
    print("\n5. Inspecting Random Samples:")
    
    # Pick 3 random indices
    import random
    random_indices = random.sample(range(num_docs), 3)
    
    for idx in random_indices:
        doc_id = doc_ids[idx]
        doc_id_int = int(doc_id) # Ensure int for dictionary lookup
        
        meta = indexer.document_metadata.get(doc_id_int, {})
        title = meta.get('title', 'Unknown Title')
        lang = meta.get('language', '??')
        
        print(f"\n   ------------------------------------------------------------")
        print(f"   Sample Index: {idx} -> DocID: {doc_id}")
        print(f"   Title: {title[:60]}...")
        print(f"   Language: {lang}")
        print(f"   First 10 values of embedding vector:")
        print(f"   {emb_np[idx][:10]}")
        
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    inspect_embeddings()
