# retrieval.py

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import os
from datetime import datetime

from config import FAISS_INDEX_PATH, CHUNKS_DATA_PATH, EMBEDDING_MODEL_NAME

# Global cache for loaded resources to avoid reloading on every request
_faiss_index: Optional[faiss.Index] = None
_chunk_data: List[Dict] = []
_embedding_model: Optional[SentenceTransformer] = None

def _load_retrieval_resources_if_needed():
    """Loads FAISS index, chunk data, and embedding model if not already loaded."""
    global _faiss_index, _chunk_data, _embedding_model
    
    if _embedding_model is None:
        print(f"Loading embedding model for retrieval: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
    
    if _faiss_index is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Please run the ingestion script first.")
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"FAISS index loaded. Total vectors: {_faiss_index.ntotal}")

    if not _chunk_data:
        if not os.path.exists(CHUNKS_DATA_PATH):
            raise FileNotFoundError(f"Chunk data not found at {CHUNKS_DATA_PATH}. Please run the ingestion script first.")
        print(f"Loading chunk data from {CHUNKS_DATA_PATH}...")
        with open(CHUNKS_DATA_PATH, 'r', encoding='utf-8') as f:
            _chunk_data = json.load(f)
        print(f"Chunk data loaded. Total chunks: {len(_chunk_data)}")

    if _faiss_index and _chunk_data and _faiss_index.ntotal != len(_chunk_data):
        raise ValueError("FATAL: Mismatch between FAISS index size and chunk data length. Please re-run ingestion.")
    
    return _faiss_index, _chunk_data, _embedding_model

def parse_date_robustly(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str or date_str.lower() == "unknown date":
        return None
    
    formats_to_try = ["%B %d, %Y", "%Y-%m-%d"]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    print(f"Warning: Could not parse date string: '{date_str}'")
    return None

def filter_chunks_by_metadata(
    all_chunks_data: List[Dict],
    category_filter: Optional[str] = None,
    date_before_filter_str: Optional[str] = None,
    date_after_filter_str: Optional[str] = None
) -> Tuple[List[Dict], List[int]]:
    # This function is now correctly called with None for placeholder values
    # The logic here is fine as-is
    filtered_chunks_info = []
    dt_date_before = parse_date_robustly(date_before_filter_str)
    dt_date_after = parse_date_robustly(date_after_filter_str)

    for original_idx, chunk_dict in enumerate(all_chunks_data):
        metadata = chunk_dict.get("metadata", {})
        passes_filter = True

        if category_filter:
            chunk_category = metadata.get("category", "")
            if not (chunk_category and category_filter.lower() in chunk_category.lower()):
                passes_filter = False
        
        if passes_filter and (dt_date_before or dt_date_after):
            chunk_pub_date = parse_date_robustly(metadata.get("publication_date"))
            if not chunk_pub_date:
                passes_filter = False
            else:
                if dt_date_before and chunk_pub_date >= dt_date_before:
                    passes_filter = False
                if dt_date_after and chunk_pub_date <= dt_date_after:
                    passes_filter = False
            
        if passes_filter:
            filtered_chunks_info.append((chunk_dict, original_idx))
    
    if not filtered_chunks_info:
        return [], []
        
    filtered_chunks_list, original_indices_for_faiss = zip(*filtered_chunks_info)
    return list(filtered_chunks_list), list(original_indices_for_faiss)


def search_vector_store(
    query_embedding: np.ndarray,
    top_k: int,
    faiss_index: faiss.Index,
    candidate_faiss_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if candidate_faiss_indices is not None and len(candidate_faiss_indices) > 0:
        if not faiss_index.ntotal > 0: return np.array([]), np.array([])
        valid_candidate_indices = [idx for idx in candidate_faiss_indices if 0 <= idx < faiss_index.ntotal]
        if not valid_candidate_indices: return np.array([]), np.array([])

        candidate_vectors = np.array([faiss_index.reconstruct(int(i)) for i in valid_candidate_indices]).astype('float32')
        if candidate_vectors.shape[0] == 0: return np.array([]), np.array([])

        sub_index = faiss.IndexFlatL2(candidate_vectors.shape[1])
        sub_index.add(candidate_vectors)
        
        actual_k_search = min(top_k, sub_index.ntotal)
        if actual_k_search == 0: return np.array([]), np.array([])

        distances, sub_indices = sub_index.search(query_embedding, actual_k_search)
        actual_faiss_indices = np.array([valid_candidate_indices[i] for i in sub_indices[0]]).reshape(1, -1)
        return distances, actual_faiss_indices
    else:
        actual_k_search = min(top_k, faiss_index.ntotal)
        if actual_k_search == 0: return np.array([]), np.array([])
        return faiss_index.search(query_embedding, actual_k_search)


def retrieve_relevant_chunks(
    query: str,
    top_k: int,
    category: Optional[str] = None,
    date_before: Optional[str] = None,
    date_after: Optional[str] = None
) -> List[Dict]:
    faiss_idx, all_chunk_dicts, emb_model = _load_retrieval_resources_if_needed()

    if emb_model is None or faiss_idx is None:
        print("Error: Retrieval resources not available.")
        return []

    candidate_chunks_for_vector_search = all_chunk_dicts
    candidate_faiss_indices: Optional[List[int]] = None

    if category or date_before or date_after:
        print(f"Applying metadata filters: Category='{category}', DateBefore='{date_before}', DateAfter='{date_after}'")
        candidate_chunks_for_vector_search, candidate_faiss_indices = filter_chunks_by_metadata(
            all_chunk_dicts, category, date_before, date_after
        )
        if not candidate_chunks_for_vector_search:
            print("No chunks matched metadata filters.")
            return []
        print(f"Found {len(candidate_chunks_for_vector_search)} chunks after metadata filtering.")
    
    query_embedding_np = emb_model.encode([query], convert_to_numpy=True).astype('float32')
    
    distances, retrieved_faiss_indices = search_vector_store(
        query_embedding_np, top_k, faiss_idx, candidate_faiss_indices
    )

    results = []
    if distances.size > 0:
        for i in range(retrieved_faiss_indices.shape[1]):
            original_doc_idx = retrieved_faiss_indices[0, i]
            if 0 <= original_doc_idx < len(all_chunk_dicts):
                chunk_info = all_chunk_dicts[original_doc_idx]
                
                # --- CORRECTED SCORING LOGIC ---
                l2_distance = float(distances[0, i])
                # For normalized vectors, cosine_similarity = 1 - (L2_distance^2 / 2)
                cosine_similarity = 1.0 - (l2_distance ** 2) / 2
                # Clamp score between 0 and 1 to handle potential floating point inaccuracies
                score = max(0.0, min(1.0, cosine_similarity))
                # --- END CORRECTION ---
                
                results.append({**chunk_info, "score": score})
            else:
                print(f"Warning: Stale or invalid index {original_doc_idx} from FAISS search.")
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]