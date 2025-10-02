from sentence_transformers.cross_encoder import CrossEncoder

def re_rank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Re-ranks a list of chunks using a CrossEncoder model."""
    if not chunks:
        return []
    
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Create pairs of [query, chunk_text]
    pairs = [[query, chunk['text']] for chunk in chunks]
    
    # Predict scores
    scores = model.predict(pairs)
    
    # Add new scores to chunks and sort
    for i in range(len(chunks)):
        chunks[i]['re_rank_score'] = scores[i]
        
    sorted_chunks = sorted(chunks, key=lambda x: x['re_rank_score'], reverse=True)
    return sorted_chunks