# evaluate.py
import asyncio
import json
import argparse # For command-line arguments
from typing import List, Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import httpx

# Import your existing RAG components
from retrieval import retrieve_relevant_chunks, _load_retrieval_resources_if_needed
from generation import generate_llm_response
from re_ranking import re_rank_chunks
from config import TOGETHER_API_KEY, TOGETHER_API_BASE_URL, LLM_MODEL_NAME_FALLBACK
from utils import load_and_sample_dataset # Assuming utils.py exists

# --- Evaluation Metrics Implementation (Revised) ---

def evaluate_semantic_context_recall(retrieved_chunks: List[Dict], ground_truth_answer: str, model: SentenceTransformer, threshold: float = 0.80) -> float:
    """
    Checks if any retrieved chunk is semantically similar to the ground truth answer.
    This is more robust than a simple substring match.
    Metric: Binary score (1.0 for found, 0.0 for not found).
    """
    if not retrieved_chunks:
        return 0.0
    
    ground_truth_embedding = model.encode([ground_truth_answer])
    chunk_texts = [chunk['text'] for chunk in retrieved_chunks]
    chunk_embeddings = model.encode(chunk_texts)
    
    similarities = cosine_similarity(ground_truth_embedding, chunk_embeddings)
    
    # Check if any chunk's similarity score is above the threshold
    if np.max(similarities) >= threshold:
        return 1.0
    return 0.0

def evaluate_answer_similarity(generated_answer: str, ground_truth_answer: str, model: SentenceTransformer) -> float:
    """Measures semantic similarity between generated and ground truth answers."""
    if not generated_answer or not ground_truth_answer:
        return 0.0
    embeddings = model.encode([generated_answer, ground_truth_answer])
    similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    return float(similarity_matrix[0][0])

async def evaluate_faithfulness(generated_answer: str, retrieved_context_str: str) -> float:
    """Uses an LLM-as-a-judge to check if the answer is factually supported by the context."""
    if not generated_answer:
        return 0.0
    
    system_prompt = "You are a meticulous fact-checker... Respond with only 'YES' or 'NO'." # Omitted for brevity
    user_prompt = f"Context:\n---\n{retrieved_context_str}\n---\n\nAnswer:\n---\n{generated_answer}\n---\n\nIs the Answer fully supported by the Context? Respond with only 'YES' or 'NO'."
    
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    payload = {"model": LLM_MODEL_NAME_FALLBACK, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "max_tokens": 5, "temperature": 0.0}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(f"{TOGETHER_API_BASE_URL}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            judgment = response.json()['choices'][0]['message']['content'].strip().upper()
            return 1.0 if "YES" in judgment else 0.0
        except Exception as e:
            print(f"  [Faithfulness check failed: {e}]")
            return 0.0

# --- Main Evaluation Loop (Corrected) ---

async def run_evaluation(args):
    """Main function to run the RAG evaluation pipeline."""
    print("Starting RAG evaluation...")
    if not TOGETHER_API_KEY:
        print("ERROR: TOGETHER_API_KEY is not set.")
        return

    # Load resources
    _load_retrieval_resources_if_needed()
    
    # --- FIX: Corrected model name ---
    print("Loading embedding model for evaluation...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded.")
    
    try:
        dataset = load_and_sample_dataset(args.eval_file, args.num_questions)
    except Exception as e:
        print(f"Error: Could not load or sample the dataset. {e}")
        return

    print(f"\nProcessing {len(dataset)} questions through the RAG pipeline...")
    
    results = []
    for i, item in enumerate(dataset):
        print(f"\n--- Evaluating Question {i+1}/{len(dataset)} ---")
        question = item['question']
        ground_truth = item['ground_truth'] # FIX: Use the new data format
        print(f"Question: {question}")
        
        # --- Evaluate the full pipeline including re-ranking ---
        # 1. Initial Retrieval (get more documents for re-ranker)
        initial_chunks = retrieve_relevant_chunks(query=question, top_k=10)
        
        # 2. Evaluate Context Recall on the initial retrieval set
        context_recall_score = evaluate_semantic_context_recall(initial_chunks, ground_truth, embedding_model)
        
        if not initial_chunks:
            print("  [No chunks retrieved. Scoring 0 for all metrics.]")
            results.append({"question": question, "ground_truth": ground_truth, "context_recall": 0, "faithfulness": 0, "answer_similarity": 0, "answer": "N/A - No context found"})
            continue

        # 3. Re-ranking
        reranked_chunks = re_rank_chunks(question, initial_chunks)
        
        # 4. Select final context for generation
        final_context_for_llm = reranked_chunks[:5]
        
        # 5. Generation
        llm_result = await generate_llm_response(query=question, context_chunks=final_context_for_llm)
        generated_answer = llm_result.get("answer", "")
        
        # 6. Evaluate Generation
        answer_similarity_score = evaluate_answer_similarity(generated_answer, ground_truth, embedding_model)
        context_text = "\n\n".join([c['text'] for c in final_context_for_llm])
        faithfulness_score = await evaluate_faithfulness(generated_answer, context_text)

        print(f"  Generated Answer: {generated_answer}")
        print(f"  Scores -> Context Recall: {context_recall_score:.2f}, Answer Similarity: {answer_similarity_score:.2f}, Faithfulness: {faithfulness_score:.2f}")

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "context_recall": context_recall_score,
            "faithfulness": faithfulness_score,
            "answer_similarity": answer_similarity_score,
            "answer": generated_answer,
            "retrieved_context": [c['text'] for c in final_context_for_llm]
        })

    # --- Final Report ---
    df = pd.DataFrame(results)
    average_scores = df[['context_recall', 'faithfulness', 'answer_similarity']].mean()
    
    print("\n\n--- RAG Evaluation Summary ---")
    print(f"Total Questions Evaluated: {len(df)}")
    print(average_scores.to_string())
    print("----------------------------")
    
    report_path = "evaluation_report.csv"
    df.to_csv(report_path, index=False)
    print(f"Detailed report saved to {report_path}")

if __name__ == "__main__":
    # --- FIX: Correctly set up argument parsing ---
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    parser.add_argument(
        "--eval_file", 
        type=str, 
        default="data/evaluation_qna.json", 
        help="Path to evaluation Q&A JSON file."
    )
    parser.add_argument(
        "--num_questions", 
        type=int, 
        default=5,  # Default to a small number for quick tests
        help="Number of questions to evaluate. Use -1 for all questions."
    )
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(args))