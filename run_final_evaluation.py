# run_final_evaluation.py

import asyncio
import json
from typing import List, Dict

# Import the necessary functions from your existing RAG project
from retrieval import retrieve_relevant_chunks, _load_retrieval_resources_if_needed
from re_ranking import re_rank_chunks
from generation import generate_llm_response

# --- 1. The list of 40 questions provided by your teacher ---
EVALUATION_QUESTIONS = [
    "Between which latitudes does metropolitan France lie?",
    "Name the three main geological regions of France.",
    "Why does periglacial action play an important role in shaping France’s landscape?",
    "Which mountain regions in France were directly sculpted by Pleistocene glaciation?",
    "What is loess, and what role did it play in French lowland soil formation?",
    "During which geologic time period did Hercynian folding occur in France?",
    "Identify the four principal massifs formed by Hercynian orogeny.",
    "What are “ballons,” and where can they be found?",
    "Describe the volcanic significance of the Chaîne des Puys.",
    "How did Alpine tectonic movements affect the Massif Central?",
    "What forms the geological core of the Paris Basin?",
    "Name three plains or basins within France’s great lowlands.",
    "What soil type covers the Paris Basin and contributes to its agricultural fertility?",
    "Describe the landscape and vegetation characteristics of the Alsace Plain.",
    "What coastal landform is typical of the Aquitaine Basin?",
    "Which mountain ranges in France are classified as “younger”?",
    "How long do the Pyrenees stretch, and what role do they play geographically?",
    "What is France’s highest mountain peak?",
    "Define the Camargue region and its environmental significance.",
    "What coastal characteristics define the French Riviera?",
    "What major geographical feature determines France’s river systems?",
    "List the four main French river systems and their sources.",
    "Which French river is the longest, and why is its flow considered irregular?",
    "How does the Rhône River’s flow regime differ from that of the Seine?",
    "Name at least two types of natural lakes found in France and how they originated.",
    "Which soil type dominates in most of France and under what original vegetation did it form?",
    "In what ways do Mediterranean soils differ from brown earths?",
    "Explain the agricultural importance of windblown limon in the Paris Basin.",
    "How does underlying bedrock influence soil quality in different regions?",
    "What human actions have historically improved or degraded French soils?",
    "What are the three major climate zones of France?",
    "Describe the defining features of the pure oceanic climate region.",
    "Which city has the greatest temperature range in France, and what climate does it represent?",
    "Characterize Mediterranean climate patterns, including rainfall distribution, and notable winds.",
    "How does elevation affect climate and vegetation in the French mountains?",
    "What are the two major biogeographic vegetation provinces in France?",
    "Name three tree species typical of the Holarctic-division forests in western France.",
    "What vegetation types appear in the Mediterranean zone during summer droughts?",
    "Describe the altitudinal progression of trees in the French high mountains.",
    "List three xerophytic plants common to southern France’s Mediterranean flora."
]

def format_top_k_for_output(reranked_chunks: List[Dict]) -> Dict[str, str]:
    """
    Formats the top 3 chunks into the required dictionary structure.
    Handles cases where fewer than 3 chunks are retrieved.
    """
    top_k_output = {}
    for i in range(3):
        if i < len(reranked_chunks):
            top_k_output[str(i + 1)] = reranked_chunks[i].get('text', 'N/A')
        else:
            # If there are fewer than 3 chunks, fill the rest with a placeholder
            top_k_output[str(i + 1)] = "Not enough chunks retrieved."
    return top_k_output


async def main():
    """
    Main async function to run the full evaluation process.
    """
    print("Starting final evaluation process...")
    
    # Load the RAG models and data once
    _load_retrieval_resources_if_needed()
    
    final_results = []

    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n--- Processing Question {i+1}/{len(EVALUATION_QUESTIONS)}: '{question}' ---")
        
        # 1. Retrieve initial set of chunks (get more for the re-ranker)
        initial_chunks = retrieve_relevant_chunks(query=question, top_k=10)
        
        if not initial_chunks:
            print("  WARNING: No chunks retrieved for this question. Skipping.")
            # Create a placeholder entry
            formatted_entry = {
                "question": question,
                "answer": "Could not generate an answer because no relevant context was found.",
                "top_k": {
                    "1": "N/A",
                    "2": "N/A",
                    "3": "N/A"
                }
            }
            final_results.append(formatted_entry)
            continue
            
        # 2. Re-rank the retrieved chunks to find the most relevant ones
        reranked_chunks = re_rank_chunks(question, initial_chunks)
        
        # 3. Select the top 5 chunks to use as context for the LLM
        context_for_llm = reranked_chunks[:5]
        
        # 4. Generate the answer using the LLM
        llm_response = await generate_llm_response(query=question, context_chunks=context_for_llm)
        generated_answer = llm_response.get("answer", "Failed to generate an answer.")
        
        print(f"  Generated Answer: {generated_answer[:100]}...")
        
        # 5. Format the entry for the final JSON file
        # The 'top_k' required by the validator uses the top 3 re-ranked chunks.
        top_k_formatted = format_top_k_for_output(reranked_chunks)
        
        formatted_entry = {
            "question": question,
            "answer": generated_answer,
            "top_k": top_k_formatted
        }
        
        final_results.append(formatted_entry)

    # 6. Save the final list of results to a JSON file
    output_filename = "my_final_output.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n\n✅ Evaluation complete. All {len(final_results)} results have been saved to '{output_filename}'")
    print(f"You can now verify this file using your teacher's script.")


if __name__ == "__main__":
    # This script can be run standalone without the FastAPI server.
    asyncio.run(main())