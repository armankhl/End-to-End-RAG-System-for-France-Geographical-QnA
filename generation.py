# generation.py
import requests
import json
from typing import List, Dict, Optional
import httpx

from config import (
    LLM_MODEL_NAME_PRIMARY,
    LLM_MODEL_NAME_FALLBACK,
    TOGETHER_API_KEY,
    TOGETHER_API_BASE_URL,
    DEFAULT_MAX_TOKENS_GENERATE,
    DEFAULT_TEMPERATURE_GENERATE,
    DEFAULT_TOP_P_GENERATE
)

def _construct_llm_messages(query: str, context_chunks: List[Dict]) -> List[Dict]:
    """
    Constructs the message list for the LLM API, following a chat format.
    Includes system prompt, user prompt with injected context.
    """
    context_str = "\n\n---\n\n".join([chunk['text'] for chunk in context_chunks])
    
    # System Prompt: Guides the LLM's behavior, tone, and constraints.
    system_prompt = (
        "You are an expert AI assistant specializing in the geography and land features of France. "
        "Your role is to answer questions based *exclusively* on the provided context. "
        "If the information is not in the context, clearly state that the context does not provide an answer. "
        "Do not invent information or use external knowledge. "
        "Provide concise, factual answers. If quoting, be brief and relevant."
    )
    
    # User Prompt: Contains the actual query and the retrieved context.
    user_prompt = (
        f"Here is some context about the geography of France:\n\n"
        f"START OF CONTEXT\n"
        f"{context_str}\n"
        f"END OF CONTEXT\n\n"
        f"Based *only* on the provided context, please answer the following question:\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

async def generate_llm_response(
    query: str,
    context_chunks: List[Dict],
    max_tokens: int = DEFAULT_MAX_TOKENS_GENERATE,
    temperature: float = DEFAULT_TEMPERATURE_GENERATE,
    top_p: float = DEFAULT_TOP_P_GENERATE
) -> Dict:
    """
    Generates a response from the LLM using the Together AI API.

    LLM Selection:
    - Primary: 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free' (as requested)
    - Fallback: 'meta-llama/Llama-3-70b-chat-hf' (a capable Llama 3 model)
    - Justification: Llama 3 70B models are state-of-the-art open models known for strong instruction
      following, reasoning, and generation quality, suitable for RAG.

    Prompt Design: (Handled in _construct_llm_messages)
    - System Instructions: Sets the persona, constraints (use only context), and desired output style.
    - Context Injection: Clearly demarcates the retrieved context for the LLM.
    - Question Formatting: Presents the user's query clearly after the context.

    Parameters Justification:
    - max_tokens ({DEFAULT_MAX_TOKENS_GENERATE}): Limits response length to prevent overly verbose or truncated answers.
    - temperature ({DEFAULT_TEMPERATURE_GENERATE}): A lower value (e.g., 0.2-0.5) makes the output more focused,
      deterministic, and factual, which is desirable for context-based Q&A.
    - top_p ({DEFAULT_TOP_P_GENERATE}): Nucleus sampling; combined with temperature, it controls token selection
      randomness. 0.9 is a common value that allows some flexibility while maintaining coherence.
    """
    if not TOGETHER_API_KEY:
        return {"error": "TOGETHER_API_KEY is not configured in environment variables."}
    if not context_chunks: # Should ideally be handled before calling, but as a safeguard
        return {"answer": "No context was provided to the LLM.", "model_used": None, "usage": None}

    messages = _construct_llm_messages(query, context_chunks)
    
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    models_to_try = [LLM_MODEL_NAME_PRIMARY, LLM_MODEL_NAME_FALLBACK]
    last_error = None

    async with httpx.AsyncClient(timeout=90) as client:
        for model_name in models_to_try:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False # We want the full response at once
            }
            
            print(f"Attempting async LLM call with model: {model_name}")
            try:
                # Use await with the async client
                response = await client.post(
                    f"{TOGETHER_API_BASE_URL}/chat/completions", 
                    headers=headers, 
                    json=payload
                )
                response.raise_for_status()
                
                response_data = response.json()
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    answer = response_data["choices"][0].get("message", {}).get("content", "").strip()
                    usage = response_data.get("usage", {})
                    print(f"LLM call successful with {model_name}. Usage: {usage}")
                    return {"answer": answer, "model_used": model_name, "usage": usage}
                else:
                    last_error = f"LLM response format unexpected or empty from {model_name}. Details: {response_data}"
                    print(last_error)
            
            except httpx.HTTPStatusError as e:
                last_error = f"HTTPError with {model_name}: {e.response.status_code} - {e.response.text if e.response else 'No response body'}"
                print(last_error)
                if e.response and e.response.status_code == 404: # Model not found, try next
                    continue
                if e.response and e.response.status_code == 429: # Rate limit
                    return {"error": f"Rate limited by LLM API with {model_name}. Please try again later. Details: {last_error}"}

            except httpx.RequestError as e:
                last_error = f"RequestException with {model_name}: {str(e)}"
                print(last_error)
            
        # If primary model failed for reasons other than 404, and there's a fallback, loop will try fallback.
        # If it's the last model in the list or a non-404 error on primary, this attempt ends.

    # If all models failed
    return {"error": f"LLM API request failed for all attempted models. Last error: {last_error}"}