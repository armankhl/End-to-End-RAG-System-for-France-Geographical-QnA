# main.py
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Local module imports
import config
from retrieval import retrieve_relevant_chunks, _load_retrieval_resources_if_needed
from generation import generate_llm_response
from re_ranking import re_rank_chunks

# Initialize FastAPI app
app = FastAPI(
    title="France Geography RAG API",
    description="API for retrieving information and generating answers about the geography and land features of France using a RAG pipeline.",
    version="1.0.3" # Incremented version
)

# --- Pydantic Models for API Request/Response Schemas ---

class BaseQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The user's query or question about France's geography.")

class RetrievalFilters(BaseModel):
    category_filter: Optional[str] = Field(default=None, description="Filter chunks by category (e.g., 'Climate', 'Soils'). Case-insensitive, partial match.")
    date_before_filter: Optional[str] = Field(default=None, description="Filter chunks with 'publication_date' before this date (YYYY-MM-DD or Month DD, YYYY).")
    date_after_filter: Optional[str] = Field(default=None, description="Filter chunks with 'publication_date' after this date (YYYY-MM-DD or Month DD, YYYY).")

class RetrieveRequest(BaseQueryRequest, RetrievalFilters):
    top_k: int = Field(default=config.DEFAULT_TOP_K_RETRIEVAL, ge=1, le=20, description="Number of top relevant chunks to retrieve.")

# CORRECTED SCHEMA: Added re_rank_score
class RetrievedChunk(BaseModel):
    id: str
    text: str
    metadata: Dict
    score: float # Initial retrieval score from vector search
    re_rank_score: Optional[float] = Field(default=None, description="Score from the re-ranker model, if used.")

class RetrieveResponse(BaseModel):
    query: str
    retrieved_chunks: List[RetrievedChunk]
    message: Optional[str] = Field(default=None, description="Additional information about the retrieval process.")

class GenerateRequest(RetrieveRequest): # Inherits query, top_k, and filters
    use_reranker: bool = Field(default=True, description="Whether to use a re-ranking step after initial retrieval.")
    max_tokens: int = Field(default=config.DEFAULT_MAX_TOKENS_GENERATE, ge=50, le=4000, description="Maximum number of tokens for the LLM's generated response.")
    temperature: float = Field(default=config.DEFAULT_TEMPERATURE_GENERATE, ge=0.0, le=2.0, description="LLM temperature for generation. Controls randomness.")
    top_p: Optional[float] = Field(default=config.DEFAULT_TOP_P_GENERATE, ge=0.0, le=1.0, description="LLM top_p for nucleus sampling.")

class GenerateResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks_for_context: List[RetrievedChunk]
    llm_model_used: Optional[str] = Field(default=None, description="The LLM model that generated the answer.")
    llm_usage: Optional[Dict] = Field(default=None, description="Token usage statistics from the LLM API.")
    error_message: Optional[str] = Field(default=None, description="Error message if generation failed or had issues.")


# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    print("Application is starting up...")
    try:
        _load_retrieval_resources_if_needed()
        print("Retrieval resources loaded successfully.")
        if not config.TOGETHER_API_KEY:
            print("WARNING: TOGETHER_API_KEY environment variable is not set. The /generate endpoint will be unavailable.")
        else:
            print("TOGETHER_API_KEY found.")
    except Exception as e:
        print(f"FATAL STARTUP ERROR: {e}")
    print("Application startup complete.")


# --- API Endpoints ---

@app.post("/retrieve", response_model=RetrieveResponse, tags=["RAG Pipeline"])
async def retrieve_endpoint(request: RetrieveRequest):
    """
    Retrieves relevant text chunks from the knowledge base based on the query and optional filters.
    This endpoint performs vector search only and does not use the re-ranker.
    """
    try:
        # Sanitize filter inputs to handle Swagger UI defaults
        category = request.category_filter if request.category_filter and request.category_filter.lower() != "string" else None
        date_before = request.date_before_filter if request.date_before_filter and request.date_before_filter.lower() != "string" else None
        date_after = request.date_after_filter if request.date_after_filter and request.date_after_filter.lower() != "string" else None

        chunks = retrieve_relevant_chunks(
            query=request.query,
            top_k=request.top_k,
            category=category,
            date_before=date_before,
            date_after=date_after
        )
        message = None
        if not chunks and (category or date_before or date_after):
            message = "No chunks found matching the specified metadata filters and query."
        elif not chunks:
            message = "No relevant chunks found for the query."
            
        return RetrieveResponse(query=request.query, retrieved_chunks=chunks, message=message)
    except Exception as e:
        print(f"Unexpected error in /retrieve: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during retrieval: {str(e)}")


@app.post("/generate", response_model=GenerateResponse, tags=["RAG Pipeline"])
async def generate_endpoint(request: GenerateRequest):
    """
    Generates an answer to the query using retrieved context and an LLM.
    Optionally uses a re-ranking step for improved context relevance.
    """
    if not config.TOGETHER_API_KEY:
        raise HTTPException(status_code=503, detail="LLM generation service is unavailable: TOGETHER_API_KEY not configured.")

    try:
        # Sanitize filter inputs
        category = request.category_filter if request.category_filter and request.category_filter.lower() != "string" else None
        date_before = request.date_before_filter if request.date_before_filter and request.date_before_filter.lower() != "string" else None
        date_after = request.date_after_filter if request.date_after_filter and request.date_after_filter.lower() != "string" else None

        # Determine how many initial chunks to retrieve.
        # If using re-ranker, get more candidates to give it more to work with.
        initial_k_to_retrieve = (request.top_k + 5) if request.use_reranker else request.top_k

        # 1. Initial Retrieval
        retrieved_chunks = retrieve_relevant_chunks(
            query=request.query,
            top_k=initial_k_to_retrieve,
            category=category,
            date_before=date_before,
            date_after=date_after
        )

        if not retrieved_chunks:
            return GenerateResponse(
                query=request.query,
                answer="I could not find relevant information in the knowledge base to answer your question.",
                retrieved_chunks_for_context=[],
                error_message="No relevant context chunks found during initial retrieval."
            )

        # 2. Optional Re-ranking
        if request.use_reranker:
            final_context_chunks = re_rank_chunks(request.query, retrieved_chunks)
            # The top chunks to be passed to the LLM are the first `top_k` from the re-ranked list
            context_for_llm = final_context_chunks[:request.top_k]
        else:
            final_context_chunks = retrieved_chunks
            # If not re-ranking, the context is already the top_k retrieved chunks
            context_for_llm = final_context_chunks

        # 3. Generation Step
        llm_result = await generate_llm_response(
            query=request.query,
            context_chunks=context_for_llm, # Pass the refined context to the LLM
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        if "error" in llm_result:
            return GenerateResponse(
                query=request.query,
                answer="", # No answer from LLM
                retrieved_chunks_for_context=final_context_chunks, # Still show what context was attempted
                error_message=f"LLM Error: {llm_result['error']}"
            )

        return GenerateResponse(
            query=request.query,
            answer=llm_result.get("answer", "Error: LLM did not return a valid answer."),
            retrieved_chunks_for_context=final_context_chunks, # Show the final (potentially re-ranked) list
            llm_model_used=llm_result.get("model_used"),
            llm_usage=llm_result.get("usage")
        )
    except Exception as e:
        print(f"An unexpected error occurred in /generate endpoint:")
        traceback.print_exc() # Print full traceback to server console for debugging
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred: {str(e)}")