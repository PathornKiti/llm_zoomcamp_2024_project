# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langfuse.callback import CallbackHandler
from langfuse.decorators import observe
from typing import Dict
from llm import compile_llm_pipeline 
import os

app = FastAPI()

# Initialize Langfuse handler
langfuse_handler = CallbackHandler(
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    host=os.environ.get("LANGFUSE_HOST")
)

# Initialize the LLM pipeline
conversational_rag_chain = compile_llm_pipeline()

# Define the request body structure
class QueryRequest(BaseModel):
    input: str
    session_id: str

def get_llm_response(user_input: str, session_id: str) -> Dict:
    """Get a response from the LLM with Langfuse tracking."""
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": session_id},
            "callbacks": [langfuse_handler]
        }
    )
    return response["answer"]

# API endpoint for querying the LLM
@app.post("/query")
async def query_llm(request: QueryRequest):
    try:
        user_input = request.input
        session_id = request.session_id
        response = get_llm_response(user_input, session_id)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "API is running"}

