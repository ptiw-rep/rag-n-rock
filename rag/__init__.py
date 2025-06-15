# rag/__init__.py
import os

from .rag_pipeline import RAGPipeline
from .llm_provider import LLMProvider

CHROMA_PATH = os.path.abspath("./data/chroma_db")

# Common LLM Provider instantiation for the Server.
model_provider = LLMProvider(embedding_model="nomic-embed-text:latest", llm_model="gemma3:4b")

# Common object instatiation for the RAG pipeline.
rag_pipeline = RAGPipeline(model_provider=model_provider, vector_db_path=CHROMA_PATH)