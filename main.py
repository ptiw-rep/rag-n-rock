import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database.db_session import get_db, init_db

from rag import rag_pipeline, model_provider, CHROMA_PATH
from config import get_env
from util import logger

from rag.models.data_schema import ChatRequest
from rag.models.data_schema import  ChatResponse, AdminClearAllResponse

from util.sudo_handler import clear_all_service
from util.chat_handler import chat_service
from util.error_handler import (
    http_exception_handler, 
    sqlalchemy_exception_handler, 
    generic_exception_handler
    )

from routes import file_routes
from routes import auth_routes

ollama_llm = model_provider.get_inference_model()

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/files"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize DB and RAG pipeline
init_db()
logger.info("Database and RAG pipeline initialized at startup.")

app = FastAPI(title="RAG-n-rock")
logger.info("FastAPI app initialized.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.debug("CORS middleware configured to allow all origins.")

# Define and add routes for each module.
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])
logger.info("Auth router included at /api/auth")
app.include_router(file_routes.router, prefix="/api", tags=["files"])
logger.info("File router included at /api")

# Register centralized error handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
logger.debug("Global exception handlers registered.")

os.makedirs(CHROMA_PATH, exist_ok=True)

# Base routes definition added here to reduce unnecessary complexity with API Router.
@app.post("/api/admin/clear_all", response_model=AdminClearAllResponse)
def clear_all(
    admin_token: str = Header(..., alias="admin-token", min_length=8, max_length=128), 
    db: Session = Depends(get_db)
    ) -> AdminClearAllResponse:
    """
    Danger: Delete ALL files and chat history from DB and vectorstore.
    Validates admin token length (8-128 chars).
    """
    logger.info("Received request to clear all data via /api/admin/clear_all")

    ADMIN_TOKEN = get_env("CHAT_RAG_ADMIN_TOKEN", "supersecret")
    if not (admin_token.isalnum() or '-' in admin_token or '_' in admin_token):
        logger.warning("Invalid admin token format.")
        raise HTTPException(status_code=422, detail="Invalid admin token format.")
    
    if admin_token != ADMIN_TOKEN:
        logger.warning("Unauthorized attempt to clear data. Invalid admin token provided.")
        raise HTTPException(status_code=403, detail="Invalid admin token.")

    try:
        logger.info("Starting clear_all_service...")
        result = clear_all_service(
            admin_token=admin_token,
            db=db,
            rag_pipeline=rag_pipeline,
            admin_env_token=ADMIN_TOKEN
        )
        logger.info("Successfully cleared all data.")
        return AdminClearAllResponse(**result)
    
    except Exception as e:
        logger.error(f"Failed to clear all data: {str(e)}", exc_info=True)
        raise

@app.get("/api/health")
def health_check():
    """
    Health check endpoint: verifies DB, vectorstore, and LLM health.
    """
    logger.debug("Health check initiated.")

    # DB health
    db_ok = False
    db_msg = ""
    try:
        session_gen = get_db()
        session = next(session_gen)
        session.execute(text("SELECT 1"))
        db_ok = True
        db_msg = "OK"
        logger.debug("Database connection successful.")
    except Exception as e:
        db_msg = str(e)
        logger.error(f"Database health check failed: {db_msg}", exc_info=True)
    finally:
        session.close()

    # Vectorstore health
    vec_ok = False
    vec_msg = ""
    try:
        _ = rag_pipeline.vectorstore.get()
        vec_ok = True
        vec_msg = "OK"
        logger.debug("Vectorstore is healthy.")
    except Exception as e:
        vec_msg = str(e)
        logger.error(f"Vectorstore health check failed: {vec_msg}", exc_info=True)

    # LLM health
    llm_ok = False
    llm_msg = ""
    llm_model: Optional[str] = None
    try:
        _ = ollama_llm.invoke("Health check?")
        llm_model = getattr(ollama_llm, 'model', None)
        if not llm_model and hasattr(ollama_llm, '__dict__'):
            llm_model = ollama_llm.__dict__.get('model')
        llm_ok = True
        llm_msg = "OK"
        logger.debug(f"LLM ({llm_model}) is responding.")

    except Exception as e:
        llm_msg = str(e)
        llm_ok = False
        logger.error(f"LLM health check failed: {llm_msg}", exc_info=True)

    status = "ok" if all([db_ok, vec_ok, llm_ok]) else "degraded"
    logger.info(f"Health check completed. Status: {status}")

    return {
        "status": status,
        "db": {"ok": db_ok, "msg": db_msg},
        "vectorstore": {"ok": vec_ok, "msg": vec_msg},
        "llm": {"ok": llm_ok, "msg": llm_msg, "model": llm_model}
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    chat_req: ChatRequest = None,
    question: str = Query(None, min_length=3, max_length=500),
    file_id: int = Query(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(auth_routes.get_current_user)
    ) -> ChatResponse:
    """
    Chat endpoint: supports hybrid retrieval (keywords, metadata, MMR, k). Accepts ChatRequest body or legacy query params.
    """
    # Prefer body if provided, else fallback to query params for legacy clients
    logger.info(f"New chat request received from user: {current_user.get("username", "unknown")}")
    
    if chat_req is not None:
        req = chat_req
    else:
        req = ChatRequest(question=question, file_id=file_id)

    try:
        logger.debug(f"Processing chat request: '{req.question[:50]}...' for file_id={req.file_id}")
        response = chat_service(
            question=req.question,
            file_id=req.file_id,
            db=db,
            rag_pipeline=rag_pipeline,
            ollama_llm=ollama_llm,
            current_user=current_user,
            keywords=req.keywords,
            metadata_filter=req.metadata_filter,
            k=req.k
        )
        logger.debug(f"Chat response generated: '{response['answer'][:50]}...'")
        return ChatResponse(**response)
    
    except Exception as e:
        logger.error(f"Error during chat processing: {str(e)}", exc_info=True)
        raise
