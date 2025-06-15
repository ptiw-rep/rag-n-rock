import os

from fastapi import FastAPI, UploadFile, HTTPException, status, Depends, Query, Header, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import timedelta

from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database.db_session import get_db, init_db
from database.models import File as DBFile, User

from data_plane.rag_pipeline import RAGPipeline, ALLOWED_FILE_EXTENSIONS
from data_plane.models.data_schema import ChatRequest, RegisterRequest
from data_plane.models.data_schema import FileUploadResponse, FileListItem, ChatResponse, AdminClearAllResponse
from routes import file_routes

from routes import auth_routes
from util.sudo_handler import clear_all_service
from util.chat_handler import chat_service
from util.error_handler import (
    http_exception_handler, 
    sqlalchemy_exception_handler, 
    generic_exception_handler
    )
from util.file_handler import (
    upload_file, 
    list_files, 
    delete_file
    )
from util.auth_handler import (
    get_password_hash, 
    authenticate_user, 
    create_access_token, 
    ACCESS_TOKEN_EXPIRE_MINUTES,
    )

from langchain_ollama import OllamaLLM
ollama_llm = OllamaLLM(model="gemma3:4b")

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/files"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="RAG-n-rock")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define and add routes for each module.
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])
app.include_router(file_routes.router, prefix="/api/files", tags=["files"])

# Register centralized error handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data/chroma_db"))
os.makedirs(CHROMA_PATH, exist_ok=True)

# Initialize DB and RAG pipeline
init_db()
rag_pipeline = RAGPipeline(vector_db_path=CHROMA_PATH)



@app.post("/api/admin/clear_all", response_model=AdminClearAllResponse)
def clear_all(
    admin_token: str = Header(..., alias="admin-token", min_length=8, max_length=128), 
    db: Session = Depends(get_db)
    ) -> AdminClearAllResponse:
    """
    Danger: Delete ALL files and chat history from DB and vectorstore.
    Validates admin token length (8-128 chars).
    """
    ADMIN_TOKEN = os.environ.get("CHAT_RAG_ADMIN_TOKEN", "supersecret")
    if not (admin_token.isalnum() or '-' in admin_token or '_' in admin_token):
        raise HTTPException(status_code=422, detail="Invalid admin token format.")
    return AdminClearAllResponse(**clear_all_service(
        admin_token=admin_token,
        db=db,
        rag_pipeline=rag_pipeline,
        admin_env_token=ADMIN_TOKEN
    ))

@app.get("/api/health")
def health_check():
    """
    Health check endpoint: verifies DB, vectorstore, and LLM health.
    """
    # DB health
    try:
        db_ok = True
        db_msg = "OK"
        session_gen = get_db()
        session = next(session_gen)
        session.execute(text("SELECT 1"))
        session.close()
    except Exception as e:
        db_ok = False
        db_msg = str(e)
    # Vectorstore health
    try:
        vec_ok = True
        vec_msg = "OK"
        _ = rag_pipeline.vectorstore.get()
    except Exception as e:
        vec_ok = False
        vec_msg = str(e)

    # LLM health
    try:
        llm_ok = True
        llm_msg = "OK"
        _ = ollama_llm.invoke("Health check?")
        llm_model = getattr(ollama_llm, 'model', None)
        if not llm_model and hasattr(ollama_llm, '__dict__'):
            llm_model = ollama_llm.__dict__.get('model')

    except Exception as e:
        llm_ok = False
        llm_msg = str(e)
        llm_model = getattr(ollama_llm, 'model', None)
        if not llm_model and hasattr(ollama_llm, '__dict__'):
            llm_model = ollama_llm.__dict__.get('model')
    status = "ok" if all([db_ok, vec_ok, llm_ok]) else "degraded"
    
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
    if chat_req is not None:
        req = chat_req
    else:
        req = ChatRequest(question=question, file_id=file_id)
    return ChatResponse(
        **chat_service(
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
    )
