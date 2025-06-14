import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database.db_session import get_db, init_db
from database.models import File as DBFile, User

from data_plane.models.data_schema import ChatRequest
from data_plane.rag_pipeline import RAGPipeline, ALLOWED_FILE_EXTENSIONS
from data_plane.models.data_schema import FileUploadResponse, FileListItem, ChatResponse, AdminClearAllResponse

from util.error_handler import http_exception_handler, sqlalchemy_exception_handler, generic_exception_handler
from util.sudo_handler import clear_all_service
from util.file_handler import upload_file, list_files, delete_file
from util.chat_handler import chat_service

from langchain_ollama import OllamaLLM
ollama_llm = OllamaLLM(model="gemma3:4b")

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/files"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="RAG-narok")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register centralized error handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "./data/chroma_db"))
os.makedirs(CHROMA_PATH, exist_ok=True)

# Initialize DB and RAG pipeline
init_db()
rag_pipeline = RAGPipeline(vector_db_path=CHROMA_PATH)

SECRET_KEY = "my-secret-key"
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return {"username": username}
    
    except JWTError:
        raise credentials_exception

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

@app.post("/api/upload", response_model=FileUploadResponse)
def upload_file_in_db(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
    ) -> FileUploadResponse:
    """
    Upload a file and ingest into the RAG pipeline. Delegates business logic to file_service.
    Validates file extension against SUPPORTED_EXTENSIONS.
    """
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {ext}")
    
    # Get or create the user in DB based on current_user['username']
    db_user = db.query(User).filter(User.username == current_user["username"]).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db_file = upload_file(file=file, db=db, user_id=db_user.id)
    # Ingest into RAG pipeline (kept here to avoid circular dependency)
    try:
        rag_pipeline.ingest(db_file.filepath, metadata={"file_id": db_file.id, "filename": db_file.filename})
    except Exception as e:
        db.delete(db_file)
        db.commit()
        os.remove(db_file.filepath)
        raise HTTPException(status_code=500, detail=f"RAG ingestion failed: {str(e)}")
    return FileUploadResponse(id=db_file.id, filename=db_file.filename)

@app.get("/api/files", response_model=list[FileListItem])
def list_files_in_db(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
    ) -> list[FileListItem]:
    """
    List all files in the database. Delegates business logic to file_service.
    """
    # Get user from DB based on username
    db_user = db.query(User).filter(User.username == current_user["username"]).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    files = list_files(db=db, user_id=db_user.id)
    return [
        FileListItem(
            id=f.id,
            filename=f.filename,
            upload_time=f.upload_time,
            file_metadata=f.file_metadata
        ) for f in files
    ]

@app.delete("/api/files/{file_id}")
def delete_file_in_db(
    file_id: int, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
    ) -> dict:
    """
    Delete a file: removes from DB and disk, attempts vectorstore cleanup. 
    Delegates business logic to file_service for DB/disk, keeps vectorstore logic here to avoid circular dependency.
    """
    # Get the file first
    db_file = db.query(DBFile).filter(DBFile.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")

    # Get the current user from the DB
    db_user = db.query(User).filter(User.username == current_user["username"]).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Ensure current user owns this file
    if db_file.user_id != db_user.id:
        raise HTTPException(status_code=403, detail="You are not authorized to delete this file")
    
    # Remove from DB and disk
    result = delete_file(file_id=file_id, db=db)

    # Remove from vectorstore
    errors = result.get("warnings", [])
    try:
        rag_pipeline.vectorstore.delete(where={"file_id": file_id})
        
        rag_pipeline.vectorstore = RAGPipeline(vector_db_path=rag_pipeline.vector_db_path).vectorstore

    except Exception as e:
        errors.append(f"Vectorstore delete by file_id (where) error: {e}")
        print(f"[DeleteFile] Chroma delete API mismatch or failure: {e}")

    try:
        # Try by filename (if file still exists)
        db_file = db.query(DBFile).filter(DBFile.id == file_id).first()
        if db_file:
            rag_pipeline.vectorstore.delete(where={"filename": db_file.filename})
            rag_pipeline.vectorstore = RAGPipeline(vector_db_path=rag_pipeline.vector_db_path).vectorstore

    except Exception as e:
        errors.append(f"Vectorstore delete by filename (where) error: {e}")
        print(f"[DeleteFile] Chroma delete API mismatch or failure: {e}")

    return {"status": "deleted", "warnings": errors}

@app.post("/api/chat", response_model=ChatResponse)
def chat(
    chat_req: ChatRequest = None,
    question: str = Query(None, min_length=3, max_length=500),
    file_id: int = Query(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
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
