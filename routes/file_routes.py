import logging
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
import os

from data_plane.models.data_schema import FileUploadResponse,  FileListItem
from util.file_handler import (
    upload_file,
    list_files,
    delete_file,
    ALLOWED_FILE_EXTENSIONS
)
from database.models import File as DBFile, User
from data_plane.rag_pipeline import RAGPipeline
from data_plane import rag_pipeline

from .auth_routes import get_current_user
from database.db_session import get_db

router = APIRouter()


@router.post("/upload", response_model=FileUploadResponse)
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

@router.get("/files", response_model=list[FileListItem])
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

@router.delete("/files/{file_id}")
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