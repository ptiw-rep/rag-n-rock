from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
import os

from rag.models.data_schema import FileUploadResponse,  FileListItem
from util.file_handler import (
    upload_file,
    list_files,
    delete_file,
    ALLOWED_FILE_EXTENSIONS
)
from database.models import File as DBFile, User
from rag.rag_pipeline import RAGPipeline
from rag import rag_pipeline
from util import logger
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
    logger.info(f"User '{current_user['username']}' initiated file upload: {file.filename}")

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        logger.warning(f"Unsupported file type uploaded: {ext} by user '{current_user['username']}'")
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {ext}")

    # Get or create the user in DB based on current_user['username']
    db_user = db.query(User).filter(User.username == current_user["username"]).first()
    if not db_user:
        logger.error(f"User not found: {current_user['username']}")
        raise HTTPException(status_code=404, detail="User not found")

    try:
        logger.debug(f"Uploading file: {file.filename} for user ID={db_user.id}")
        db_file = upload_file(file=file, db=db, user_id=db_user.id)

        logger.info(f"File saved: {db_file.filename} (ID: {db_file.id})")
    except Exception as e:
        logger.error(f"File upload failed for user '{current_user['username']}': {str(e)}", exc_info=True)
        raise

    # Ingest into RAG pipeline
    try:
        logger.info(f"Ingesting file {db_file.id} into RAG pipeline")
        rag_pipeline.ingest(db_file.filepath, metadata={"file_id": db_file.id, "filename": db_file.filename})
        logger.info(f"Successfully ingested file into RAG: {db_file.filename}")
        return FileUploadResponse(id=db_file.id, filename=db_file.filename)

    except Exception as e:
        logger.error(f"RAG ingestion failed for file {db_file.id}: {str(e)}", exc_info=True)
        db.delete(db_file)
        db.commit()
        os.remove(db_file.filepath)
        raise HTTPException(status_code=500, detail=f"RAG ingestion failed: {str(e)}")

@router.get("/files", response_model=list[FileListItem])
def list_files_in_db(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
    ) -> list[FileListItem]:
    """
    List all files in the database. Delegates business logic to file_service.
    """
    logger.info(f"Fetching files for user: {current_user['username']}")

    db_user = db.query(User).filter(User.username == current_user["username"]).first()
    if not db_user:
        logger.error(f"User not found: {current_user['username']}")
        raise HTTPException(status_code=404, detail="User not found")

    try:
        files = list_files(db=db, user_id=db_user.id)
        logger.info(f"Found {len(files)} files for user '{current_user['username']}'")
        return [
            FileListItem(
                id=f.id,
                filename=f.filename,
                upload_time=f.upload_time,
                file_metadata=f.file_metadata
            ) for f in files
        ]
    except Exception as e:
        logger.error(f"Failed to list files for user '{current_user['username']}': {str(e)}", exc_info=True)
        raise

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
    logger.info(f"User '{current_user['username']}' requested deletion of file ID: {file_id}")

    db_file = db.query(DBFile).filter(DBFile.id == file_id).first()
    if not db_file:
        logger.warning(f"File not found: {file_id}")
        raise HTTPException(status_code=404, detail="File not found")

    db_user = db.query(User).filter(User.username == current_user["username"]).first()
    if not db_user:
        logger.error(f"User not found: {current_user['username']}")
        raise HTTPException(status_code=404, detail="User not found")

    if db_file.user_id != db_user.id:
        logger.warning(f"Unauthorized attempt to delete file {file_id} by user {current_user['username']}")
        raise HTTPException(status_code=403, detail="You are not authorized to delete this file")

    try:
        result = delete_file(file_id=file_id, db=db)
        errors = result.get("warnings", [])

        logger.info(f"Deleting embeddings for file {file_id} from vectorstore")
        try:
            rag_pipeline.vectorstore.delete(where={"file_id": file_id})
            logger.info(f"Removed embeddings via 'file_id' filter for file {file_id}")
        except Exception as e:
            error_msg = f"Vectorstore delete by file_id failed: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

        try:
            db_file = db.query(DBFile).filter(DBFile.id == file_id).first()
            if db_file:
                rag_pipeline.vectorstore.delete(where={"filename": db_file.filename})
                logger.info(f"Removed embeddings via 'filename' filter for {db_file.filename}")
        except Exception as e:
            error_msg = f"Vectorstore delete by filename failed: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

        # Reinitialize vectorstore
        rag_pipeline.vectorstore = RAGPipeline(vector_db_path=rag_pipeline.vector_db_path).vectorstore
        logger.info("Vectorstore reinitialized after deletion")

        if errors:
            logger.warning(f"Partial success during file deletion. Warnings: {errors}")
            return {"status": "partial_success", "warnings": errors}
        else:
            logger.info(f"File {file_id} deleted successfully.")
            return {"status": "deleted", "warnings": errors}

    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}", exc_info=True)
        raise