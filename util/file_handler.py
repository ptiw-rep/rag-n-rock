from datetime import datetime
import os
import shutil
import uuid
from fastapi import UploadFile, HTTPException, File, Depends
from sqlalchemy.orm import Session

from util import logger
from database.models import File as DBFile
from rag.rag_pipeline import ALLOWED_FILE_EXTENSIONS

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/files"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.debug(f"File upload directory set to: {UPLOAD_DIR}")

def upload_file(file: UploadFile = File(...), db: Session = Depends(), user_id: str = None):
    logger.info(f"Received file upload request: {file.filename} (User ID: {user_id})")

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        logger.warning(f"Unsupported file type uploaded: {ext} (User: {user_id})")
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    logger.debug(f"Saving file to: {save_path}")
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved successfully: {file.filename} (ID: {file_id})")
    except Exception as e:
        logger.error(f"Failed to write file to disk: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="File upload failed due to internal error.")

    db_file = DBFile(
        filename=file.filename,
        filepath=save_path,
        upload_time=datetime.utcnow(),
        file_metadata="{}",
        user_id=user_id
    )

    try:
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        logger.info(f"File record created in DB: {db_file.id} - {file.filename}")
        return db_file
    
    except Exception as e:
        logger.error(f"Failed to store file metadata in DB: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="File upload failed.")

def list_files(db: Session = Depends(), user_id: str = None):
    logger.info(f"Fetching files for user: {user_id}")
    try:
        files = db.query(DBFile).filter(DBFile.user_id == user_id).all()
        logger.debug(f"Found {len(files)} files for user: {user_id}")
        return files
    except Exception as e:
        logger.error(f"Error listing files for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to retrieve files.")

def delete_file(file_id: int, db: Session = Depends()):
    logger.info(f"Deleting file with ID: {file_id}")
    db_file = db.query(DBFile).filter(DBFile.id == file_id).first()

    if not db_file:
        logger.warning(f"File not found: {file_id}")
        raise HTTPException(status_code=404, detail="File not found")

    errors = []

    try:
        os.remove(db_file.filepath)
        logger.info(f"Successfully deleted file from disk: {db_file.filepath}")
    except FileNotFoundError:
        logger.warning(f"File already deleted or missing: {db_file.filepath}")
    except Exception as e:
        err_msg = f"Failed to delete file from disk: {str(e)}"
        logger.error(err_msg, exc_info=True)
        errors.append(err_msg)

    try:
        db.delete(db_file)
        db.commit()
        logger.info(f"File record deleted from DB: {file_id}")
    except Exception as e:
        logger.error(f"Failed to delete file record from DB: {str(e)}", exc_info=True)
        db.rollback()
        errors.append("Database cleanup failed.")

    if errors:
        logger.warning(f"File deletion completed with warnings: {errors}")
        for err in errors:
            logger.warning(err)
    else:
        logger.info(f"File {file_id} deleted successfully.")
        return {"status": "deleted", "warnings": errors}
