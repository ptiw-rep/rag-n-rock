from fastapi import UploadFile, HTTPException, File, Depends
from sqlalchemy.orm import Session
from database.models import File as DBFile
from data_plane.rag_pipeline import ALLOWED_FILE_EXTENSIONS

from datetime import datetime
import os
import shutil
import uuid

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/files"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# TODO; Check all invocations and add User Id param
def upload_file(file: UploadFile = File(...), db: Session = Depends(), user_id: str = None):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    db_file = DBFile(
        filename=file.filename,
        filepath=save_path,
        upload_time=datetime.utcnow(),
        file_metadata="{}",
        user_id=user_id
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

def list_files(db: Session = Depends(), user_id: str = None):
    files = db.query(DBFile).filter(DBFile.user_id == user_id).all()
    return files

def delete_file(file_id: int, db: Session = Depends()):
    db_file = db.query(DBFile).filter(DBFile.id == file_id).first()
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    errors = []
    try:
        os.remove(db_file.filepath)

    except FileNotFoundError:
        pass

    except Exception as e:
        errors.append(f"File delete error: {e}")

    db.delete(db_file)
    db.commit()

    if errors:
        for err in errors:
            print(f"[DeleteFile] {err}")
    else:
        print(f"[DeleteFile] File {file_id} deleted successfully at {datetime.now().isoformat()}")
    
    return {"status": "deleted", "warnings": errors}
