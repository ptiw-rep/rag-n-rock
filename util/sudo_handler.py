from sqlalchemy.orm import Session
from database.models import File as DBFile, ChatHistory
from data_plane.rag_pipeline import RAGPipeline
from fastapi import HTTPException
from datetime import datetime
from typing import Any, Dict

def clear_all_service(admin_token: str, db: Session, rag_pipeline, admin_env_token: str) -> Dict[str, Any]:
    """
    Danger: Delete ALL files and chat history from DB and vectorstore. Requires admin-token.
    """
    if admin_token != admin_env_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Delete all chat history
    chat_count = db.query(ChatHistory).delete()
    # Delete all files
    file_count = db.query(DBFile).delete()
    db.commit()
    # Delete all embeddings from vectorstore
    try:
        vec_data = rag_pipeline.vectorstore.get()
        all_ids = vec_data.get('ids', [])
        if all_ids:
            rag_pipeline.vectorstore.delete(ids=all_ids)

        rag_pipeline.vectorstore = RAGPipeline(vector_db_path=rag_pipeline.vector_db_path).vectorstore
        print(f"[AdminClearAll] All files and chats deleted at {datetime.now().isoformat()}")
    except Exception as e:
        print(f"[AdminClearAll] Vectorstore clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vectorstore clear failed: {e}")
    return {"status": "cleared", "files_deleted": file_count, "chats_deleted": chat_count}