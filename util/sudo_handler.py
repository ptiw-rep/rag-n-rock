from fastapi import HTTPException
from datetime import datetime
from typing import Any, Dict
from sqlalchemy.orm import Session

from util import logger
from database.models import File as DBFile, ChatHistory
from rag.rag_pipeline import RAGPipeline


def clear_all_service(admin_token: str, db: Session, rag_pipeline, admin_env_token: str) -> Dict[str, Any]:
    """
    Danger: Delete ALL files and chat history from DB and vectorstore. Requires admin-token.
    """
    logger.info("Admin clear_all request received.")

    if admin_token != admin_env_token:
        logger.warning("Unauthorized attempt to clear all data: Invalid admin token")
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Delete all chat history
        chat_count = db.query(ChatHistory).delete()
        logger.debug(f"Deleted {chat_count} chat records from DB")

        # Delete all files
        file_count = db.query(DBFile).delete()
        logger.debug(f"Deleted {file_count} file records from DB")

        db.commit()
        logger.info(f"Successfully deleted {file_count} files and {chat_count} chat entries from the database")

        # Delete all embeddings from vectorstore
        vec_data = rag_pipeline.vectorstore.get()
        all_ids = vec_data.get('ids', [])
        if all_ids:
            logger.info(f"Deleting {len(all_ids)} embeddings from vectorstore")
            rag_pipeline.vectorstore.delete(ids=all_ids)

        # Reinitialize vectorstore to ensure clean state
        rag_pipeline.vectorstore = RAGPipeline(vector_db_path=rag_pipeline.vector_db_path).vectorstore
        logger.info("Vectorstore reinitialized successfully")

        logger.info("All data cleared successfully.")
        return {"status": "cleared", "files_deleted": file_count, "chats_deleted": chat_count}

    except Exception as e:
        logger.error(f"Error during clear_all operation: {str(e)}", exc_info=True)
        db.rollback()
        logger.debug("Database transaction rolled back due to error")
        raise HTTPException(status_code=500, detail=f"Failed to clear all data: {str(e)}")