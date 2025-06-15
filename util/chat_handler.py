import os, re
from sqlalchemy.orm import Session
from database.models import File as DBFile, ChatHistory, User
from datetime import datetime
from fastapi import HTTPException
from typing import Optional, Dict, Any

from util import logger

def chat_service(
    question: str,
    file_id: Optional[int],
    db: Session,
    rag_pipeline,
    ollama_llm,
    current_user: dict,
    keywords: Optional[list] = None,
    metadata_filter: Optional[dict] = None,
    k: Optional[int] = 4
) -> Dict[str, Any]:
    """
    Handles chat logic: retrieves relevant docs (hybrid/vector/keyword/MMR), constructs prompt, calls LLM, logs history.
    """
    logger.info(f"Chat request received from user: {current_user.get('username', 'unknown')}")
    logger.debug("Starting chat_service execution")
    try:
        # Get user from DB based on username
        db_user = db.query(User).filter(User.username == current_user["username"]).first()
        if not db_user:
            logger.warning("User not found in DB")
            raise HTTPException(status_code=404, detail="User not found")

        logger.debug(f"User authenticated: {db_user.username}, ID={db_user.id}")

        # Fetch all files owned by this user
        db_files = db.query(DBFile).filter(DBFile.user_id == db_user.id).all()

        if not db_files:
            logger.info("No files available for answering.")
            return {
                "answer": "No files are available for answering. Please upload a file first.",
                "sources": []
            }

        # Validate file_id if provided
        if file_id:
            valid_file_ids = {f.id for f in db_files}
            if file_id not in valid_file_ids:
                logger.warning(f"User tried to access unauthorized file ID: {file_id}")
                raise HTTPException(status_code=403, detail="You do not have access to this file")

            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter["file_id"] = file_id
            logger.debug(f"Filtering by file_id: {file_id}")

        # Retrieve documents
        logger.info("Starting RAG pipeline retrieval")
        docs = rag_pipeline.retrieve(
            question,
            k=k,
            keywords=keywords,
            metadata_filter=metadata_filter
        )
        logger.debug(f"Retrieved {len(docs)} documents from RAG pipeline")

        # Filter documents by valid file IDs
        current_file_ids = {str(f.id) for f in db_files}
        docs = [d for d in docs if str(d.metadata.get("file_id")) in current_file_ids]
        logger.debug(f"Filtered down to {len(docs)} valid documents")

        # Build context
        context = "\n\n".join([d.page_content for d in docs])
        system_prompt = (
            "You are a helpful AI assistant. Always answer in well-structured markdown. "
            "Use headings, bullet points, spacing and tables where appropriate. "
            "Format code and data for maximum readability."
            "Keep it concise and to the point."
            "If you don't know the answer, say so.\n"
        )
        prompt = f"{system_prompt}Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        # Call LLM
        logger.info("Invoking LLM inference")
        answer = ollama_llm.invoke(prompt)
        logger.debug(f"LLM response: '{answer[:100]}...'")

        # Log chat history
        chat = ChatHistory(
            user_id=db_user.id,
            file_id=file_id,
            question=question,
            answer=answer,
            timestamp=datetime.utcnow()
        )
        db.add(chat)
        db.commit()
        logger.info("Chat history logged successfully")

        # Process sources
        file_counts = {}
        for d in docs:
            name = d.metadata.get('source') or d.metadata.get('filename') or d.metadata.get('file_name')
            if name:
                base = os.path.basename(name)
                base = re.sub(r'^[0-9a-fA-F-]+_', '', base)
            else:
                base = 'Unknown File'
            file_counts[base] = file_counts.get(base, 0) + 1


        # Find the file with the most chunks in the answer
        if file_counts:
            most_relevant_file = max(file_counts, key=file_counts.get)
        else:
            most_relevant_file = 'Unknown File'
        
        logger.info(f"Chat completed successfully. Answer generated using: {most_relevant_file}")
        return {"answer": answer, "sources": [most_relevant_file]}
    
    except Exception as e:
        logger.error(f"Error during chat processing: {str(e)}", exc_info=True)
        db.rollback()
        raise