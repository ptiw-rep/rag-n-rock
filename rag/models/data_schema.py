from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime

class RegisterRequest(BaseModel):
    username: str
    password: str

class FileUploadResponse(BaseModel):
    id: int
    filename: str

class FileListItem(BaseModel):
    id: int
    filename: str
    upload_time: datetime
    file_metadata: str

class FileListResponse(BaseModel):
    files: List[FileListItem]

class ChatRequest(BaseModel):
    """
    Chat request for hybrid RAG retrieval.
    - question: User query
    - file_id: (optional) restrict search to a file
    - keywords: (optional) boost/filter by keywords
    - metadata_filter: (optional) dict to restrict by metadata
    - use_mmr: (optional) use MMR re-ranking
    - k: (optional) number of chunks to retrieve
    """
    question: str = Field(..., min_length=1)
    file_id: Optional[int] = None
    keywords: Optional[List[str]] = None
    metadata_filter: Optional[dict] = None
    k: Optional[int] = 4

class ChatResponse(BaseModel):
    answer: str
    sources: List[Any]

class AdminClearAllResponse(BaseModel):
    status: str
    files_deleted: int
    chats_deleted: int
