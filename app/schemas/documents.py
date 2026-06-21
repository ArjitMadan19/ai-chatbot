from typing import Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    id: int
    file_name: str
    doc_type: str
    source: str
    chunks_indexed: int
    status: str
    error_message: Optional[str] = None


class DocumentResponse(BaseModel):
    id: int
    file_name: str
    doc_type: str
    source: str
    chunks_indexed: int
    status: str
    error_message: Optional[str] = None
