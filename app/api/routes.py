import shutil
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.rag.memory import select_conversation_history_for_query
from app.rag.pipeline import (
    LLMGenerationError,
    VectorStoreError,
    ask_rag
)
from app.schemas.chat import AskRequest, AskResponse, ChatSessionResponse
from app.schemas.documents import DocumentResponse, UploadResponse
from app.services.cache import build_cache_key, get_cached_answer, set_cached_answer
from app.services.config import settings
from app.services.database import (
    create_chat_session,
    get_chat_session,
    get_document_record,
    get_db,
    get_recent_chat_messages,
    save_chat_message,
    save_document_record
)
from app.services.ingestion import index_uploaded_document
from app.services.serialization import serialize_sources
from app.services.uploads import get_upload_path


router = APIRouter()


@router.get("/")
def health_check(request: Request):
    return {
        "status": "ok",
        "message": "RAG Chatbot API is running"
    }


@router.post("/ask", response_model=AskResponse)
def ask_question(
    request: Request,
    ask_request: AskRequest,
    db: Session = Depends(get_db)
):
    question = ask_request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chat_session = None

    if ask_request.session_id is not None:
        chat_session = get_chat_session(db, ask_request.session_id)

        if chat_session is None:
            raise HTTPException(status_code=404, detail="Chat session not found.")
    else:
        chat_session = create_chat_session(db, title=question[:80])

    recent_messages = get_recent_chat_messages(
        db,
        session_id=chat_session.id,
        limit=settings.api_memory_turns
    )
    conversation_history = [
        {
            "question": message.question,
            "answer": message.answer
        }
        for message in recent_messages
    ]
    effective_conversation_history = select_conversation_history_for_query(
        question,
        conversation_history
    )
    cache_key = build_cache_key(
        question=question,
        doc_type_filter=ask_request.doc_type_filter,
        conversation_history=effective_conversation_history
    )
    cached_response = get_cached_answer(cache_key)

    if cached_response is not None:
        save_chat_message(
            db=db,
            session_id=chat_session.id,
            question=question,
            answer=cached_response["answer"],
            doc_type_filter=ask_request.doc_type_filter,
            sources=cached_response["sources"]
        )

        return {
            "session_id": chat_session.id,
            "answer": cached_response["answer"],
            "sources": cached_response["sources"]
        }

    try:
        result = ask_rag(
            query=question,
            doc_type_filter=ask_request.doc_type_filter,
            conversation_history=effective_conversation_history
        )
    except VectorStoreError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except LLMGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error))
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected chatbot failure.")

    sources = serialize_sources(result["source_documents"])
    response_payload = {
        "answer": result["result"],
        "sources": sources
    }
    set_cached_answer(cache_key, response_payload)

    save_chat_message(
        db=db,
        session_id=chat_session.id,
        question=question,
        answer=result["result"],
        doc_type_filter=ask_request.doc_type_filter,
        sources=sources
    )

    return {
        "session_id": chat_session.id,
        "answer": result["result"],
        "sources": sources
    }


@router.post("/upload", response_model=UploadResponse, status_code=202)
def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    doc_type: Literal["contract", "research_paper", "notes"] = Form(...),
    db: Session = Depends(get_db)
):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Missing upload file.")

    try:
        destination = get_upload_path(file.filename, doc_type)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if destination.stat().st_size == 0:
        destination.unlink()
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document = save_document_record(
        db=db,
        file_name=destination.name,
        doc_type=doc_type,
        source=str(destination),
        chunks_indexed=0,
        status="pending"
    )
    background_tasks.add_task(
        index_uploaded_document,
        document.id,
        str(destination),
        doc_type
    )

    return {
        "id": document.id,
        "file_name": document.file_name,
        "doc_type": document.doc_type,
        "source": document.source,
        "chunks_indexed": document.chunks_indexed,
        "status": document.status,
        "error_message": document.error_message
    }


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    request: Request,
    document_id: int,
    db: Session = Depends(get_db)
):
    document = get_document_record(db, document_id)

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "id": document.id,
        "file_name": document.file_name,
        "doc_type": document.doc_type,
        "source": document.source,
        "chunks_indexed": document.chunks_indexed,
        "status": document.status,
        "error_message": document.error_message
    }


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(
    request: Request,
    session_id: int,
    db: Session = Depends(get_db)
):
    chat_session = get_chat_session(db, session_id)

    if chat_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    return {
        "id": chat_session.id,
        "messages": [
            {
                "id": message.id,
                "question": message.question,
                "answer": message.answer,
                "doc_type_filter": message.doc_type_filter,
                "sources": message.sources
            }
            for message in chat_session.messages
        ]
    }
