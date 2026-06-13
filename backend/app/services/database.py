from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from backend.app.services.config import settings


engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(500), nullable=False)
    chunks_indexed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    messages: Mapped[List["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type_filter: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sources: Mapped[List[dict]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


def init_db():
    Base.metadata.create_all(bind=engine)
    ensure_document_status_columns()


def ensure_document_status_columns():
    inspector = inspect(engine)

    if not inspector.has_table("documents"):
        return

    column_names = {
        column["name"]
        for column in inspector.get_columns("documents")
    }

    with engine.begin() as connection:
        if "status" not in column_names:
            connection.execute(
                text("ALTER TABLE documents ADD COLUMN status VARCHAR(50) NOT NULL DEFAULT 'pending'")
            )

        if "error_message" not in column_names:
            connection.execute(
                text("ALTER TABLE documents ADD COLUMN error_message TEXT")
            )


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def create_chat_session(db, title=None):
    chat_session = ChatSession(title=title)
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


def get_chat_session(db, session_id):
    return db.get(ChatSession, session_id)


def get_document_record(db, document_id):
    return db.get(DocumentRecord, document_id)


def get_recent_chat_messages(db, session_id, limit):
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(limit)
        .all()
    )[::-1]


def save_chat_message(db, session_id, question, answer, doc_type_filter, sources):
    message = ChatMessage(
        session_id=session_id,
        question=question,
        answer=answer,
        doc_type_filter=doc_type_filter,
        sources=sources
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def save_document_record(
    db,
    file_name,
    doc_type,
    source,
    chunks_indexed=0,
    status="pending",
    error_message=None
):
    document = DocumentRecord(
        file_name=file_name,
        doc_type=doc_type,
        source=source,
        chunks_indexed=chunks_indexed,
        status=status,
        error_message=error_message
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def update_document_record(
    db,
    document_id,
    status,
    chunks_indexed=None,
    error_message=None
):
    document = get_document_record(db, document_id)

    if document is None:
        return None

    document.status = status

    if chunks_indexed is not None:
        document.chunks_indexed = chunks_indexed

    document.error_message = error_message

    db.commit()
    db.refresh(document)
    return document
