from app.rag.pipeline import DocumentLoadError, VectorStoreError, add_uploaded_document
from app.services.database import SessionLocal, update_document_record


def index_uploaded_document(document_id: int, file_path: str, doc_type: str):
    background_db = SessionLocal()

    try:
        update_document_record(
            background_db,
            document_id=document_id,
            status="processing"
        )
        upload_result = add_uploaded_document(file_path, doc_type)
        update_document_record(
            background_db,
            document_id=document_id,
            status="completed",
            chunks_indexed=upload_result["chunks_indexed"],
            error_message=None
        )
    except (DocumentLoadError, VectorStoreError) as error:
        update_document_record(
            background_db,
            document_id=document_id,
            status="failed",
            error_message=str(error)
        )
    except Exception as error:
        update_document_record(
            background_db,
            document_id=document_id,
            status="failed",
            error_message=f"Unexpected indexing failure: {error}"
        )
    finally:
        background_db.close()
