from typing import Any, Dict, List


def serialize_sources(source_documents) -> List[Dict[str, Any]]:
    sources = []

    for doc in source_documents:
        metadata = doc.metadata

        sources.append({
            "title": metadata.get("title"),
            "doc_type": metadata.get("doc_type"),
            "file_name": metadata.get("file_name"),
            "chunk_id": metadata.get("chunk_id"),
            "source": metadata.get("source")
        })

    return sources
