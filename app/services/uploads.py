from pathlib import Path

from app.services.config import settings


def get_upload_path(file_name: str, doc_type: str) -> Path:
    safe_name = Path(file_name).name

    if not safe_name:
        raise ValueError("Uploaded file must have a file name.")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in settings.allowed_upload_extensions:
        raise ValueError("Only .txt and .pdf uploads are supported.")

    upload_dir = settings.docs_dir / settings.doc_type_folders[doc_type]
    upload_dir.mkdir(parents=True, exist_ok=True)

    destination = upload_dir / safe_name
    stem = destination.stem
    counter = 1

    while destination.exists():
        destination = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    return destination
