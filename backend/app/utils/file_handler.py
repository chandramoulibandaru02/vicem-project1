from pathlib import Path

from fastapi import UploadFile


def ensure_upload_dir(upload_dir: str) -> Path:
    path = Path(upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


async def save_upload_file(upload_file: UploadFile, upload_dir: str) -> str:
    if not upload_file.filename:
        raise ValueError("Filename is required")

    content = await upload_file.read()
    if len(content) == 0:
        raise ValueError("File is empty")

    safe_filename = Path(upload_file.filename).name
    upload_path = ensure_upload_dir(upload_dir)
    destination = upload_path / safe_filename
    destination.write_bytes(content)

    return str(destination)
