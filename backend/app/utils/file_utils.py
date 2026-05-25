from pathlib import Path

from fastapi import UploadFile


def ensure_upload_dir(upload_dir: str) -> Path:
    path = Path(upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


async def save_upload_file(upload_file: UploadFile) -> str:
    upload_path = ensure_upload_dir("app/uploads")
    destination = upload_path / upload_file.filename
    content = await upload_file.read()
    destination.write_bytes(content)
    return str(destination)
