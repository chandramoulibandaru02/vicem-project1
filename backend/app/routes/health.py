from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", summary="Health check")
async def health_check():
    settings = get_settings()

    return {
        "status": "ok",
        "service": "ecm-ai-backend",
        "environment": settings.app_env,
        "model": settings.model_name,
    }
