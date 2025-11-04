from fastapi import APIRouter

router = APIRouter(
    prefix="/recognize",
    tags=["recognition"]
)