"""Prometheus 메트릭 API 컨트롤러"""
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    #Prometheus용 메트릭 노출 엔드포인트
   
    metrics = generate_latest()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)

