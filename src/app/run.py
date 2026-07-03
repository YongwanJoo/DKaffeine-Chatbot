"""애플리케이션 진입점"""
from __future__ import annotations

import logging
from app.infrastructure.config.config_loader import get_config_bool, get_config

# LangSmith 추적 설정
logger = logging.getLogger(__name__)
if get_config_bool("tracing_v2", False, "langsmith"):
    api_key = get_config("api_key", None, "langsmith")
    project = get_config("project", "chatbot-production", "langsmith")
    
    if api_key:
        logger.info(f"✅ LangSmith 추적 활성화: project={project}")
        logger.info("   대시보드: https://smith.langchain.com")
    else:
        logger.warning("⚠️  LangSmith tracing_v2=true이지만 api_key가 설정되지 않았습니다.")
        logger.warning("   .secrets.toml의 [langsmith] 섹션에 api_key를 추가하세요.")

import uvicorn
from app.setup.app_factory import create_app

app = create_app()

if __name__ == "__main__":
    # Security Fix: reload 옵션을 환경변수로 제어 (프로덕션에서는 False)
    # 환경변수 RELOAD가 "true", "1", "yes"일 때만 reload 활성화
    reload_enabled = get_config_bool("RELOAD", False, section="app")
    
    uvicorn.run("app.run:app", host="0.0.0.0", port=8000, reload=reload_enabled)
