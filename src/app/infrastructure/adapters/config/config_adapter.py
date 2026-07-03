"""설정 관리 어댑터"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from app.domain.ports.config_port import ConfigPort
from app.infrastructure.config.config_loader import (
    get_config,
    get_config_int,
    get_config_float,
    get_config_bool
)

logger = logging.getLogger(__name__)


class ConfigAdapter(ConfigPort):
    """설정 관리 어댑터 구현
    
    ConfigPort 인터페이스를 구현하여 설정 파일 및 환경 변수 접근을 제공합니다.
    """
    
    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """설정 값 조회"""
        return get_config(key, default, section)
    
    def get_int(self, key: str, default: int = 0, section: Optional[str] = None) -> int:
        """정수 설정 값 조회"""
        return get_config_int(key, default, section)
    
    def get_float(self, key: str, default: float = 0.0, section: Optional[str] = None) -> float:
        """실수 설정 값 조회"""
        return get_config_float(key, default, section)
    
    def get_bool(self, key: str, default: bool = False, section: Optional[str] = None) -> bool:
        """불린 설정 값 조회"""
        return get_config_bool(key, default, section)
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        config_path = Path(file_path)
        if not config_path.exists():
            # 프로젝트 루트에서 찾기
            config_path = Path(__file__).parent.parent.parent.parent.parent / file_path.lstrip("./")
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}")
        
        return {}

