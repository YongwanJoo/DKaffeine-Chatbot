"""파일 시스템 설정 어댑터"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from app.domain.ports.config_port import ConfigPort

logger = logging.getLogger(__name__)


class FileConfigAdapter(ConfigPort):
    """파일 시스템 설정 어댑터
    
    설정 파일 접근을 제공하는 어댑터입니다.
    """
    
    def __init__(self, base_path: str = "./config"):
        self.base_path = Path(base_path)
    
    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """설정 값 조회 (파일 시스템에서는 지원하지 않음)"""
        return default
    
    def get_int(self, key: str, default: int = 0, section: Optional[str] = None) -> int:
        """정수 설정 값 조회 (파일 시스템에서는 지원하지 않음)"""
        return default
    
    def get_float(self, key: str, default: float = 0.0, section: Optional[str] = None) -> float:
        """실수 설정 값 조회 (파일 시스템에서는 지원하지 않음)"""
        return default
    
    def get_bool(self, key: str, default: bool = False, section: Optional[str] = None) -> bool:
        """불린 설정 값 조회 (파일 시스템에서는 지원하지 않음)"""
        return default
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        config_path = Path(file_path)
        if not config_path.is_absolute():
            # 상대 경로인 경우 base_path 기준으로 찾기
            config_path = self.base_path / file_path.lstrip("./")
        
        if not config_path.exists():
            # 프로젝트 루트에서 찾기
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_path = project_root / file_path.lstrip("./")
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}")
        
        return {}

