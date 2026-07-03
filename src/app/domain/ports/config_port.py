"""설정 관리 포트 인터페이스"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class ConfigPort(ABC):
    """설정 관리 포트 인터페이스
    
    설정 파일 및 환경 변수 접근을 추상화하는 포트입니다.
    도메인 레이어는 이 인터페이스를 통해 설정에 접근합니다.
    """
    
    @abstractmethod
    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """설정 값 조회"""
        pass
    
    @abstractmethod
    def get_int(self, key: str, default: int = 0, section: Optional[str] = None) -> int:
        """정수 설정 값 조회"""
        pass
    
    @abstractmethod
    def get_float(self, key: str, default: float = 0.0, section: Optional[str] = None) -> float:
        """실수 설정 값 조회"""
        pass
    
    @abstractmethod
    def get_bool(self, key: str, default: bool = False, section: Optional[str] = None) -> bool:
        """불린 설정 값 조회"""
        pass
    
    @abstractmethod
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        pass

