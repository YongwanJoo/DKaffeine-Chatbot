"""세션 관리 서비스 (Redis 기반)"""
import logging
import json
from typing import List, Optional, Dict, Any

from app.domain.ports.config_port import ConfigPort

logger = logging.getLogger(__name__)


class SessionService:
    """세션 관리 서비스 (Redis 기반, 최근 3턴만 유지, TTL로 자동 삭제)
    
    대화 히스토리를 관리하는 서비스입니다.
    최근 3턴(6개 메시지)만 유지하여 컨텍스트를 제공하면서도
    메모리 사용량을 제한합니다.
    
    특징:
    - Redis 기반 세션 저장
    - Connection Pool 사용
    - 최근 3턴만 유지 (메모리 최적화)
    - TTL 기반 자동 삭제
    
    Example:
        ```python
        service = SessionService()
        
        # 히스토리 추가
        service.add_message("session_001", "user", "질문")
        service.add_message("session_001", "assistant", "답변")
        
        # 히스토리 조회
        history = service.get_history("session_001")
        ```
    
    클래스 변수:
        MAX_TURNS: 최대 유지할 턴 수 (기본값: 3)
        MAX_MESSAGES: 최대 메시지 수 (기본값: 6)
        DEFAULT_TTL: 기본 TTL 초 (기본값: 3600, 1시간) - 환경변수 SESSION_TTL로 설정 가능
    """
    
    MAX_TURNS = 3  # 최대 유지할 턴 수
    MAX_MESSAGES = MAX_TURNS * 2  # 3턴 = 6개 메시지 (user-assistant 쌍)
    
    _redis_client = None
    _redis_pool = None
    _config_port: Optional[ConfigPort] = None
    
    @classmethod
    def set_config_port(cls, config_port: ConfigPort):
        """설정 포트 설정 (의존성 주입)"""
        cls._config_port = config_port
    
    @classmethod
    def _get_config_port(cls) -> ConfigPort:
        """설정 포트 가져오기 (없으면 기본 어댑터 사용)"""
        if cls._config_port is None:
            from app.infrastructure.adapters.config.config_adapter import ConfigAdapter
            cls._config_port = ConfigAdapter()
        return cls._config_port
    
    @classmethod
    def _get_default_ttl(cls) -> int:
        """기본 TTL 가져오기"""
        config_port = cls._get_config_port()
        return config_port.get_int("ttl", 3600, section="session")
    
    @classmethod
    def _get_redis_client(cls):
        """Redis 클라이언트 싱글톤 (redis_client.py의 싱글톤 재사용)
        
        ConnectionPool 중복 생성 방지, redis_client.py의 싱글톤 재사용
        """
        if cls._redis_client is None:
            try:
                # redis_client.py의 싱글톤 재사용 (ConnectionPool 중복 생성 방지)
                from app.infrastructure.utils.redis_client import get_redis_client
                cls._redis_client = get_redis_client()
                
                if cls._redis_client is None:
                    raise ValueError("Redis가 비활성화되어 있습니다 (use_redis=False)")
                
                # 연결 테스트
                cls._redis_client.ping()
                logger.info("✅ SessionService: Redis client initialized (redis_client.py 싱글톤 재사용)")
            except ImportError:
                logger.error("⚠️ SessionService: Redis 패키지가 설치되지 않았습니다.")
                raise
            except Exception as e:
                logger.error(f"⚠️ SessionService: Redis 연결 실패: {e}")
                raise
        return cls._redis_client
    
    @staticmethod
    def _make_history_key(session_id: str) -> str:
        """히스토리 Redis 키 생성"""
        return f"session:history:{session_id}"
    
    @staticmethod
    def _make_session_key(session_id: str) -> str:
        """세션 메타데이터 Redis 키 생성"""
        return f"session:meta:{session_id}"
    
    @classmethod
    def get_history(cls, session_id: str) -> List[Dict[str, str]]:
        """세션의 히스토리 로드 (최근 3턴, 동기)
        
        Redis List를 사용하여 Atomic하게 조회
        """
        try:
            redis_client = cls._get_redis_client()
            key = cls._make_history_key(session_id)
            
            # Redis List에서 직접 조회 (JSON 문자열 리스트)
            # LRANGE로 최근 MAX_MESSAGES개만 가져오기
            history_json_list = redis_client.lrange(key, -cls.MAX_MESSAGES, -1)
            
            if history_json_list:
                # JSON 문자열 리스트를 파싱
                history = [json.loads(msg_json) for msg_json in history_json_list]
                logger.info(f"✅ 히스토리 로드 성공: session_id={session_id}, 메시지 수={len(history)}")
                return history
            else:
                logger.info(f"⚠️ 히스토리 없음: session_id={session_id} (Redis에 해당 키가 없음)")
                return []
        except Exception as e:
            logger.warning(f"히스토리 로드 실패: {e}, 빈 히스토리 반환")
            return []
    
    @classmethod
    async def get_history_async(cls, session_id: str) -> List[Dict[str, str]]:
        """세션의 히스토리 로드 (최근 3턴, 비동기)"""
        import asyncio
        
        # 동기 메서드를 비동기로 실행
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cls.get_history, session_id)
    
    @classmethod
    def save_message(
        cls,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        ttl: Optional[int] = None,
        intent_type: Optional[str] = None
    ) -> None:
        """메시지 저장 및 오래된 메시지 정리 (최근 3턴만 유지)
        
        Redis RPUSH + LTRIM을 사용하여 Atomic하게 저장 (Race Condition 방지)
        """
        try:
            redis_client = cls._get_redis_client()
            history_key = cls._make_history_key(session_id)
            session_key = cls._make_session_key(session_id)
            
            # 새 메시지 생성 (intent_type 포함)
            message = {
                "role": role,
                "content": content
            }
            if intent_type:
                message["intent_type"] = intent_type
            message_json = json.dumps(message, ensure_ascii=False)
            
            # Redis Pipeline을 사용하여 Atomic 연산 보장
            pipe = redis_client.pipeline()
            
            # RPUSH: 리스트 끝에 메시지 추가 (Atomic)
            pipe.rpush(history_key, message_json)
            
            # LTRIM: 최근 MAX_MESSAGES개만 유지 (Atomic)
            # 음수 인덱스: -MAX_MESSAGES부터 끝까지 유지
            pipe.ltrim(history_key, -cls.MAX_MESSAGES, -1)
            
            # TTL 설정
            final_ttl = ttl if ttl is not None else cls._get_default_ttl()
            pipe.expire(history_key, final_ttl)
            
            # 세션 메타데이터 저장 (user_id)
            import time
            session_meta = {
                "user_id": user_id,
                "last_updated": time.time()
            }
            pipe.setex(session_key, final_ttl, json.dumps(session_meta, ensure_ascii=False))
            
            # Pipeline 실행 (모든 명령이 Atomic하게 실행됨)
            pipe.execute()
            
            logger.debug(f"메시지 저장 완료: session_id={session_id}, role={role}, TTL={final_ttl}초")
        except Exception as e:
            logger.error(f"메시지 저장 실패: {e}", exc_info=True)
            # 저장 실패해도 응답은 정상 반환 (세션 관리는 선택적 기능)
    
    @classmethod
    def save_conversation_turn(
        cls,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: Optional[str] = None,
        ttl: Optional[int] = None,
        intent_type: Optional[str] = None
    ) -> None:
        """대화 턴 저장 (사용자 메시지 + 어시스턴트 답변)"""
        # 사용자 메시지 저장 (intent_type 포함)
        cls.save_message(
            session_id=session_id,
            user_id=user_id,
            role="user",
            content=user_message,
            ttl=ttl,
            intent_type=intent_type
        )
        
        # 어시스턴트 답변이 있으면 저장 (assistant는 intent_type 없음, user의 intent_type 사용)
        if assistant_message:
            cls.save_message(
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                content=assistant_message,
                ttl=ttl,
                intent_type=None  # assistant 메시지는 intent_type 없음
            )
    
    @classmethod
    async def save_conversation_turn_async(
        cls,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_message: Optional[str] = None,
        ttl: Optional[int] = None,
        intent_type: Optional[str] = None
    ) -> None:
        """대화 턴 저장 (비동기)"""
        import asyncio
        
        # 동기 메서드를 비동기로 실행
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            cls.save_conversation_turn,
            session_id,
            user_id,
            user_message,
            assistant_message,
            ttl,
            intent_type
        )
    
    @classmethod
    def save_history(cls, session_id: str, history: List[Dict[str, Any]], ttl: Optional[int] = None) -> None:
        """세션 히스토리 저장 (동기) - 하위 호환성
        
        Redis Pipeline을 사용하여 Atomic하게 저장
        """
        try:
            redis_client = cls._get_redis_client()
            history_key = cls._make_history_key(session_id)
            
            # 최근 3턴만 유지
            if len(history) > cls.MAX_MESSAGES:
                history = history[-cls.MAX_MESSAGES:]
            
            # Redis Pipeline을 사용하여 Atomic 연산
            pipe = redis_client.pipeline()
            
            # 기존 리스트 삭제 후 새로 추가
            pipe.delete(history_key)
            
            # 모든 메시지를 JSON 문자열로 변환하여 RPUSH
            for msg in history:
                msg_json = json.dumps(msg, ensure_ascii=False)
                pipe.rpush(history_key, msg_json)
            
            # TTL 설정
            final_ttl = ttl if ttl is not None else cls._get_default_ttl()
            pipe.expire(history_key, final_ttl)
            
            # Pipeline 실행
            pipe.execute()
            
            logger.debug(f"히스토리 저장 완료: session_id={session_id}, 메시지 수={len(history)}, TTL={final_ttl}초")
        except Exception as e:
            logger.warning(f"히스토리 저장 실패: {e}")
    
    @classmethod
    async def save_history_async(cls, session_id: str, history: List[Dict[str, Any]], ttl: Optional[int] = None) -> None:
        """세션 히스토리 저장 (비동기)"""
        import asyncio
        
        # 동기 메서드를 비동기로 실행
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, cls.save_history, session_id, history, ttl)
    
    @classmethod
    def delete_session(cls, session_id: str) -> None:
        """세션 삭제 (히스토리 + 메타데이터)"""
        try:
            redis_client = cls._get_redis_client()
            history_key = cls._make_history_key(session_id)
            session_key = cls._make_session_key(session_id)
            
            redis_client.delete(history_key)
            redis_client.delete(session_key)
            logger.info(f"세션 삭제: session_id={session_id}")
        except Exception as e:
            logger.warning(f"세션 삭제 실패: {e}")
    
    @classmethod
    def get_session_meta(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 메타데이터 조회"""
        try:
            redis_client = cls._get_redis_client()
            session_key = cls._make_session_key(session_id)
            
            meta_json = redis_client.get(session_key)
            if meta_json:
                return json.loads(meta_json)
            return None
        except Exception as e:
            logger.warning(f"세션 메타데이터 조회 실패: {e}")
            return None

