"""Circuit Breaker & Retry 패턴 구현"""
import time
import logging
from enum import Enum
from typing import Callable, TypeVar, Optional, Any
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit Breaker 상태"""
    CLOSED = "closed"  # 정상 동작
    OPEN = "open"  # 차단됨 (실패율 임계값 초과)
    HALF_OPEN = "half_open"  # 테스트 중 (일부 요청 허용)


class CircuitBreaker:
    """Circuit Breaker 패턴 구현
    
    외부 서비스가 일시적으로 실패할 때 불필요한 요청을 차단하여
    시스템 부하를 줄이고 빠르게 실패 응답을 반환합니다.
    
    상태 전이:
    - CLOSED: 정상 동작 (모든 요청 허용)
    - OPEN: 차단됨 (임계값 초과 시, 모든 요청 차단)
    - HALF_OPEN: 테스트 중 (회복 시간 후 일부 요청 허용)
    
    Example:
        ```python
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        
        def api_call():
            return external_api.request()
        
        try:
            result = cb.call(api_call)
        except RuntimeError:
            # Circuit이 OPEN 상태일 때 발생
            pass
        ```
    
    Args:
        failure_threshold: 연속 실패 횟수 임계값 (기본값: 5)
        recovery_timeout: 회복 대기 시간 초 (기본값: 60.0)
        expected_exception: 예상되는 예외 타입 (기본값: Exception)
        name: Circuit Breaker 이름 (로깅용, 기본값: "circuit")
        redis_client: Redis 클라이언트 인스턴스 (Optional)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "circuit",
        redis_client: Optional[Any] = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        self.redis_client = redis_client
        
        # Local fallback state
        self._local_failure_count = 0
        self._local_last_failure_time: Optional[float] = None
        self._local_state = CircuitState.CLOSED

        # Redis keys
        self._key_prefix = f"circuit:{self.name}"
        
        # Lua script for atomic failure increment
        # Keys: [failure_count_key, state_key, last_failure_time_key]
        # Args: [failure_threshold, current_time, ttl]
        self._increment_script = """
        local failure_count = redis.call('INCR', KEYS[1])
        redis.call('SET', KEYS[3], ARGV[2])
        
        -- Set TTL for all keys (to prevent garbage data)
        local ttl = tonumber(ARGV[3])
        redis.call('EXPIRE', KEYS[1], ttl)
        redis.call('EXPIRE', KEYS[2], ttl)
        redis.call('EXPIRE', KEYS[3], ttl)
        
        if failure_count >= tonumber(ARGV[1]) then
            redis.call('SET', KEYS[2], 'open')
            redis.call('EXPIRE', KEYS[2], ttl) -- Refresh TTL for state
            return 1 -- Opened
        end
        return 0 -- Still closed
        """

    @property
    def state(self) -> CircuitState:
        """현재 상태 반환 (Redis 우선, 실패 시 로컬)"""
        if self.redis_client:
            try:
                state_str = self.redis_client.get(f"{self._key_prefix}:state")
                if state_str:
                    # Redis returns bytes, need to decode if not decode_responses=True
                    if isinstance(state_str, bytes):
                        state_str = state_str.decode('utf-8')
                    return CircuitState(state_str)
            except Exception as e:
                logger.warning(f"CircuitBreaker Redis error (get state): {e}. Using local state.")
        return self._local_state

    @state.setter
    def state(self, value: CircuitState):
        """상태 설정 (Redis 우선, 실패 시 로컬)"""
        if self.redis_client:
            try:
                self.redis_client.set(f"{self._key_prefix}:state", value.value)
            except Exception as e:
                logger.warning(f"CircuitBreaker Redis error (set state): {e}. Using local state.")
        self._local_state = value

    @property
    def failure_count(self) -> int:
        """현재 실패 횟수 반환"""
        if self.redis_client:
            try:
                count = self.redis_client.get(f"{self._key_prefix}:failure_count")
                return int(count) if count else 0
            except Exception as e:
                logger.warning(f"CircuitBreaker Redis error (get failure count): {e}. Using local count.")
        return self._local_failure_count

    @failure_count.setter
    def failure_count(self, value: int):
        """실패 횟수 설정"""
        if self.redis_client:
            try:
                self.redis_client.set(f"{self._key_prefix}:failure_count", value)
            except Exception as e:
                logger.warning(f"CircuitBreaker Redis error (set failure count): {e}. Using local count.")
        self._local_failure_count = value

    @property
    def last_failure_time(self) -> Optional[float]:
        """마지막 실패 시간 반환"""
        if self.redis_client:
            try:
                timestamp = self.redis_client.get(f"{self._key_prefix}:last_failure_time")
                return float(timestamp) if timestamp else None
            except Exception as e:
                logger.warning(f"CircuitBreaker Redis error (get last failure): {e}. Using local time.")
        return self._local_last_failure_time

    @last_failure_time.setter
    def last_failure_time(self, value: Optional[float]):
        """마지막 실패 시간 설정"""
        if self.redis_client:
            try:
                if value is None:
                    self.redis_client.delete(f"{self._key_prefix}:last_failure_time")
                else:
                    self.redis_client.set(f"{self._key_prefix}:last_failure_time", value)
            except Exception as e:
                logger.warning(f"CircuitBreaker Redis error (set last failure): {e}. Using local time.")
        self._local_last_failure_time = value
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Circuit Breaker를 통해 함수 호출
        
        Args:
            func: 호출할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자
        
        Returns:
            함수 실행 결과
        
        Raises:
            RuntimeError: Circuit이 OPEN 상태일 때
            Exception: 함수 실행 중 발생한 예외
        """
        # OPEN 상태이고 아직 회복 시간이 지나지 않았으면 즉시 실패
        current_state = self.state
        if current_state == CircuitState.OPEN:
            last_fail = self.last_failure_time
            if last_fail and (time.time() - last_fail) < self.recovery_timeout:
                raise RuntimeError(
                    f"Circuit breaker [{self.name}] is OPEN. "
                    f"Service is unavailable. Retry after {self.recovery_timeout}s"
                )
            # 회복 시간이 지났으면 HALF_OPEN으로 전환
            self.state = CircuitState.HALF_OPEN
            logger.info(f"Circuit breaker [{self.name}] transitioning to HALF_OPEN")
        
        # 함수 실행
        try:
            result = func(*args, **kwargs)
            # 성공 시 상태 리셋
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit breaker [{self.name}] recovered: CLOSED")
            self.reset()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
        except Exception as e:
            # 예상치 못한 예외는 그대로 전파
            raise
    
    def _on_failure(self):
        """실패 처리 (Atomic)"""
        current_time = time.time()
        
        # Local fallback update
        self._local_failure_count += 1
        self._local_last_failure_time = current_time
        if self._local_failure_count >= self.failure_threshold:
            self._local_state = CircuitState.OPEN

        if self.redis_client:
            try:
                # Use Lua script for atomic increment and state check
                keys = [
                    f"{self._key_prefix}:failure_count",
                    f"{self._key_prefix}:state",
                    f"{self._key_prefix}:last_failure_time"
                ]
                # TTL: 1 hour (3600s) - sufficient time for recovery or manual intervention
                ttl = 3600 
                args = [self.failure_threshold, current_time, ttl]
                
                is_opened = self.redis_client.eval(self._increment_script, 3, *keys, *args)
                
                if is_opened:
                    logger.warning(
                        f"Circuit breaker [{self.name}] opened after failures. "
                        f"Will retry after {self.recovery_timeout}s"
                    )
            except Exception as e:
                logger.warning(f"CircuitBreaker Redis error (on failure): {e}. Using local state.")
                # Fallback logging if local state triggered open
                if self._local_state == CircuitState.OPEN and self._local_failure_count == self.failure_threshold:
                     logger.warning(
                        f"Circuit breaker [{self.name}] opened (local fallback) after {self._local_failure_count} failures."
                    )
        elif self._local_state == CircuitState.OPEN and self._local_failure_count == self.failure_threshold:
             logger.warning(
                f"Circuit breaker [{self.name}] opened after {self._local_failure_count} failures. "
                f"Will retry after {self.recovery_timeout}s"
            )
    
    def reset(self):
        """수동 리셋"""
        self._local_failure_count = 0
        self._local_last_failure_time = None
        self._local_state = CircuitState.CLOSED
        
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                pipe.delete(f"{self._key_prefix}:failure_count")
                pipe.delete(f"{self._key_prefix}:last_failure_time")
                pipe.set(f"{self._key_prefix}:state", CircuitState.CLOSED.value)
                pipe.execute()
            except Exception as e:
                logger.warning(f"CircuitBreaker Redis error (reset): {e}. Using local state.")

        logger.info(f"Circuit breaker [{self.name}] manually reset")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """Exponential Backoff를 사용한 Retry 데코레이터
    
    지수 백오프 전략을 사용하여 재시도 간격을 점진적으로 증가시킵니다.
    Jitter를 추가하여 동시 재시도를 방지합니다.
    
    Example:
        ```python
        @retry_with_backoff(
            max_retries=3,
            initial_delay=1.0,
            retry_on=(RateLimitError, APITimeoutError)
        )
        def api_call():
            return external_api.request()
        ```
    
    Args:
        max_retries: 최대 재시도 횟수 (기본값: 3)
        initial_delay: 초기 지연 시간 초 (기본값: 1.0)
        max_delay: 최대 지연 시간 초 (기본값: 60.0)
        exponential_base: 지수 증가 베이스 (기본값: 2.0)
        jitter: 랜덤 지터 추가 여부 (기본값: True)
        retry_on: 재시도할 예외 타입 튜플 (기본값: (Exception,))
        on_retry: 재시도 시 호출할 콜백 함수 (예외, 시도 횟수)
    
    Returns:
        데코레이터 함수
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # 마지막 시도 실패
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise
                    
                    # 지연 시간 계산 (exponential backoff)
                    delay = min(
                        initial_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Jitter 추가 (랜덤 지터로 동시 재시도 방지)
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                except Exception as e:
                    # retry_on에 없는 예외는 즉시 전파
                    raise
            
            # 이론적으로 도달 불가능하지만 타입 체커를 위해
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in retry logic")
        
        return wrapper
    return decorator


def rate_limit_handler(
    max_calls: int = 60,
    period: float = 60.0,
    on_rate_limit: Optional[Callable[[], None]] = None
):
    """Rate Limit 처리 데코레이터 (간단한 토큰 버킷)
    
    지정된 기간 내 최대 호출 횟수를 제한합니다.
    제한을 초과하면 자동으로 대기합니다.
    
    Example:
        ```python
        @rate_limit_handler(max_calls=60, period=60.0)
        def api_call():
            return external_api.request()
        ```
    
    Args:
        max_calls: 기간 내 최대 호출 횟수 (기본값: 60)
        period: 기간 초 (기본값: 60.0)
        on_rate_limit: Rate limit 도달 시 호출할 콜백 함수
    
    Returns:
        데코레이터 함수
    """
    call_times: list[float] = []
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal call_times
            current_time = time.time()
            
            # 오래된 호출 기록 제거
            call_times = [t for t in call_times if current_time - t < period]
            
            # Rate limit 체크
            if len(call_times) >= max_calls:
                if on_rate_limit:
                    on_rate_limit()
                oldest_call = call_times[0]
                wait_time = period - (current_time - oldest_call)
                if wait_time > 0:
                    logger.warning(
                        f"Rate limit reached for {func.__name__}. "
                        f"Waiting {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                    # 대기 후 다시 정리
                    current_time = time.time()
                    call_times = [t for t in call_times if current_time - t < period]
            
            # 호출 기록 추가
            call_times.append(current_time)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

