"""Domain 레이어 상수 정의

비즈니스 로직에서 사용하는 상수들을 중앙에서 관리합니다.
"""

# ==================== RAG 관련 ====================

# 신뢰도 임계값
CONFIDENCE_THRESHOLD_HIGH = 0.65  # 높은 신뢰도 (FAQ 생성 가능)
CONFIDENCE_THRESHOLD_MEDIUM = 0.5  # 중간 신뢰도 (답변 제공)
CONFIDENCE_THRESHOLD_LOW = 0.3    # 낮은 신뢰도 (재질문 권장)

# 검색 설정
DEFAULT_TOP_K = 5  # 기본 검색 문서 수
DEFAULT_SEARCH_RESULTS_COUNT = 3  # 기본 검색 결과 수


# ==================== 챗봇 설정 ====================

# 기본 모델 설정
DEFAULT_CHAT_MODEL_ID = 2
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 1024

# 응답 길이 제한
MAX_ANSWER_LENGTH = 500  # 카카오워크 Blockit 제한


# ==================== FAQ 관련 ====================

# FAQ 생성 임계값
FAQ_MIN_CONFIDENCE = 0.60  # FAQ 생성 최소 신뢰도
FAQ_MIN_FREQUENCY = 10     # FAQ 생성 최소 빈도수
FAQ_CLUSTER_SIMILARITY = 0.7  # 의미 기반 클러스터링 유사도

# FAQ 검색 설정
FAQ_SEARCH_TOP_K = 10  # FAQ 검색 시 최대 결과 수


# ==================== 세션 관리 ====================

# 세션 TTL
SESSION_DEFAULT_TTL = 3600  # 1시간 (초)
SESSION_MAX_HISTORY = 3     # 최근 3턴만 유지

# 질문 로그 TTL
QUESTION_LOG_DEFAULT_TTL = 604800  # 7일 (초)


# ==================== 캐시 관련 ====================

# 캐시 TTL
CACHE_DEFAULT_TTL = 3600  # 1시간 (초)
CACHE_SIMILARITY_THRESHOLD = 0.90  # 캐시 히트 유사도 임계값


# ==================== Guardrail 관련 ====================

# 길이 제한
MAX_QUESTION_LENGTH = 1000  # 최대 질문 길이 (문자)
MIN_QUESTION_LENGTH = 2     # 최소 질문 길이 (문자)


# ==================== Celery 작업 ====================

# 재시도 설정
CELERY_MAX_RETRIES = 3
CELERY_RETRY_DELAY = 60  # 초

# 타임아웃
CELERY_TASK_TIMEOUT = 300  # 5분 (초)
