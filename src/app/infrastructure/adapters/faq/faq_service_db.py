"""FAQ 서비스 (DB 전용) - Cosine Similarity 기반"""
from typing import Tuple, Optional
import logging
import numpy as np

from app.domain.ports.faq_port import FAQPort
from app.infrastructure.config.config_loader import get_config_int
from app.infrastructure.adapters.cache import HybridCache
from app.infrastructure.utils.utils_math import cosine_similarity

logger = logging.getLogger(__name__)


def normalize_question_for_search(question: str) -> str:
    """FAQ 검색을 위한 질문 정규화
    
    사용자 질문에서 요청 표현("알려줘", "알려주세요" 등)을 제거하여
    FAQ 질문과의 의미적 유사도를 더 정확하게 계산합니다.
    
    Args:
        question: 원본 질문
        
    Returns:
        정규화된 질문
    """
    import re
    
    request_patterns = [
        r'알려\s*줘\s*',
        r'알려\s*주세요\s*',
        r'알려\s*주시겠어요\s*',
        r'알려\s*주시겠습니까\s*',
        r'알려\s*주실\s*수\s*있나요\s*',
        r'알려\s*주실\s*수\s*있어요\s*',
        r'알려\s*주실\s*수\s*있습니까\s*',
        r'어떻게\s*해\s*',
        r'어떻게\s*해야\s*',
        r'어떻게\s*해야\s*하나요\s*',
        r'어떻게\s*하나요\s*',
        r'어떻게\s*하나\s*',
        r'어떻게\s*하는\s*',
        r'어떻게\s*하는지\s*',
        r'어떻게\s*하는지\s*알려\s*',
        r'\s*야\s*하나요\s*$',
        r'방법\s*알려\s*',
        r'방법\s*알려줘\s*',
        r'방법\s*알려주세요\s*',
        r'설명\s*해\s*줘\s*',
        r'설명\s*해\s*주세요\s*',
        r'설명\s*해\s*주시겠어요\s*',
        r'가르쳐\s*줘\s*',
        r'가르쳐\s*주세요\s*',
        r'말해\s*줘\s*',
        r'말해\s*주세요\s*',
        r'말씀\s*해\s*주세요\s*',
        r'\s*하나요\s*$',
        r'\s*인가요\s*$',
        r'\s*인지\s*$',
        r'\s*인지요\s*$',
        r'\s*야\s*하나요\s*$',
        r'\s*무엇인가요\s*$',  # "무엇인가요?" 제거
        r'\s*무엇인가\s*$',
        r'\s*무엇인지\s*$',
        r'\s*무엇인지요\s*$',
        # 추가: "뭐야", "뭐예요" 등 질문 표현 정규화
        r'\s*뭐야\s*$',
        r'\s*뭐예요\s*$',
        r'\s*뭐야요\s*$',
        r'\s*뭐니\s*$',
        r'\s*뭐냐\s*$',
        r'\s*뭔가\s*$',
        r'\s*뭔가요\s*$',
        r'\s*뭔지\s*$',
        r'\s*뭔지요\s*$',
        # 추가: "이란", "이란?" 등 질문 표현 정규화
        r'\s*이란\s*$',
        r'\s*이란\s*\?',
        r'\s*이란\s*요\s*$',
        r'\s*이란\s*가\s*$',
    ]
    
    normalized = question.strip()
    
    for pattern in request_patterns:
        normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
    
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


class FAQServiceDB(FAQPort):
    """FAQ 서비스 (DB 전용) - Cosine Similarity 기반
    
    FAQ 검색 서비스로, Cosine Similarity를 사용하여 의미 기반 검색을 수행합니다.
    임베딩 기반 검색이 실패할 경우 키워드 매칭으로 자동 폴백합니다.
    
    특징:
    - Cosine Similarity 기반 의미 검색
    - DB에서 실시간 FAQ 조회
    - LRU + TTL 하이브리드 캐시 (임베딩 캐싱)
    - 키워드 매칭 폴백 지원
    
    Example:
        ```python
        service = FAQServiceDB()
        match, answer, score = service.search(
            user_message="휴가 신청 방법",
            threshold=0.6
        )
        ```
    """
    
    def __init__(self):
        try:
            from app.infrastructure.adapters.llm import LLMProvider
            self.embeddings = LLMProvider().get_embeddings()
            max_cache_size = get_config_int("FAQ_EMBEDDING_CACHE_SIZE", 1000, section="faq")
            cache_ttl = get_config_int("FAQ_EMBEDDING_CACHE_TTL", 3600, section="faq")
            self._faq_embeddings_cache = HybridCache(
                max_size=max_cache_size,
                ttl_seconds=cache_ttl
            )
            logger.info(f"FAQ 임베딩 초기화 완료: model=amazon.titan-embed-text-v2:0, cache_size={max_cache_size}, ttl={cache_ttl}s")
        except Exception as e:
            logger.warning(f"임베딩 모델 초기화 실패: {e}. 키워드 매칭으로 폴백합니다.")
            self.embeddings = None
            self._faq_embeddings_cache = None
    
    def search(
        self, 
        user_message: str, 
        threshold: float = 0.6,
        user_embedding: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """FAQ 검색 (DB 실시간 조회)
        
        사용자 메시지와 DB의 활성 FAQ를 Cosine Similarity로 비교하여
        가장 유사한 FAQ를 찾습니다.
        
        Args:
            user_message: 사용자 질문
            threshold: 유사도 임계값 (0.0 ~ 1.0, 기본값: 0.6)
            user_embedding: 사용자 질문 임베딩 (제공되면 재사용, None이면 생성)
        
        Returns:
            (faq_match, faq_answer, faq_confidence) 튜플
        """
        return self._search_db(user_message, threshold, user_embedding)

    def _search_db(self, user_message: str, threshold: float, user_embedding: Optional[np.ndarray] = None) -> Tuple[bool, Optional[str], Optional[float]]:
        """DB에서 FAQ 검색 (Cosine Similarity 기반)"""
        logger.info(f"[FAQ DB 검색] query='{user_message}'")
        
        try:
            from app.infrastructure.persistence.session import db_session
            from app.infrastructure.persistence.repositories.faq_repository import FaqRepository
            
            # 임베딩 모델이 없으면 키워드 검색으로 폴백 (DB 기반)
            if not self.embeddings or self._faq_embeddings_cache is None:
                return self._search_db_keyword_fallback(user_message, threshold)

            normalized_user_message = normalize_question_for_search(user_message)
            
            if user_embedding is None:
                user_embedding = np.array(self.embeddings.embed_query(normalized_user_message))
            else:
                user_embedding = np.array(user_embedding)

            with db_session() as db:
                repo = FaqRepository(db)
                # 활성 FAQ 모두 조회
                active_faqs = repo.find_all_active(limit=1000)
                
                if not active_faqs:
                    logger.info("[FAQ DB 검색] 활성 FAQ가 없습니다.")
                    return False, None, None
                
                best_match = None
                best_score = 0.0
                
                for faq in active_faqs:
                    # FAQ 질문도 정규화하여 임베딩 생성 (사용자 질문과 동일한 정규화 적용)
                    normalized_faq_question = normalize_question_for_search(faq.question)
                    
                    # 캐시 키 생성 (ID와 정규화된 질문 기준)
                    cache_key = f"faq_embedding:db:{faq.id}:{normalized_faq_question}"
                    
                    cached_embedding = self._faq_embeddings_cache.get(cache_key)
                    if cached_embedding is None:
                        try:
                            # 정규화된 FAQ 질문으로 임베딩 생성 및 캐시
                            faq_embedding = np.array(self.embeddings.embed_query(normalized_faq_question))
                            self._faq_embeddings_cache.set(cache_key, faq_embedding)
                        except Exception as e:
                            logger.warning(f"FAQ 임베딩 생성 실패 (ID={faq.id}): {e}")
                            continue
                    else:
                        faq_embedding = cached_embedding
                    
                    # 유사도 계산
                    score = cosine_similarity(user_embedding, faq_embedding)
                    
                    if score > best_score:
                        best_score = score
                        best_match = faq
                
                if best_score >= threshold and best_match:
                    logger.info(f"[FAQ DB 매칭 성공] 질문='{best_match.question}', 점수={best_score:.3f}, 임계값={threshold:.2f}")
                    return True, best_match.answer, best_score
                
                logger.info(f"[FAQ DB 매칭 실패] 최고점수={best_score:.3f}, 임계값={threshold:.2f}. 키워드 검색으로 폴백합니다.")
                return self._search_db_keyword_fallback(user_message, threshold)

        except Exception as e:
            logger.error(f"FAQ DB 검색 오류: {e}", exc_info=True)
            return False, None, None

    def _search_db_keyword_fallback(self, user_message: str, threshold: float) -> Tuple[bool, Optional[str], Optional[float]]:
        """DB에서 FAQ 검색 (키워드 매칭 폴백)"""
        try:
            from app.infrastructure.persistence.session import db_session
            from app.infrastructure.persistence.repositories.faq_repository import FaqRepository
            
            normalized_user_message = normalize_question_for_search(user_message)
            
            # 조사 제거 및 키워드 추출
            import re
            particles = ['의', '에', '에서', '에게', '을', '를', '이', '가', '은', '는', '와', '과', '도', '로', '으로', 
                        '에 대해', '에 관한', '에 대한', '으로서', '로서', '부터', '까지', '만', '조차', '마저']
            
            # 조사 제거
            cleaned_message = normalized_user_message.lower()
            for particle in particles:
                # 단어 끝에 붙은 조사 제거 (예: "카카오워크의" -> "카카오워크")
                cleaned_message = re.sub(rf'(\w+){re.escape(particle)}\b', r'\1', cleaned_message)
                # 단독 조사 제거
                cleaned_message = cleaned_message.replace(f' {particle} ', ' ')
                cleaned_message = cleaned_message.replace(f' {particle}', '')
                cleaned_message = cleaned_message.replace(f'{particle} ', '')
            
            # 공백 정리 및 키워드 추출
            cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()
            all_keywords = [w.strip() for w in cleaned_message.split() if len(w.strip()) > 1]
            
            if not all_keywords:
                return False, None, None
            
            # 공통 키워드 필터링 (가중치 낮춤)
            # 일반적인 형용사, 조사, 질문 형식 등은 핵심 키워드가 아님
            common_keywords = {
                '주요', '중요', '대한', '관련', '위한', '있는', '있는지', '없는', '없는지',
                '무엇', '무엇인가', '무엇인가요', '무엇인지', '어떤', '어떻게', '어떤지',
                '기능', '장점', '단점', '특징', '방법', '절차', '과정', '순서',
                '설명', '알려', '알려줘', '알려주세요', '말해', '말해줘', '말해주세요',
                '서비스', '시스템', '플랫폼'  # 너무 일반적인 단어도 공통 키워드로
            }
            
            # 핵심 키워드와 공통 키워드 분리
            core_keywords = [kw for kw in all_keywords if kw.lower() not in common_keywords]
            common_keywords_found = [kw for kw in all_keywords if kw.lower() in common_keywords]
            
            # 핵심 키워드가 없으면 매칭 실패 (너무 일반적인 질문)
            if not core_keywords:
                logger.info(f"[FAQ DB 키워드 매칭 실패] 핵심 키워드가 없음. 모든 키워드가 공통 키워드: {all_keywords}")
                return False, None, None
            
            # 검색에는 모든 키워드 사용 (후보 확보)
            search_keywords = all_keywords
                
            with db_session() as db:
                repo = FaqRepository(db)
                # 키워드로 검색
                candidates = repo.find_by_keywords(search_keywords, limit=10)
                
                if not candidates:
                    return False, None, None
                
                # 개선된 점수 계산 (핵심 키워드에 더 높은 가중치)
                best_match = None
                best_score = 0.0
                
                for faq in candidates:
                    faq_question_lower = faq.question.lower()
                    faq_answer_lower = faq.answer.lower()
                    target_text = f"{faq_question_lower} {faq_answer_lower}"
                    
                    # 핵심 키워드 매칭 (높은 가중치)
                    core_question_matches = sum(1 for kw in core_keywords if kw.lower() in faq_question_lower)
                    core_answer_matches = sum(1 for kw in core_keywords if kw.lower() in faq_answer_lower)
                    core_total_matches = sum(1 for kw in core_keywords if kw.lower() in target_text)
                    
                    # 공통 키워드 매칭 (낮은 가중치)
                    common_question_matches = sum(1 for kw in common_keywords_found if kw.lower() in faq_question_lower)
                    common_answer_matches = sum(1 for kw in common_keywords_found if kw.lower() in faq_answer_lower)
                    
                    # 핵심 키워드 점수 (높은 가중치)
                    core_question_score = (core_question_matches / len(core_keywords)) * 2.0 if core_keywords else 0.0
                    core_answer_score = (core_answer_matches / len(core_keywords)) * 1.5 if core_keywords else 0.0
                    core_total_score = (core_total_matches / len(core_keywords)) * 1.0 if core_keywords else 0.0
                    
                    # 공통 키워드 점수 (낮은 가중치)
                    common_question_score = (common_question_matches / len(common_keywords_found)) * 0.3 if common_keywords_found else 0.0
                    common_answer_score = (common_answer_matches / len(common_keywords_found)) * 0.2 if common_keywords_found else 0.0
                    
                    # 최종 점수: 핵심 키워드 매칭에 더 높은 가중치
                    score = max(
                        core_question_score + common_question_score,  # 질문 매칭
                        core_answer_score + common_answer_score,      # 답변 매칭
                        core_total_score                              # 전체 매칭
                    )
                    
                    # 핵심 키워드가 질문에 모두 매칭되어야 높은 점수 부여
                    if core_question_matches == len(core_keywords) and len(core_keywords) >= 1:
                        score = max(score, 0.75)  # 모든 핵심 키워드 매칭 시 높은 점수
                    elif core_question_matches >= len(core_keywords) * 0.5 and len(core_keywords) >= 2:
                        score = max(score, 0.6)   # 핵심 키워드 절반 이상 매칭 시
                    
                    # 핵심 키워드가 하나도 매칭되지 않으면 매칭 실패 (필수 조건)
                    if core_question_matches == 0 and core_total_matches == 0:
                        # 핵심 키워드가 전혀 매칭되지 않으면 이 FAQ는 제외
                        score = 0.0
                        logger.debug(
                            f"[FAQ 키워드 매칭] 핵심 키워드 미매칭으로 제외: "
                            f"FAQ='{faq.question[:30]}...', 핵심키워드={core_keywords}"
                        )
                        continue  # 다음 FAQ로 넘어감
                    
                    if score > best_score:
                        best_score = score
                        best_match = faq
                
                # 키워드 매칭 임계값 (핵심 키워드 기반으로 더 엄격하게)
                keyword_threshold = 0.75  # 일반적으로 사용되는 FAQ 매칭 임계값
                
                if best_score >= keyword_threshold and best_match:
                    logger.info(
                        f"[FAQ DB 키워드 매칭 성공] 질문='{best_match.question}', "
                        f"점수={best_score:.2f}, 임계값={keyword_threshold:.2f}, "
                        f"핵심키워드={core_keywords}, 공통키워드={common_keywords_found}"
                    )
                    return True, best_match.answer, best_score
                
                logger.info(
                    f"[FAQ DB 키워드 매칭 실패] 최고점수={best_score:.2f}, 임계값={keyword_threshold:.2f}, "
                    f"핵심키워드={core_keywords}, 공통키워드={common_keywords_found}"
                )
                return False, None, None
        except Exception as e:
            logger.error(f"FAQ DB 키워드 검색 오류: {e}", exc_info=True)
            return False, None, None

