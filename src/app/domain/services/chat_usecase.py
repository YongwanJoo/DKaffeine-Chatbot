"""채팅 UseCase (비즈니스 로직 캡슐화)

컨트롤러에서 비즈니스 로직을 분리하여 재사용 가능한 UseCase로 캡슐화
웹소켓 채팅이나 Slack 봇 연동 시에도 동일한 로직을 재사용 가능
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

from app.domain.constants import DEFAULT_CHAT_MODEL_ID, FAQ_MIN_CONFIDENCE
from app.domain.services.state_service import create_initial_state, format_response
from app.domain.services.chat_history_validator import ChatHistoryValidator
from app.domain.services.session_service import SessionService
from app.domain.services.chatbot_config_sync_service import ChatbotConfigSyncService
from app.domain.ports import GuardrailPort
from app.orchestration import create_chatbot_graph
from app.infrastructure.jobs.tasks import save_chat_log_task, generate_faq_candidate_task
from app.infrastructure.persistence.repositories.category_repository import CategoryRepository
from app.infrastructure.persistence.session import async_db_session

logger = logging.getLogger(__name__)


class ChatUsecase:
    """채팅 UseCase
    
    세션 관리, 히스토리 검증, 로그 저장 등 채팅 관련 비즈니스 로직을 캡슐화
    """
    
    def __init__(
        self,
        guardrail_service: GuardrailPort,
        services_config: Dict[str, Any]
    ):
        """ChatUsecase 초기화
        
        Args:
            guardrail_service: Guardrail 서비스
            services_config: LangGraph config (서비스 의존성)
        """
        self.guardrail_service = guardrail_service
        self.services_config = services_config
        
        # LangGraph 그래프는 UseCase 인스턴스마다 생성 (또는 싱글톤으로 관리)
        self.graph = create_chatbot_graph()
    
    async def process_message(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str] = None,
        chat_model_id: Optional[int] = None,
        sync_log: bool = False,
        is_requery: bool = False,
        response_format: str = "plain"
    ) -> Dict[str, Any]:
        """메시지 처리 (핵심 비즈니스 로직)
        
        세션 ID 생성, 히스토리 검증, 로그 저장 등 모든 로직을 하나의 메서드로 캡슐화
        
        Args:
            message: 사용자 질문
            user_id: 사용자 ID
            session_id: 세션 ID (없으면 자동 생성)
            chat_model_id: 챗봇 모델 ID (선택)
            
        Returns:
            처리 결과 딕셔너리 (ChatResponse로 변환 가능)
        """
        start_time = datetime.now()
        
        # 1. 세션 ID 결정 (없으면 자동 생성)
        final_session_id = session_id
        if not final_session_id or not final_session_id.strip():
            final_session_id = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:10]}"
        else:
            final_session_id = final_session_id.strip()
        
        # 2. 히스토리 로드: Redis에서 자동 로드
        chat_history = await SessionService.get_history_async(final_session_id)
        
        # 3. 히스토리 검증 및 정제
        validated_history = ChatHistoryValidator.sanitize_history(
            chat_history=chat_history,
            guardrail_service=self.guardrail_service,
            trusted_source=True
        )
        
        # 4. 챗봇 설정 로드 (캐시 적용)
        chatbot_settings = await self._load_chatbot_settings(chat_model_id)
        
        # 5. 초기 상태 생성
        state = create_initial_state(
            user_message=message,
            user_id=user_id,
            session_id=final_session_id,
            chat_history=validated_history,
            chatbot_settings=chatbot_settings,
            is_requery=is_requery
        )
        
        # 6. LangGraph 워크플로우 실행 (response_format 전달)
        config = {**self.services_config, "response_format": response_format}
        result = await self.graph.ainvoke(state, config=config)
        
        # 7. user_message를 result에 추가 (FAQ 생성 시 필요)
        result["user_message"] = message
        
        # final_answer가 없으면 rag_answer나 cached_response 사용
        if not result.get("final_answer"):
            result["final_answer"] = result.get("rag_answer") or result.get("cached_response") or result.get("faq_answer")
            if result.get("final_answer"):
                logger.info(f"process_message: final_answer가 없어서 rag_answer/cached_response/faq_answer에서 복구: answer_len={len(result['final_answer'])}")
            else:
                logger.warning(f"process_message: final_answer가 없고 대체 답변도 없음, result keys={list(result.keys())}")
        
        # 8. 결과 포맷팅
        response_data = format_response(result)
        
        # 9. 세션 히스토리 저장
        await self._save_conversation_turn(
            session_id=final_session_id,
            user_id=user_id,
            user_message=message,
            result=result
        )
        
        # 9-1. 카테고리 ID 조회 (RAG 결과에 카테고리가 있는 경우)
        category_id = None
        rag_category = result.get("rag_category")
        logger.info(f"🔍 [Category] rag_category from result: {rag_category}")
        if rag_category:
            try:
                async with async_db_session() as session:
                    category_repo = CategoryRepository(session)
                    
                    # rag_category가 숫자 문자열이면 카테고리 ID로 조회
                    if isinstance(rag_category, str) and rag_category.isdigit():
                        potential_id = int(rag_category)
                        logger.debug(f"🔍 [Category] DB에서 category_id={potential_id} 조회 시도")
                        category = await category_repo.find_by_id(potential_id)
                        if category:
                            category_id = potential_id
                            # name 속성이 있을 수 있지만 없을 수도 있음
                            category_name = getattr(category, 'name', 'N/A')
                            logger.info(f"✅ Category ID verified from S3 path: {rag_category} -> category_id={category_id} (name: {category_name})")
                        else:
                            logger.warning(f"⚠️ Category ID {rag_category} from S3 path not found in database. Skipping category_id.")
                    else:
                        # 카테고리 이름으로 조회
                        category = await category_repo.find_by_name(rag_category)
                        if category:
                            category_id = category.id
                            logger.info(f"✅ Category resolved by name: {rag_category} -> category_id={category_id}")
                        else:
                            logger.warning(f"⚠️ Category not found by name: {rag_category}")
            except Exception as e:
                logger.error(f"Failed to resolve category id: {e}", exc_info=True)
        
        # 10. 채팅 로그 저장 (동기/비동기)
        end_time = datetime.now()
        latency_ms = int((end_time - start_time).total_seconds() * 1000)
        
        logger.info(f"💾 [Chat Log] 저장 준비: category_id={category_id}, sync_log={sync_log}")
        
        chat_log_id = await self._save_chat_log(
            message=message,
            result=result,
            chat_model_id=chat_model_id,
            start_time=start_time,
            end_time=end_time,
            latency_ms=latency_ms,
            sync_log=sync_log,
            is_requery=is_requery,
            category_id=category_id
        )
        
        if chat_log_id:
            response_data["chat_log_id"] = chat_log_id
        
        # 11. FAQ 후보 생성 (Celery 백그라운드)
        self._generate_faq_candidate(result, chatbot_settings)
        
        # 11. 최종 응답 데이터 구성
        response_data["session_id"] = final_session_id
        return response_data
    
    async def _load_chatbot_settings(
        self,
        chat_model_id: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """챗봇 설정 로드"""
        logger.debug(f"_load_chatbot_settings 호출: chat_model_id={chat_model_id}")
        
        if chat_model_id is None:
            # 기본값 사용 (프론트 UI에서 chat_model_id를 보내지 않은 경우)
            chat_model_id = DEFAULT_CHAT_MODEL_ID
            logger.info(
                f"⚠️ [챗봇 설정] chat_model_id가 없어 기본값 사용: model_id={DEFAULT_CHAT_MODEL_ID} "
                "(프론트 담당자: API 요청에 chat_model_id 필드를 추가하면 모델 선택 가능)"
            )
        
        try:
            # DB에서 조회 (캐싱은 ChatbotConfigSyncService 내부 또는 Redis에서 처리 권장)
            from app.infrastructure.persistence.session import async_db_session
            async with async_db_session() as session:
                config_service = ChatbotConfigSyncService(session)
                chatbot_settings = await config_service.get_config_for_model(chat_model_id)
                
                if chatbot_settings:
                    chatbot_settings["model_id"] = chat_model_id
                    logger.info(
                        f"🔄 [챗봇 설정 로드] model_id={chat_model_id}, "
                        f"temp={chatbot_settings.get('temperature')}, top_p={chatbot_settings.get('top_p')}, "
                        f"max_tokens={chatbot_settings.get('max_tokens')}, search_count={chatbot_settings.get('search_results_count')}, "
                        f"llm_model={chatbot_settings.get('llm_model', 'N/A')[:50]}, "
                        f"persona={chatbot_settings.get('persona_type', 'N/A')}"
                    )
                else:
                    logger.warning(f"⚠️ [챗봇 설정] DB에서 설정을 찾을 수 없음: model_id={chat_model_id}")
                
                return chatbot_settings
        except Exception as e:
            logger.warning(f"챗봇 설정 로드 실패 (기본값 사용): {e}", exc_info=True)
            return None
    
    async def _save_conversation_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        result: Dict[str, Any]
    ) -> None:
        """대화 턴 저장 (Redis)"""
        try:
            # 블랙리스트에 차단되어도 답변(거절 메시지)이 나갔다면 히스토리에 저장해야 문맥이 유지됨
            if result.get("final_answer"):
                final_answer = result.get("final_answer", "")
                
                # 보안: RAG LLM 출력 검증 (로깅용)
                blacklist_result = self.guardrail_service.check_blacklist(final_answer)
                if blacklist_result.blocked:
                    logger.warning(
                        f"⚠️ 보안: LLM 출력이 블랙리스트에 차단됨 (히스토리는 저장): "
                        f"category={blacklist_result.category}, reason={blacklist_result.reason}"
                    )
                
                # 히스토리 저장 (블랙리스트 차단 여부와 관계없이 저장)
                intent_type = result.get("intent_type")
                await SessionService.save_conversation_turn_async(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    assistant_message=final_answer,
                    intent_type=intent_type
                )
            else:
                logger.info(f"⚠️ 히스토리 저장 스킵: final_answer 없음")
        except Exception as e:
            logger.warning(f"세션 히스토리 저장 실패 (응답은 정상 반환): {e}")
    
    async def _save_chat_log(
        self,
        message: str,
        result: Dict[str, Any],
        chat_model_id: Optional[int],
        start_time: datetime,
        end_time: datetime,
        latency_ms: int,
        sync_log: bool = False,
        is_requery: bool = False,
        category_id: Optional[int] = None
    ) -> Optional[int]:
        """채팅 로그 저장 (동기/비동기)"""
        # 로그 상태 결정
        route = result.get("route", "")
        if result.get("blocked"):
            log_status = "GUARDRAIL"
        # User Request: 재질문이어도 새로운 로그는 SUCCESS로 저장 (이전 로그만 REQUERY로 업데이트)
        # elif is_requery or "requery" in route.lower():
        #     log_status = "REQUERY"
        else:
            log_status = "SUCCESS"
        
        # 챗봇 모델 ID 가져오기
        chatbot_settings_result = result.get("chatbot_settings", {})
        chat_model_id_for_log = chatbot_settings_result.get("model_id") if chatbot_settings_result else None
        if chat_model_id_for_log is None:
            chat_model_id_for_log = chat_model_id if chat_model_id is not None else 1
        
        # 저장할 데이터가 없으면 스킵
        if not (result.get("final_answer") or result.get("blocked")):
            return None

        # 토큰 사용량 추출 (LangGraph는 prompt_tokens, completion_tokens 사용)
        token_usage = result.get("token_usage", {})
        if isinstance(token_usage, dict):
            prompt_tokens = token_usage.get("prompt_tokens") or 0
            completion_tokens = token_usage.get("completion_tokens") or 0
        else:
            prompt_tokens = 0
            completion_tokens = 0

        # 가드레일 차단 사유 추출 및 매핑
        guardrail_reason = None
        if log_status == "GUARDRAIL":
            category = result.get("guardrail_category") or result.get("blacklist_category") or result.get("category")
            # GuardrailReason Enum 매핑 (Java 백엔드와 일치)
            category_map = {
                "profanity": "BLOCKED_PROFANITY",
                "sexual": "BLOCKED_SEXUAL",
                "violence": "BLOCKED_VIOLENCE",
                "personal_info": "BLOCKED_PERSONAL_INFO",
                "spam": "BLOCKED_SPAM",
                "off_topic": "BLOCKED_OFF_TOPIC",
                "length": "BLOCKED_TOO_LONG", # 기본적으로 길이 초과로 매핑 (상세 구분 필요 시 reason 확인)
            }
            guardrail_reason = category_map.get(category, "BLOCKED_GENERAL")
            
            # 길이 관련 상세 구분
            if category == "length":
                reason_text = result.get("reason", "")
                if "짧습니다" in reason_text:
                    guardrail_reason = "BLOCKED_TOO_SHORT"

        if sync_log:
            # 동기 저장 (DB에 직접 저장하고 ID 반환)
            try:
                from app.infrastructure.persistence.session import async_db_session
                from app.domain.services.chat_log_service import ChatLogService
                
                async with async_db_session() as session:
                    chat_log_service = ChatLogService(session)
                    
                    # Requery인 경우 이전 로그 상태 업데이트 (최근 1시간 내 동일 쿼리/모델)
                    if is_requery:
                        updated = await chat_log_service.update_previous_log_status(
                            query=message,
                            chat_model_id=chat_model_id_for_log
                        )
                        if updated:
                            logger.info(f"🔄 [Requery] 이전 채팅 로그 상태를 REQUERY로 업데이트함: query={message[:20]}...")
                        else:
                            logger.info(f"⚠️ [Requery] 이전 채팅 로그를 찾을 수 없음: query={message[:20]}...")

                    chat_log = await chat_log_service.save_log(
                        query=message,
                        response=result.get("final_answer", ""),
                        status=log_status,
                        chat_model_id=chat_model_id_for_log,
                        input_time=start_time,
                        output_time=end_time,
                        latency_ms=latency_ms,
                        category_id=category_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        guardrail_reason=guardrail_reason
                    )
                    logger.info(f"✅ [Sync] 채팅 로그 저장 완료: id={chat_log.id}, status={log_status}, category_id={category_id}")
                    return chat_log.id
            except Exception as e:
                logger.error(f"⚠️ [Sync] 채팅 로그 저장 실패: {e}", exc_info=True)
                return None
        else:
            # 비동기 저장 (Celery Task)
            save_chat_log_task.delay(
                query=message,
                response=result.get("final_answer", ""),
                status=log_status,
                chat_model_id=chat_model_id_for_log,
                input_time_iso=start_time.isoformat(),
                output_time_iso=end_time.isoformat(),
                latency_ms=latency_ms,
                category_id=category_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                guardrail_reason=guardrail_reason
            )
            logger.info(f"💾 [Celery] 채팅 로그 저장 작업 큐에 추가: status={log_status}, category_id={category_id}")
            return None
    
    def _generate_faq_candidate(
        self,
        result: Dict[str, Any],
        chatbot_settings: Optional[Dict[str, Any]]
    ) -> None:
        """FAQ 후보 생성 (Celery 백그라운드)"""
        should_gen = result.get("should_generate_faq", False)
        rag_ans = result.get("rag_answer")
        rag_conf = result.get("rag_confidence", 0)
        user_msg = result.get("user_message", "") or ""
        
        # None 값 처리
        rag_conf_str = f"{rag_conf:.3f}" if rag_conf is not None else "None"
        user_msg_preview = user_msg[:50] if user_msg else ""
        
        logger.info(f"🔍 [FAQ] 생성 체크: should_gen={should_gen}, has_rag_ans={bool(rag_ans)}, conf={rag_conf_str}, min_conf={FAQ_MIN_CONFIDENCE}, question='{user_msg_preview}...'")
        
        if (should_gen and 
            rag_ans and 
            rag_conf is not None and
            rag_conf >= FAQ_MIN_CONFIDENCE):
            # User Request: 실시간 FAQ 생성 활성화
            try:
                # None 값 처리
                rag_conf_value = rag_conf if rag_conf is not None else 0.0
                user_msg_safe = user_msg or ""
                
                task_result = generate_faq_candidate_task.delay(
                    user_question=user_msg_safe,
                    rag_answer=rag_ans,
                    rag_confidence=rag_conf_value,
                    chatbot_settings=chatbot_settings or result.get("chatbot_settings")
                )
                logger.info(
                    f"✅ [FAQ] Celery Task 큐에 추가 완료: "
                    f"task_id={task_result.id}, confidence={rag_conf_value:.3f}, "
                    f"question='{user_msg_safe[:50]}...'"
                )
            except Exception as e:
                rag_conf_str = f"{rag_conf:.3f}" if rag_conf is not None else "None"
                user_msg_preview = user_msg[:50] if user_msg else ""
                logger.error(
                    f"❌ [FAQ] Celery Task 큐 추가 실패: {e}, "
                    f"confidence={rag_conf_str}, question='{user_msg_preview}...'",
                    exc_info=True
                )
        else:
            reason = []
            if not should_gen:
                reason.append("should_generate_faq=False")
            if not rag_ans:
                reason.append("rag_answer=None")
            if rag_conf is None:
                reason.append("confidence=None")
            elif rag_conf < FAQ_MIN_CONFIDENCE:
                reason.append(f"confidence={rag_conf:.3f} < {FAQ_MIN_CONFIDENCE}")
            logger.info(f"⚠️ [FAQ] 생성 스킵: {', '.join(reason)}")

