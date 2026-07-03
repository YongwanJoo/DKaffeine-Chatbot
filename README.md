# ☕ DKaffeine - 카카오워크 RAG 기반 AI 챗봇

> **가천 SW 아카데미 7기 - 기업 실무 프로젝트 (DK Techin)**

**기업 내규와 정책 문서**를 기반으로 정확한 **질의응답**을 제공하는 **RAG 챗봇**과 **관리자 운영 시스템**을 구축하고,  
검색 성능 최적화, 보안, 비용 절감 및 운영 자동화를 통해 고객사가 지속적으로 품질을 개선할 수 있는 **AI 플랫폼**입니다.

<br/>

## 📌 Tech Stack

| 분류 | 기술 |
|:---|:---|
| **Backend** | FastAPI · LangGraph · Python 3.12 |
| **Database** | PostgreSQL (HA Proxy) · Redis · AWS Knowledge Base (VectorDB) |
| **AI & LLM** | AWS Bedrock (Claude Sonnet 4.5 · Claude Haiku 4.5 · Titan Embeddings v2) |
| **Infra & DevOps** | Kakao Cloud · Kubernetes · ArgoCD · Jenkins · GitHub Actions · Kaniko |
| **Monitoring** | Prometheus · Grafana · ELK Stack (Elasticsearch · Logstash · Kibana · Filebeat) |
| **Task Queue** | Celery + Redis |

<br/>

---

## 1. 기획 의도

### 배경 및 목적

최근 수많은 기업이 LLM을 활용한 기술 지원 및 사내 지식 검색 시스템을 도입하고 있습니다.  
하지만 LLM의 고질적인 문제인 **'환각(Hallucination)'** 을 극복하기 위해,  
외부의 신뢰할 수 있는 사내 문서를 근거로 제시하는 **RAG(Retrieval-Augmented Generation)** 기술이 필수가 되었습니다.

DKaffeine 챗봇은 산재된 사내 기술 문서, HR 정책, 고객 응대 매뉴얼 등 기업의 핵심 자산을  
**통합 검색 엔진**으로 구축하여, 재학습 없이도 실시간으로 최신 정보를 제공하는 신뢰성 있는 AI 챗봇을 목표로 합니다.

### 핵심 목표

| 목표 | 설명 |
|:---|:---|
| **자율적인 운영 시스템** | 개발자 개입 없이 운영팀이 직접 데이터를 추가/수정/삭제하고 챗봇 품질을 관리할 수 있는 **통합 어드민 대시보드** |
| **RAG 기반 신뢰성 있는 응답** | 사전 학습 데이터가 아닌, 기업의 실제 내부 데이터를 기반으로 답변을 생성하여 **정보의 정확성 보장** |

<br/>

---

## 2. 핵심 개선 사항

### 📊 RAG 성능 비교 및 최적화

다양한 청킹(Chunking) 방식과 토큰 사이즈를 변경해가며 테스트하여 최적의 RAG 품질을 도출했습니다.

- **Semantic Chunking 도입**: 계층 기반 Chunking → 의미 기반 청킹(Semantic Chunking)으로 전환
- **최적 토큰 사이즈 적용**: 500~700 토큰 범위에서 최적의 정확도 달성
- **정확도 40% 증가**: 기존 대비 약 **40% 증가**하여 최고 **0.60** 기록
- **Reranker 도입**: 최대 **+0.26점** 추가 향상

<p align="center">
  <img src="assets/성능테스트.png" alt="RAG 성능 테스트 결과" width="700"/>
</p>
<p align="center"><em>▲ 청킹 방식 및 토큰 사이즈별 RAG 정확도 비교 결과</em></p>

<br/>

### ⚡ 성능, 비용 및 안정성 개선

| 개선 항목 | 내용 |
|:---|:---|
| **Token 비용 절감** | 반복 유사 질문의 LLM 호출 비용 절감을 위해 **Redis 기반 응답 캐시** 적용 및 **FAQ 관리 기능** 구축 |
| **보안성 강화** | Prompt Injection, 민감 정보 유출, 악의적 입력 방지를 위한 **다단계 Guardrail** 적용 (Blacklist → LLM 기반 분석) |
| **응답 경험 개선** | RAG 응답 지연 문제 해결을 위한 **Celery 기반 비동기 처리** 구조 설계 |
| **배포 안정성** | **Multi-stage Build** + `slim` 이미지 적용으로 컨테이너 이미지 용량 **약 80% 감소** |

<br/>

---

## 3. 아키텍처

프로젝트는 **Clean Architecture** 원칙을 따라 Domain, Infrastructure, Orchestration, Presentation 레이어로  
명확히 분리하여 결합도를 낮추고 테스트 용이성을 높였습니다.

### 논리 아키텍처

전체 시스템의 계층 구조와 모듈 간 의존 관계를 나타냅니다.

<p align="center">
  <img src="assets/논리%20아키텍처.png" alt="논리 아키텍처" width="700"/>
</p>

<br/>

### 시스템 아키텍처

Kakao Cloud 기반 Kubernetes 환경에서의 전체 인프라 구성도입니다.  
Jenkins CI → Kaniko 빌드 → KCR 푸시 → ArgoCD GitOps 배포의 자동화된 파이프라인을 운영합니다.

<p align="center">
  <img src="assets/시스템%20아키텍처.png" alt="시스템 아키텍처" width="700"/>
</p>

<br/>

### AI 아키텍처

RAG 파이프라인의 내부 구조입니다.  
AWS Bedrock 기반 임베딩 및 LLM 호출, Knowledge Base를 통한 벡터 검색을 수행합니다.

<p align="center">
  <img src="assets/AI%20아키텍처.png" alt="AI 아키텍처" width="700"/>
</p>

<br/>

### AI 서비스 워크플로우 (LangGraph)

**LangGraph** 기반의 상태 머신(State Machine) 방식으로 설계된 챗봇 응답 파이프라인입니다.

```
사용자 입력
  │
  ▼
[Blacklist Check] ─── 차단 ──→ END (차단 응답)
  │ 통과
  ▼
[Unified Analysis] ─── 가드레일 위반 ──→ END (차단 응답)
  │
  ├── 일상 대화 ──→ [Casual Response] ──→ END
  ├── 뉴스 요청 ──→ [News Summary] ──→ END
  │
  ▼ (업무 질문)
[Cache Check]
  │
  ├── 캐시 히트 ──→ [FAQ Verify] ──→ RAG 또는 END
  └── 캐시 미스 ──→ [FAQ Search] ──→ RAG 또는 END
                                        │
                                        ▼
                                      [RAG]
                                        │
                                  [Confidence Check]
                                   │            │
                                   ▼            ▼
                              [Save Tokens]  [RAG Fallback]
                                   │            │
                                   ▼            ▼
                                  END          END
```

<p align="center">
  <img src="assets/AI%20서비스%20워크플로우.png" alt="AI 서비스 워크플로우" width="700"/>
</p>
<p align="center"><em>▲ LangGraph 기반 챗봇 응답 워크플로우</em></p>

<br/>

---

## 4. 관리자 운영 시스템 (Admin)

고객사가 **개발자 개입 없이** 지속적으로 품질을 개선할 수 있도록 설계된 통합 관리자 어드민입니다.

<br/>

### 🔧 관리자 설정

챗봇의 LLM 모델, 온도(Temperature), 시스템 프롬프트 등을 관리자가 직접 조정할 수 있습니다.  
설정 변경 이력이 자동 기록되어 추적 가능성(Traceability)을 확보합니다.

<p align="center">
  <img src="assets/관리자%20설정.png" alt="관리자 설정 화면" width="700"/>
</p>

<br/>

### 👥 사용자 관리

챗봇을 이용하는 사용자 목록과 권한을 관리합니다.

<p align="center">
  <img src="assets/사용자%20관리.png" alt="사용자 관리 화면" width="700"/>
</p>

<br/>

### 📂 데이터 소스 관리

PDF, PPT 등 사내 문서를 업로드하면 **자동으로 청킹 및 임베딩** 처리합니다.  
문서 업데이트 시 전/후 **버전 비교(Diff)** 기능으로 변경 사항을 시각적으로 확인할 수 있습니다.

<p align="center">
  <img src="assets/데이터%20관리.png" alt="데이터 소스 관리 화면" width="700"/>
</p>

<br/>

### ❓ FAQ 생성 및 관리

다빈도 질의와 RAG를 통해 신뢰도 높게 답변된 내역을 분석하여  
**새로운 FAQ 후보를 관리자에게 자동 추천**합니다.  
승인된 FAQ는 캐시에 반영되어 후속 질문에 즉시 응답할 수 있습니다.

<p align="center">
  <img src="assets/FAQ%20관리.png" alt="FAQ 관리 화면" width="700"/>
</p>

<br/>

### 📈 분석 대시보드

총 대화 횟수, 평균 응답 시간, 모델별 API 비용, 가드레일 차단 비율,  
카테고리별 사용 패턴 등을 시각화하여 운영 현황을 한눈에 파악할 수 있습니다.

<p align="center">
  <img src="assets/분석%20대시보드.png" alt="분석 대시보드 화면" width="700"/>
</p>

<br/>

---

## 5. 챗봇 사용 예시 및 가드레일

### 💬 챗봇 사용 예시

사용자가 카카오워크에서 업무 관련 질문을 하면, RAG 파이프라인을 통해  
사내 문서를 기반으로 **근거가 포함된 정확한 답변**을 제공합니다.

<p align="center">
  <img src="assets/디카페인%20사용%20예시.png" alt="DKaffeine 챗봇 사용 예시" width="700"/>
</p>

<br/>

### 🛡️ 가드레일 (Guardrail)

업무 외 질문, Prompt Injection, 민감 정보 요청 등 **악의적이거나 부적절한 입력**을 차단합니다.  
다단계 필터링 구조(Blacklist → LLM 기반 분석)로 정교하게 동작합니다.

<p align="center">
  <img src="assets/가드레일%20예시1.png" alt="가드레일 차단 예시 1" width="500"/>
</p>
<p align="center"><em>▲ 업무 외 질문 차단 예시</em></p>

<br/>

<p align="center">
  <img src="assets/가드레일%20예시%202.png" alt="가드레일 차단 예시 2" width="500"/>
</p>
<p align="center"><em>▲ Prompt Injection 차단 예시</em></p>

<br/>

---

## 6. 프로젝트 구조

```
DKaffeine-Chatbot/
├── src/app/
│   ├── domain/                  # 🔵 Domain Layer
│   │   ├── entities/            #    도메인 모델 (DTO, Value Object)
│   │   ├── ports/               #    인터페이스 (Repository, Service Port)
│   │   ├── services/            #    비즈니스 로직 (Use Case)
│   │   │   ├── chat_usecase.py
│   │   │   ├── faq_generation_service.py
│   │   │   ├── session_service.py
│   │   │   └── ...
│   │   ├── constants.py
│   │   └── exceptions.py
│   │
│   ├── infrastructure/          # 🟢 Infrastructure Layer
│   │   ├── adapters/            #    외부 서비스 어댑터
│   │   │   ├── cache/           #      Redis 캐시
│   │   │   ├── guardrail/       #      가드레일 (Blacklist + LLM)
│   │   │   ├── llm/             #      AWS Bedrock LLM 클라이언트
│   │   │   ├── rag/             #      RAG 파이프라인
│   │   │   ├── retrievers/      #      벡터 검색 리트리버
│   │   │   └── ...
│   │   ├── persistence/         #    DB 접근 (PostgreSQL)
│   │   └── config/              #    설정 로더
│   │
│   ├── orchestration/           # 🟠 Orchestration Layer (LangGraph)
│   │   ├── graph.py             #    LangGraph 워크플로우 정의
│   │   ├── edges.py             #    조건부 라우팅 로직
│   │   ├── state.py             #    그래프 상태 정의
│   │   └── nodes/               #    각 노드 구현
│   │       ├── blacklist.py
│   │       ├── unified_analysis.py
│   │       ├── cache.py
│   │       ├── faq.py
│   │       ├── rag.py
│   │       └── ...
│   │
│   ├── presentation/            # 🔴 Presentation Layer
│   │   └── http/
│   │       ├── controllers/     #    FastAPI 라우터
│   │       └── models/          #    요청/응답 스키마
│   │
│   ├── setup/                   #    앱 초기화 및 DI 설정
│   ├── run.py                   #    FastAPI 앱 엔트리포인트
│   └── worker.py                #    Celery Worker 설정
│
├── config/
│   ├── guardrail_config.json    #    가드레일 규칙 설정
│   ├── blacklist.json           #    블랙리스트 키워드
│   ├── faq.json                 #    FAQ 데이터
│   └── prod/                    #    환경별 설정 파일
│
├── Dockerfile                   #    Multi-stage 컨테이너 빌드
├── Jenkinsfile.prod             #    CI/CD 파이프라인 (Kaniko + ArgoCD)
└── README.md
```

<br/>

---

## 7. Quick Start

### 사전 요구 사항

- Python 3.12+
- Docker & Docker Compose
- AWS Bedrock 접근 권한 (Claude Sonnet 4.5, Haiku 4.5, Titan Embeddings v2)
- PostgreSQL 15+
- Redis 7+

### 로컬 실행

```bash
# 1. 저장소 클론
git clone https://github.com/YongwanJoo/DKaffeine-Chatbot.git
cd DKaffeine-Chatbot

# 2. 환경변수 설정
cp config/prod/.secrets.toml.example .secrets.toml
# .secrets.toml 파일에 AWS Bedrock 자격 증명, Redis, PostgreSQL 정보를 입력합니다.

# 3. Redis 및 PostgreSQL 실행
docker run -d --name redis -p 6379:6379 redis:latest
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=dkaffeine \
  postgres:latest

# 4. 의존성 설치
pip install -e .

# 5. Celery Worker 시작 (별도 터미널)
celery -A app.worker.celery_app worker --loglevel=info

# 6. FastAPI 서버 시작
./run.sh
```

### Docker 빌드

```bash
# 프로덕션 이미지 빌드
docker build -t dkaffeine-chatbot:latest .

# 실행
docker run -p 8000:8000 \
  --env-file .secrets.toml \
  dkaffeine-chatbot:latest
```

<br/>

---

## 8. CI/CD 파이프라인

```
GitHub Push (main)
      │
      ▼
  [Jenkins]
      │
      ▼
  [Kaniko Build] ──→ KCR (Kakao Container Registry)
      │
      ▼
  [Helm values.yaml 업데이트] ──→ Infra Repo Push
      │
      ▼
  [ArgoCD] ──→ Kubernetes 자동 배포
```

- **Jenkins**: `main` 브랜치 변경 감지 시 자동 트리거
- **Kaniko**: DinD 없이 안전한 컨테이너 이미지 빌드
- **ArgoCD**: GitOps 기반 무중단 자동 배포

<br/>

---

## 9. 회고

### 팀워크 및 협업 방식

| 도구 | 용도 |
|:---|:---|
| **Notion** | 회의록 및 WBS 관리 |
| **Figma** | 화면 설계 및 프로토타이핑 |
| **GitHub** | 이슈 관리 및 코드 리뷰 (Git Flow 전략) |
| **Google Sheets** | 테이블 명세서, 예산안 관리 |

- 매일 **데일리 스크럼**을 통해 진행 상황과 블로커(Blocker)를 공유
- **Git Flow** 브랜치 전략 및 파트별 TL 체제를 통한 체계적 협업

### 아쉬운 점

- **LLM 성능 검증 파이프라인 부재**: 모델 배포 시점마다 성능 편차가 발생했으나, **LLM-as-a-Judge**와 같은 자동화된 정량적 평가 방식을 CI/CD 파이프라인에 통합하지 못했습니다. 결과적으로 성능 저하 상태의 배포를 사전에 완벽히 차단하지 못한 점이 아쉬움으로 남습니다.

### 향후 로드맵

- **서비스 안정화**: JMeter를 활용한 부하 테스트로 다수 사용자 대상 안정적 서비스 운영
- **기능 추가**: 음성 인식 기반 회의 필기 요약 및 이미지 스케치 기반 아이디어 요약 파이프라인 도입
