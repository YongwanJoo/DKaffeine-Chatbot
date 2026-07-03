FROM python:3.12-slim

WORKDIR /app

# 런타임 의존성 및 빌드 도구 설치 (빌드 후 제거 예정)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2 \
    libxslt1.1 \
    libgomp1 \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 프로젝트 파일 복사
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

# config.toml 복사 (일반 설정)
# 환경별 설정 파일 지원 (ENV_NAME 빌드 인자로 선택, 기본값: prod)
# 예: docker build --build-arg ENV_NAME=prod
# 예: docker build --build-arg ENV_NAME=dev
ARG ENV_NAME=prod
COPY config/${ENV_NAME}/config.toml ./config.toml

# 의존성 및 프로젝트 설치 (시스템 Python에 직접 설치, 가상환경 없이)
# 1. PyTorch CPU 버전 먼저 설치 (이미지 크기 및 빌드 메모리 최적화)
# --index-url을 사용하여 CPU 버전을 명시적으로 지정
RUN uv pip install --system --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. 나머지 프로젝트 의존성 설치
# pyproject.toml의 의존성을 설치하되, 이미 설치된 torch는 건너뜀
RUN uv pip install --system --no-cache . && \
    rm -rf /root/.cache/uv

# 빌드 도구 제거 (이미지 크기 최적화)
RUN apt-get purge -y build-essential libxml2-dev libxslt1-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 환경 변수 설정
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV APP_ENV=prod

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행 (uv run은 가상환경을 자동으로 찾아서 사용)
CMD ["uv", "run", "uvicorn", "app.run:app", "--host", "0.0.0.0", "--port", "8000"]
