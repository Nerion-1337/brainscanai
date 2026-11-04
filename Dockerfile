# syntax=docker/dockerfile:1.7
# Image reproductible : uv (dépendances verrouillées par uv.lock) + PyTorch CPU ou CUDA au choix.
#   CPU (défaut) : docker build -t brainscanai .
#   GPU          : docker build --build-arg TORCH_VARIANT=cu121 -t brainscanai:gpu .
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
WORKDIR /app

# ---- Couche dépendances (mise en cache tant que pyproject/uv.lock ne changent pas)
FROM base AS deps
ARG TORCH_VARIANT=cpu
COPY pyproject.toml uv.lock .python-version ./
# 1) tout sauf torch/torchvision, versions figées par uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev \
        --no-install-package torch --no-install-package torchvision
# 2) torch/torchvision depuis l'index PyTorch (cpu par défaut : image bien plus légère)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python torch torchvision \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# ---- Image d'exécution
FROM deps AS runtime
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY notebooks ./notebooks
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-package torch --no-install-package torchvision
RUN useradd -m appuser && mkdir -p data reports && chown -R appuser /app
USER appuser
# Les données et résultats sont montés en volumes (jamais dans l'image)
VOLUME ["/app/data", "/app/reports"]
CMD ["python", "scripts/inspect_data.py"]

# ---- Cible de développement / tests (pytest, ruff, jupyter)
FROM deps AS dev
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev --group notebooks \
        --no-install-package torch --no-install-package torchvision
EXPOSE 8888
CMD ["pytest"]
