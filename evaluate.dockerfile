# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY src/ src/
COPY data/ data/
RUN mkdir -p models/ reports/figures/

# Install project dependencies using uv package manager
WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

# Set environment variable to ensure Python output is unbuffered
ENV PYTHONUNBUFFERED=1  

# Command to run the training script when the container starts
ENTRYPOINT ["uv", "run", "src/mlops_project/evaluate.py"]