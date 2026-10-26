# Training image for Vertex AI custom jobs (linux/amd64, NVIDIA GPU).
# The CUDA runtime libraries come with the torch wheel; the GPU driver comes from the host machine.
#
# Build in Cloud Build (native amd64, nothing large uploaded from this machine):
#   gcloud builds submit --tag <region>-docker.pkg.dev/<project>/<repo>/cs336-train:<tag> .
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

# Use the image's Python instead of downloading one, and never re-sync at `uv run` time
# (that would pull in the dev group: jupyter, matplotlib, ...).
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_SYNC=1 \
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Dependencies first, so a code change doesn't reinstall torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY cs336_basics ./cs336_basics
COPY tokenizers ./tokenizers
COPY scripts ./scripts
RUN uv sync --locked --no-dev

ENTRYPOINT ["scripts/vertex_entrypoint.sh"]
