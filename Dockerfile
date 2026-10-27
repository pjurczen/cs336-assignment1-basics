# Training image for Vertex AI custom jobs (linux/amd64, NVIDIA GPU).
# The CUDA runtime libraries come with the torch wheel; the GPU driver comes from the host machine.
#
# Build in Cloud Build (native amd64, nothing large uploaded from this machine): scripts/build_image.sh
# Pinned by digest: the tag is re-published upstream, and a new base image invalidates every cached layer.
FROM python:3.13-slim-bookworm@sha256:e32be0e55814a08977501ca863271d9bd02443710db8f2afaf8372e2c4508530

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

# torch.compile (inductor) generates Triton kernels, and Triton compiles a small C helper at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Code only. Tokenizers and datasets live in the bucket and are copied in when a job starts (vertex_entrypoint.sh).
COPY README.md ./
COPY cs336_basics ./cs336_basics
COPY scripts ./scripts
RUN uv sync --locked --no-dev

# Vertex AI mounts the host's NVIDIA driver (libcuda, nvidia-smi) at /usr/local/nvidia. The nvidia/cuda base images
# put it on these paths; this slim Python image has to do it itself. Set last, so the dependency layers stay cached.
ENV PATH="/usr/local/nvidia/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/nvidia/lib" \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["scripts/vertex_entrypoint.sh"]
