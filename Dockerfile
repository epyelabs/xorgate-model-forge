# xorgate-model-forge (xmf): two images from one Dockerfile (design 3a).
#
#   cpu   python:3.11-slim + CPU torch; linux/amd64 and linux/arm64
#         (the laptop, the ECS cluster, SageMaker ml.m5 CPU jobs)
#   cuda  a PyTorch CUDA runtime base; linux/amd64 (SageMaker ml.g4dn / ml.g6)
#
# Both bake the recipes' base weights under /opt/forge/weights so a training
# job runs with network isolation on, set WORKDIR /work for the mounted
# local case, and default to the SageMaker training-job entrypoint.
#
#   docker build --target cpu -t xmf .
#   docker buildx build --target cpu --platform linux/amd64,linux/arm64 -t xmf .
#   docker buildx build --target cuda --platform linux/amd64 -t xmf:cuda .
#   docker run --rm -v "$PWD":/work xmf run fixtures/specs/quick.json --dataset fixtures/ds --out out
#   docker run --rm -v "$PWD":/work xmf doctor
#
# The forge is installed from the checkout with pip (not uv) so the cuda
# image keeps the torch its base ships; the cpu image installs torch from
# the CPU wheel index first so pip never resolves the CUDA stack.

ARG PYTORCH_CUDA_BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# --- cpu --------------------------------------------------------------------------
FROM python:3.11-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XMF_WEIGHTS_DIR=/opt/forge/weights \
    XMF_WEIGHTS_PULL=0 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    OMP_NUM_THREADS=4

# opencv (an ultralytics dependency) needs libGL and glib at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" "torchvision>=0.19"

WORKDIR /build
COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src ./src
COPY contracts ./contracts
RUN pip install ".[all]" && rm -rf /build

# Bake the base weights (the recipes' pretrained refs) so jobs never download.
RUN XMF_WEIGHTS_PULL=1 xmf weights pull --dir /opt/forge/weights \
    && ls -la /opt/forge/weights

WORKDIR /work
ENTRYPOINT ["xmf"]
CMD ["run", "/opt/ml/input/config/job.json", "--host", "sagemaker"]

# --- cuda -------------------------------------------------------------------------
FROM ${PYTORCH_CUDA_BASE} AS cuda

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XMF_WEIGHTS_DIR=/opt/forge/weights \
    XMF_WEIGHTS_PULL=0 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src ./src
COPY contracts ./contracts
# torch and torchvision are the base image's; pip leaves a satisfied requirement alone.
RUN pip install ".[all]" && rm -rf /build

RUN XMF_WEIGHTS_PULL=1 xmf weights pull --dir /opt/forge/weights \
    && ls -la /opt/forge/weights

WORKDIR /work
ENTRYPOINT ["xmf"]
CMD ["run", "/opt/ml/input/config/job.json", "--host", "sagemaker"]
