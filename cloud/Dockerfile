# syntax=docker/dockerfile:1

# Base image: newest patch of the newest published CUDA minor line, verified against the Docker Hub
# tag API on 2026-08-19. 13.3.1 (pushed 2026-07-28) is the newest 13.3 patch and there is no 13.4 or
# 14.x line yet. The repo's torch 2.13.0 wheel is a CUDA 13.0.3 build, which runs fine here under
# CUDA minor-version compatibility.
#
# Query the registry rather than trusting this comment when you next touch the file -- an earlier
# revision of it asserted 13.3.0 was the only 13.3 patch on a date when 13.3.1 had already been
# published for three weeks, so a stale base shipped while the comment said otherwise:
#   curl -s 'https://hub.docker.com/v2/repositories/nvidia/cuda/tags?page_size=100&name=13.' \
#     | jq -r '.results[].name' | grep runtime-ubuntu24.04
# A stale base is the usual source of the CRITICAL findings cloud/push_ecr.sh refuses to publish on.
#
# `runtime` rather than `devel` on purpose: no nvcc and no toolchain, which is both a smaller
# image and a much smaller CVE surface, and nothing in the v1 path compiles CUDA kernels. Bumping
# to a devel base is the first thing to change if the DeltaNet fast kernels ever get built in --
# see docs/scratch/games-cloud-notes-2026-08-17.md, which also records the driver floor this
# implies (CUDA 13.x needs host driver >= 580, so the Batch AMI must be new enough).
FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04

# Pinned to the uv that produced the tracked uv.lock on the dev box, so the image resolves the
# same dependency graph the local gates ran against. This one does NOT track upstream: it is
# deliberately behind (0.12.5 was current on 2026-08-19) because the pin's job is to match whatever
# `uv --version` reports on the dev box. Bumping it means regenerating uv.lock with the same newer
# uv and re-running the gates, in that order -- not editing this line.
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /usr/local/bin/

# Baked at build time so a running job can always say which commit it is. Empty is a hard error:
# an unattributable training run is the failure the plan's stale-image rule exists to prevent.
ARG GIT_SHA
ENV GIT_SHA=${GIT_SHA}
RUN test -n "${GIT_SHA}" || { \
      echo "GIT_SHA build-arg is required; build via cloud/push_ecr.sh, not a bare docker build." \
      >&2; exit 1; }

ENV DEBIAN_FRONTEND=noninteractive
# Deliberately unpinned: ca-certificates is exactly the package that must track upstream
# security updates, and pinning it would freeze the trust store the CVE rule wants current.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PYTHONUNBUFFERED is load-bearing, not hygiene: without it a job's logs arrive in CloudWatch in
# 8 KB blocks, so a run that dies mid-step appears to have stopped logging minutes earlier and the
# actual failure is the part still sitting in the buffer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/games \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH}

WORKDIR /opt/games

# Dependencies before source, so editing code does not re-resolve torch. `--no-dev` leaves ruff,
# pytest, and basedpyright out of the image: the gates run on the dev box, and every package that
# is not here is a package that cannot carry a CVE into Batch.
#
# UV_EXTRAS is empty by default because v1 trains on TRL's native generate path. Pass
# `--build-arg UV_EXTRAS=--extra vllm` for the 27B two-GPU topology, which is the only thing that
# needs the vllm wheel (and roughly doubles the image).
ARG UV_EXTRAS=""
COPY pyproject.toml uv.lock ./

# awscli v1 from PyPI rather than the v2 zip installer: v1 covers `s3 sync` completely, installs
# as a normal Python tool with no curl/unzip step, and keeps the image free of a second package
# manager. The sync command is configurable anyway (see games/s3_sync.py), so swapping in v2 is a
# one-variable change for anyone who wants it. One layer with the sync above because both depend
# only on the lockfile copied above them.
RUN uv sync --frozen --no-dev ${UV_EXTRAS} \
    && uv tool install --no-progress awscli

COPY . .

# Model weights are deliberately NOT baked. HF_HOME points at container-local scratch and the
# weights are pulled when the job starts: baking them would make every image tens of gigabytes,
# tie an image to one model, and quietly serve a stale snapshot after an upstream update.
ENV HF_HOME=/scratch/hf \
    HF_HUB_DISABLE_TELEMETRY=1
RUN mkdir -p /scratch/hf /scratch/runs

ENTRYPOINT ["/opt/games/cloud/entrypoint.sh"]
