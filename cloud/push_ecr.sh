#!/usr/bin/env bash
# Build the games training image and push it to ECR as both :latest and :<sha>.
#
# Two tags, one resolution rule. `:<sha>` exists so any image can be traced to a commit forever;
# `:latest` is the only tag a job definition ever names, because Batch resolves a tag to a digest
# once, at job-definition registration, and never again. Submitting against `:<sha>` would work but
# would tempt reuse of an old job-definition revision, which is precisely the silent stale-code
# failure cloud/submit_job.py exists to prevent.
#
# The build context is `git archive HEAD`, not the working tree. That makes "the image contains
# only committed code" structural rather than a promise: an uncommitted edit cannot reach the
# image even by accident, the baked GIT_SHA provably describes the contents, and .venv, artifacts/,
# and .git stay out of the context without needing a .dockerignore.
#
# Usage:
#   cloud/push_ecr.sh <ecr-repository> [aws-region]
# Example:
#   cloud/push_ecr.sh games-grpo us-west-2

set -Eeuo pipefail

REPOSITORY="${1:-}"
REGION="${2:-${AWS_REGION:-us-west-2}}"

if [[ -z "${REPOSITORY}" ]]; then
  echo "usage: cloud/push_ecr.sh <ecr-repository> [aws-region]" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

# ------------------------------------------------------------------------------ refuse a dirty tree
DIRTY="$(git status --porcelain)"
if [[ -n "${DIRTY}" ]]; then
  echo "ERROR: the working tree is dirty, so the pushed image would not match any commit." >&2
  echo "       Commit or stash first; the image is built from git archive HEAD." >&2
  echo "${DIRTY}" >&2
  exit 1
fi

GIT_SHA="$(git rev-parse HEAD)"
SHORT_SHA="$(git rev-parse --short HEAD)"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPOSITORY}"

echo "repository=${IMAGE}"
echo "git_sha=${GIT_SHA}"
echo "region=${REGION}"

# ECR repositories are not created implicitly; describe-repositories tells us plainly rather than
# letting the push fail with a less obvious error.
if ! aws ecr describe-repositories --repository-names "${REPOSITORY}" --region "${REGION}" \
  >/dev/null 2>&1; then
  echo "ERROR: ECR repository ${REPOSITORY} does not exist in ${REGION}." >&2
  echo "       Create it first: aws ecr create-repository --repository-name ${REPOSITORY} \\" >&2
  echo "         --region ${REGION} --image-scanning-configuration scanOnPush=true" >&2
  exit 1
fi

aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

# ---------------------------------------------------------------------------------- build from HEAD
echo "building from git archive HEAD (committed tree only)"
git archive --format=tar HEAD \
  | DOCKER_BUILDKIT=1 docker build \
    --file cloud/Dockerfile \
    --build-arg "GIT_SHA=${GIT_SHA}" \
    --tag "${IMAGE}:latest" \
    --tag "${IMAGE}:${GIT_SHA}" \
    --tag "${IMAGE}:${SHORT_SHA}" \
    -

# ---------------------------------------------------------------------------------- CVE hard gate
# 0 CRITICAL findings before any push: inheriting a CVE from a base image is still shipping one, and
# :latest is the only tag a job definition ever resolves, so publishing it is what puts an image into
# active use. The scan runs against the sha tag, which is pushed first so there is something to scan;
# :latest is only published once the gate is green.
echo "pushing ${IMAGE}:${GIT_SHA} for scanning"
docker push "${IMAGE}:${GIT_SHA}"

# The waiter stays advisory on purpose. Its only success acceptor is status COMPLETE, which an
# enhanced-scanning (Inspector) repository never reaches -- there the status is ACTIVE and findings
# arrive continuously -- so making it fatal would block every push on such a repository. The status
# check below is what actually decides, which is why the waiter timing out is only a warning.
echo "waiting for the ECR image scan"
aws ecr wait image-scan-complete \
  --repository-name "${REPOSITORY}" \
  --image-id "imageTag=${GIT_SHA}" \
  --region "${REGION}" \
  || echo "WARNING: could not wait on the scan (scanOnPush may be off); checking findings anyway"

# The status is read BEFORE the counts, and both are required. findingSeverityCounts is a map, so a
# severity with no findings is simply absent from it and the CLI prints "None" -- which reads the
# same whether the scan finished clean or has not produced findings yet. Only the status tells
# those apart, and since :latest is the sole tag a job definition ever resolves, reading the second
# as the first is how a CRITICAL image reaches Batch. Status first because counts fetched afterwards
# can only be more complete than the status claimed, never less.
SCAN_STATUS="$(aws ecr describe-image-scan-findings \
  --repository-name "${REPOSITORY}" \
  --image-id "imageTag=${GIT_SHA}" \
  --region "${REGION}" \
  --query 'imageScanStatus.status' \
  --output text 2>/dev/null || echo "UNKNOWN")"

CRITICAL="$(aws ecr describe-image-scan-findings \
  --repository-name "${REPOSITORY}" \
  --image-id "imageTag=${GIT_SHA}" \
  --region "${REGION}" \
  --query 'imageScanFindings.findingSeverityCounts.CRITICAL' \
  --output text 2>/dev/null || echo "UNKNOWN")"

# COMPLETE is a finished basic scan and ACTIVE is live enhanced scanning; those are the two states
# whose findings are real, so those are the two where an absent CRITICAL key means zero.
if [[ "${SCAN_STATUS}" == "COMPLETE" || "${SCAN_STATUS}" == "ACTIVE" ]]; then
  if [[ "${CRITICAL}" == "None" || -z "${CRITICAL}" ]]; then
    CRITICAL="0"
  fi
fi
echo "scan_status=${SCAN_STATUS} critical_findings=${CRITICAL}"

if [[ ! "${CRITICAL}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: scan status ${SCAN_STATUS}, so the CVE gate cannot be evaluated." >&2
  echo "       A COMPLETE basic scan or an ACTIVE enhanced one is required; under any other" >&2
  echo "       status an absent CRITICAL count means the findings are not in yet, not that" >&2
  echo "       there are none. Enable scanning on the repository, or check findings by hand" >&2
  echo "       before using this image. Refusing to publish :latest on an unknown posture." >&2
  exit 1
fi
if [[ "${CRITICAL}" != "0" ]]; then
  echo "ERROR: ${CRITICAL} CRITICAL finding(s). Not publishing :latest." >&2
  echo "       Bump the base image in cloud/Dockerfile (or the offending package) and rebuild." >&2
  echo "       The sha tag stays pushed for inspection but no job definition will resolve it." >&2
  exit 1
fi

echo "pushing ${IMAGE}:${SHORT_SHA} and ${IMAGE}:latest"
docker push "${IMAGE}:${SHORT_SHA}"
docker push "${IMAGE}:latest"

echo "done. submit with: cloud/submit_job.py --image ${IMAGE}:latest ..."
echo "note: job definitions must resolve :latest, never a sha tag."
