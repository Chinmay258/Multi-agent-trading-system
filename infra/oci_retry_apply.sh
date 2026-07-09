#!/usr/bin/env bash
# =============================================================================
# Oracle ARM "out of host capacity" retry helper.
# =============================================================================
# Oracle's Always-Free ARM (VM.Standard.A1.Flex) is frequently out of capacity in
# busy regions. This loops `terraform apply` until the instance is created, which
# is the standard community workaround — leave it running (it can take minutes to
# hours). The network resources are created once; only the instance keeps retrying.
#
# Usage (from repo root, with ~/.oci configured and Docker running):
#   bash infra/oci_retry_apply.sh [interval_seconds] [max_attempts]
# Defaults: 180s interval, 0 = unlimited attempts.
# =============================================================================
set -u
INTERVAL="${1:-180}"
MAX="${2:-0}"
cd "$(dirname "$0")/.." || exit 1        # repo root
REPO="$(pwd -W 2>/dev/null || pwd)"       # Windows path for Docker Desktop, else POSIX
export MSYS_NO_PATHCONV=1                  # stop Git Bash mangling container paths

attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "=== $(date '+%H:%M:%S') attempt $attempt ==="
  OUT=$(MSYS_NO_PATHCONV=1 docker run --rm --entrypoint sh \
    -v "$HOME/.oci:/mnt/oci:ro" -v "$REPO/infra:/infra" -w /infra/terraform \
    hashicorp/terraform:1.9 -c '
      mkdir -p /root/.oci && cp /mnt/oci/config /root/.oci/ && cp /mnt/oci/oci_api_key.pem /root/.oci/ && chmod 600 /root/.oci/oci_api_key.pem &&
      terraform init -input=false >/dev/null 2>&1 &&
      terraform apply -auto-approve -input=false' 2>&1)
  if echo "$OUT" | grep -qE "Apply complete"; then
    echo "$OUT" | grep -A20 "Outputs:"
    echo "=== SUCCESS after $attempt attempt(s) ==="
    exit 0
  elif echo "$OUT" | grep -q "Out of host capacity"; then
    echo "  out of capacity; sleeping ${INTERVAL}s"
  else
    echo "  non-capacity error:"; echo "$OUT" | tail -12; exit 1
  fi
  [ "$MAX" != "0" ] && [ "$attempt" -ge "$MAX" ] && { echo "reached max attempts"; exit 2; }
  sleep "$INTERVAL"
done
