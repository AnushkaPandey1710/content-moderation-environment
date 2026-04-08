#!/usr/bin/env bash

set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

PING_URL="${PING_URL%/}"

echo "========================================"
echo "  Content Moderation Validator (Final)"
echo "========================================"

PASS=0

pass() { echo "✅ PASSED -- $1"; PASS=$((PASS+1)); }
fail() { echo "❌ FAILED -- $1"; exit 1; }
log()  { echo "[INFO] $1"; }

# -----------------------------
# Dependency check
# -----------------------------
command -v curl >/dev/null || fail "curl is required"

if ! command -v jq &>/dev/null; then
  echo "jq not found. Please install jq before running this script."
  exit 1
fi

# -----------------------------
# Retry helper (HF cold start safe)
# -----------------------------
retry() {
  local retries=10
  local delay=5
  local count=0

  until "$@"; do
    count=$((count+1))
    if [ $count -ge $retries ]; then
      return 1
    fi
    log "Retry $count/$retries..."
    sleep $delay
  done
}

# -----------------------------
# Step 1: RESET
# -----------------------------
echo ""
echo "Step 1/5: Checking /reset..."

RESET_RESPONSE=$(retry curl -s -X POST "$PING_URL/reset" -H "Content-Type: application/json" -d '{}') \
  || fail "/reset failed"

SESSION_ID=$(echo "$RESET_RESPONSE" | jq -r '.session_id // empty')

if [ -z "$SESSION_ID" ]; then
  echo "$RESET_RESPONSE"
  fail "/reset missing session_id"
fi

pass "/reset working"
log "Session ID: $SESSION_ID"

# -----------------------------
# Validate observation fields
# -----------------------------
OBS_CHECK=$(echo "$RESET_RESPONSE" | jq '
  (.state.toxicity != null) or (.state.observation.toxicity != null)
')

if [ "$OBS_CHECK" = "true" ]; then
  pass "observation structure valid"
else
  echo "$RESET_RESPONSE"
  fail "missing observation fields (toxicity etc.)"
fi

# -----------------------------
# Step 2: STATE 
# -----------------------------
echo ""
echo "Step 2/5: Checking /state..."

STATE_RESPONSE=$(retry curl -s -X POST "$PING_URL/state" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION_ID\"}") \
  || fail "/state failed"

STATE_OK=$(echo "$STATE_RESPONSE" | jq '
  (.state.toxicity != null)
')

if [ "$STATE_OK" = "true" ]; then
  pass "/state working"
else
  echo "$STATE_RESPONSE"
  fail "/state invalid"
fi

# -----------------------------
# Step 3: STEP
# -----------------------------
echo ""
echo "Step 3/5: Checking /step..."

STEP_RESPONSE=$(retry curl -s -X POST "$PING_URL/step" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"$SESSION_ID\", \"action\":0, \"confidence\":0.9}") \
  || fail "/step failed"

STEP_OK=$(echo "$STEP_RESPONSE" | jq '
  (.state != null) and 
  ((.state.reward != null) or (.reward != null))
')

if [ "$STEP_OK" = "true" ]; then
  pass "/step working"
else
  echo "$STEP_RESPONSE"
  fail "/step invalid"
fi

# -----------------------------
# Step 4: SCHEMA
# -----------------------------
echo ""
echo "Step 4/5: Checking /schema..."

SCHEMA_RESPONSE=$(retry curl -s "$PING_URL/schema") \
  || fail "/schema failed"

SCHEMA_OK=$(echo "$SCHEMA_RESPONSE" | jq '
  (.observation_space != null) and 
  (.action_space != null)
')

if [ "$SCHEMA_OK" = "true" ]; then
  pass "/schema valid"
else
  echo "$SCHEMA_RESPONSE"
  fail "/schema invalid"
fi

# -----------------------------
# Step 5: Docker + OpenEnv
# -----------------------------
echo ""
echo "Step 5/5: Docker + OpenEnv..."

command -v docker >/dev/null || fail "Docker not installed"

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "Dockerfile not found"
fi

log "Building Docker image..."
docker build "$DOCKER_CONTEXT" > /dev/null 2>&1 || fail "Docker build failed"
pass "Docker build successful"

command -v openenv >/dev/null || fail "openenv not installed (pip install openenv-core)"

log "Running openenv validate..."
if (cd "$REPO_DIR" && openenv validate); then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
fi

# -----------------------------
# FINAL RESULT
# -----------------------------
echo ""
echo "========================================"
echo "🎉 ALL CHECKS PASSED ($PASS/5)"
echo "✅ READY FOR SUBMISSION"
echo "========================================"