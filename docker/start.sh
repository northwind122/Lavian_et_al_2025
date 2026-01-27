#!/usr/bin/env bash
set -euo pipefail

TOKEN="${JUPYTER_TOKEN:-}"
if [ -z "$TOKEN" ]; then
  if command -v openssl >/dev/null 2>&1; then
    TOKEN="$(openssl rand -hex 16)"
  else
    TOKEN="$(python - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"
  fi
fi

echo "Jupyter token: ${TOKEN}"

exec micromamba run -n vis_nav \
  jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.token="${TOKEN}" \
  --ServerApp.password=""
