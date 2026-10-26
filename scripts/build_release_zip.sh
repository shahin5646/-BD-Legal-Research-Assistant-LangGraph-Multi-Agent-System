#!/usr/bin/env bash
# build_release_zip.sh — Create a clean release zip excluding sensitive & ephemeral files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ZIP="${ROOT_DIR}/research_assistant_release.zip"

echo "Creating clean release archive at: ${OUTPUT_ZIP}"

cd "${ROOT_DIR}"

# Exclude sensitive keys, caches, git, and virtualenvs
zip -r "${OUTPUT_ZIP}" . \
  -x "*.env*" \
     ".langgraph_api/*" \
     ".git/*" \
     ".venv/*" \
     "__pycache__/*" \
     "*/__pycache__/*" \
     ".pytest_cache/*" \
     "*.pyc" \
     "*.pyo" \
     "build/*" \
     "dist/*" \
     "*.egg-info/*" \
     ".DS_Store"

echo "Release package created successfully: ${OUTPUT_ZIP}"
