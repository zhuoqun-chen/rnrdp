#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

uv venv .venv_ms3
source .venv_ms3/bin/activate

export UV_PROJECT_ENVIRONMENT=.venv_ms3
uv pip sync requirements-ms3.txt --preview --torch-backend cu121