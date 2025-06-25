#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

uv venv .venv_ms2
source .venv_ms2/bin/activate

export UV_PROJECT_ENVIRONMENT=.venv_ms2
uv pip sync requirements-ms2.txt --preview --torch-backend cu121