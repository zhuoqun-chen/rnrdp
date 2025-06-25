#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

[ -L .venv_docker_ms2 ] || ln -s /workspace/uv/ms2/.venv .venv_docker_ms2
[ -L .venv_docker_ms3 ] || ln -s /workspace/uv/ms3/.venv .venv_docker_ms3

if [[ $# -eq 0 ]]; then
    exec "/bin/zsh"
else
    exec "$@"
fi