#!/usr/bin/env bash

export UV_PROJECT_ENVIRONMENT=.venv_ms3 # change to .venv_docker_ms3 if inside docker container

# ms3::rollball::state
cd rnrdp || exit 1
uv run --no-sync python bc_rnrdp_state_ms3.py \
    --exp-name test_train \
    --env-id RollBall-v1 \
    --demo-path ../data/RollBall-v1/rl/trajectory.state.pd_ee_delta_pos.cpu.h5 \
    --control-mode pd_ee_delta_pos \
    --obs-horizon 2 \
    --act-horizon 1 \
    --pred-horizon 4 \
    --num-diffusion-iters 4 \
    --prediction-type epsilon \
    --rnrdp-sample-t mix \
    --eval-noise-scheduler ddpm \
    --rnrdp-eval-method laddering
