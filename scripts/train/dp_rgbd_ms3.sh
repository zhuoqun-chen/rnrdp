#!/usr/bin/env bash

export UV_PROJECT_ENVIRONMENT=.venv_ms3 # change to .venv_docker_ms3 if inside docker container

# ms3::rollball::rgbd
cd rnrdp || exit 1
uv run --no-sync python bc_dp_rgbd_ms3.py \
    --exp-name test_train \
    --env-id RollBall-v1 \
    --demo-path ../data/RollBall-v1/rl/trajectory.rgbd.pd_ee_delta_pos.cpu.h5 \
    --control-mode pd_ee_delta_pos \
    --obs-horizon 2 \
    --act-horizon 8 \
    --pred-horizon 16 \
    --num-diffusion-iters 100
