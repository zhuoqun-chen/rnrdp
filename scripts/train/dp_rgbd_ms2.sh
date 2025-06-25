#!/usr/bin/env bash

export UV_PROJECT_ENVIRONMENT=.venv_ms2 # change to .venv_docker_ms2 if inside docker container

# ms2::stackcube::rgbd
cd rnrdp || exit 1
uv run --no-sync python bc_dp_rgbd_ms2.py \
    --exp-name test_train \
    --env-id StackCube-v0 \
    --demo-path ../data/v0/rigid_body/StackCube-v0/trajectory.rgbd.pd_ee_delta_pos.h5 \
    --control-mode pd_ee_delta_pos \
    --obs-horizon 2 \
    --act-horizon 8 \
    --pred-horizon 16 \
    --num-diffusion-iters 100
