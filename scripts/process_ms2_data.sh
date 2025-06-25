#!/usr/bin/env bash

# by running each script for ManiSkill2 tasks,
# it generates data path of convention: data/v0/rigid_body/${env_id}/trajectory.${obs_mode}.${target_control_mode}.h5
# pay attension to specific comment notes for some tasks

#################
# ms2::stackcube
#################
export UV_PROJECT_ENVIRONMENT=.venv_ms2
env_id="StackCube-v0"
target_control_mode="pd_ee_delta_pos"

# state
obs_mode="state"
uv run --no-sync python -m mani_skill2.trajectory.replay_trajectory \
    --traj-path ./data/v0/rigid_body/${env_id}/trajectory.h5 \
    --save-traj \
    --obs-mode ${obs_mode} \
    --target-control-mode ${target_control_mode} \
    --num-procs 16 \
    --use-env-state

# NOTE:
# first manually append the following key-value pair to `./data/v0/rigid_body/${env_id}/trajectory.json` env_info.env_kwargs dict:
# "camera_cfgs": {"width": 64, "height": 64}
# otherwise later you will get runtime error: `mat1` and `mat2` shapes cannot be multiplied
#
# rgbd
obs_mode="rgbd"
env_id="StackCube-v0"
target_control_mode="pd_ee_delta_pos"
uv run --no-sync python -m mani_skill2.trajectory.replay_trajectory \
    --traj-path ./data/v0/rigid_body/${env_id}/trajectory.h5 \
    --save-traj \
    --obs-mode ${obs_mode} \
    --target-control-mode ${target_control_mode} \
    --num-procs 16 \
    --use-env-state

##################
# ms2::turnfaucet
##################
env_id="TurnFaucet_COTPC-v0"
target_control_mode="pd_joint_pos"

# state
# use data/cotpc_demo_data/TurnFaucet/ten_faucets.state.pd_ee_delta_pose.h5

# rgbd
# use data/cotpc_demo_data/TurnFaucet/ten_faucets.state.pd_ee_delta_pose.rgbd.h5

#################
# ms2::pushchair
#################
env_id="PushChair-v1"
target_control_mode="base_pd_joint_vel_arm_pd_joint_vel"

# state
# use data/cotpc_demo_data/PushChair/trajectory.state.base_pd_joint_vel_arm_pd_joint_vel.h5

# rgbd
# use data/cotpc_demo_data/PushChair/trajectory.state.base_pd_joint_vel_arm_pd_joint_vel.rgbd.base_pd_joint_vel_arm_pd_joint_vel.h5
