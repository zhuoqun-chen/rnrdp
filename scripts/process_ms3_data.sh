#!/usr/bin/env bash
# shellcheck disable=all

#############
# ms3::pusht
#############
env_id="PushT-v1"
target_control_mode="pd_ee_delta_pose"

# state
# use data/PushT-v1/rl/trajectory.state.pd_ee_delta_pose.cpu.h5

# rgbd
# use data/PushT-v1/rl/trajectory.rgbd.pd_ee_delta_pose.cpu.h5

################
# ms3::rollball
################
env_id="Rollball-v1"
target_control_mode="pd_ee_delta_pos"

# state
# use data/Rollball-v1/rl/trajectory.state.pd_ee_delta_pos.cpu.h5

# rgbd
# use data/Rollball-v1/rl/trajectory.rgbd.pd_ee_delta_pos.cpu.h5
