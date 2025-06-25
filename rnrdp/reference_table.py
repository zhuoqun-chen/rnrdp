def get_env_prefix(env_id):
    return env_id.split("-")[0].lower()


def get_control_mode(env_id):
    env_prefix = get_env_prefix(env_id)

    if env_prefix == "stackcube":
        control_mode = "pd_ee_delta_pos"

    elif env_prefix == "turnfaucet" or env_id == "TurnFaucet_COTPC-v0":
        control_mode = "pd_ee_delta_pose"

    elif env_prefix == "pushchair":
        control_mode = "base_pd_joint_vel_arm_pd_joint_vel"

    elif env_prefix == "pusht":
        control_mode = "pd_ee_delta_pose"

    elif env_prefix == "rollball":
        control_mode = "pd_ee_delta_pos"

    else:
        raise ValueError(f"env_id {env_id} not supported")

    return control_mode


def get_demo_path(env_id, obs_mode, control_mode):
    env_prefix = get_env_prefix(env_id)

    if env_prefix == "stackcube":
        demo_path = (
            f"data/v0/rigid_body/{env_id}/trajectory.{obs_mode}.{control_mode}.h5"
        )

    elif env_prefix == "turnfaucet" or env_id == "TurnFaucet_COTPC-v0":
        if obs_mode == "state":
            demo_path = (
                "data/cotpc_demo_data/TurnFaucet/ten_faucets.state.pd_ee_delta_pose.h5"
            )
        elif obs_mode == "rgbd":
            demo_path = "data/cotpc_demo_data/TurnFaucet/ten_faucets.state.pd_ee_delta_pose.rgbd.pd_ee_delta_pose_64.h5"

    elif env_prefix == "pushchair":
        demo_path = (
            f"data/cotpc_demo_data/PushChair/trajectory.{obs_mode}.{control_mode}.h5"
        )

    elif env_prefix == "pusht":
        demo_path = f"data/{env_id}/rl/trajectory.{obs_mode}.{control_mode}.cpu.h5"

    elif env_prefix == "rollball":
        demo_path = f"data/{env_id}/rl/trajectory.{obs_mode}.{control_mode}.cpu.h5"

    return demo_path


def get_script(ms_version, algo, obs_mode):
    assert algo in ["dp", "rnrdp"]
    assert obs_mode in ["state", "rgbd"]

    if ms_version == 2:
        if algo == "dp":
            if obs_mode == "state":
                script = "bc_dp_state_ms2.py"
            elif obs_mode == "rgbd":
                script = "bc_dp_rgbd_ms2.py"
        elif algo == "rnrdp":
            if obs_mode == "state":
                script = "bc_rnrdp_state_ms2.py"
            elif obs_mode == "rgbd":
                script = "bc_rnrdp_rgbd_ms2.py"
    elif ms_version == 3:
        if algo == "dp":
            if obs_mode == "state":
                script = "bc_dp_state_ms3.py"
            elif obs_mode == "rgbd":
                script = "bc_dp_rgbd_ms3.py"
        elif algo == "rnrdp":
            if obs_mode == "state":
                script = "bc_rnrdp_state_ms3.py"
            elif obs_mode == "rgbd":
                script = "bc_rnrdp_rgbd_ms3.py"

    return f"rnrdp/{script}"


def append_obj_ids(cmd, env_id, mode):
    env_prefix = get_env_prefix(env_id)
    if "faucet" not in env_prefix and "chair" not in env_prefix:
        assert mode is None
        return cmd
    assert env_prefix in ["turnfaucet", "pushchair"] or env_id == "TurnFaucet_COTPC-v0"
    assert mode in ["with_object_generalization", "without_object_generalization"]
    if env_prefix == "turnfaucet" or env_id == "TurnFaucet_COTPC-v0":
        if mode == "with_object_generalization":
            obj_ids = [
                5014,
                5037,
                5053,
                5062,
            ]
        elif mode == "without_object_generalization":
            obj_ids = [
                # 5014,
                # 5037,
                # 5053,
                # 5062,
                5002,
                5021,
                5023,
                5028,
                5029,
                5045,
                5047,
                5051,
                5056,
                5063,
            ]
    elif env_prefix == "pushchair":
        if mode == "with_object_generalization":
            obj_ids = [
                3003,
                3013,
                3020,
            ]
        elif mode == "without_object_generalization":
            obj_ids = [
                # 3003,
                # 3013,
                # 3020,
                3022,
                3027,
                3030,
                3070,
                3076,
            ]

    append = f" --obj-ids {' '.join(map(str, obj_ids))} "
    return cmd + append
