# ruff: noqa: E402

ALGO_NAME = "BC_Diffusion_rgbd_UNet_RNRDP"

import argparse
import datetime
import os
import random
from collections import defaultdict, deque
from distutils.util import strtobool
from functools import partial

import gymnasium as gym
import mani_skill.envs  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from einops import rearrange
from gymnasium import spaces
from gymnasium.wrappers.frame_stack import FrameStack as gymFrameStack
from mani_skill.utils import common
from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode
from mani_skill.utils.wrappers.frame_stack import FrameStack
from nets.cnn.plain_conv import PlainConv, PlainConv_MS1
from nets.rnrdp.conditional_unet1d import ConditionalUnet1D
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from utils.common import to_tensor
from utils.profiling import NonOverlappingTimeProfiler
from utils.sampler import IterationBasedBatchSampler
from utils.torch_utils import worker_init_fn


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=None,
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--wandb-project-name", type=str, default="rnrdp",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="RollBall-v1",
        help="the id of the environment")
    parser.add_argument("--demo-path", type=str, default='data/RollBall-v1/rl/trajectory.rgbd.pd_ee_delta_pos.cpu.h5',
        help="the path of demo dataset (pkl or h5)")
    parser.add_argument("--num-demo-traj", type=int, default=None)
    parser.add_argument("--total-iters", type=int, default=1_000_000, # for easier task, we can train shorter
        help="total timesteps of the experiments")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the replay memory")

    # Diffusion Policy specific arguments
    parser.add_argument("--lr", type=float, default=1e-4) # 1e-4 is a safe choice
    parser.add_argument("--obs-horizon", type=int, default=2)
    # Seems not very important in ManiSkill, 1, 2, 4 work well
    parser.add_argument("--act-horizon", type=int, default=1)
    # Seems not very important in ManiSkill, 4, 8, 15 work well
    parser.add_argument("--pred-horizon", type=int, default=64)
    # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=64) # not very important
    parser.add_argument("--unet-dims", metavar='N', type=int, nargs='+', default=[64, 128, 256]) # ~4.5M params
    parser.add_argument("--n-groups", type=int, default=8)
    # Jiayuan told me it is better to let each group has at least 8 channels; it seems 4 and 8 are similar

    parser.add_argument("--output-dir", type=str, default='output')
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--num-eval-envs", type=int, default=10) # NOTE: should not be too large, otherwise bias to short episodes
    parser.add_argument("--num-dataload-workers", type=int, default=0) # TODO: to tune
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--obj-ids", metavar='N', type=str, nargs='+', default=[])
    parser.add_argument("--image-size", type=str, default='64x64',
        help="the size of observation image, HxW")
    parser.add_argument("--state-keys", metavar='N', type=str, nargs='+', default=['tcp_pose'])
    parser.add_argument("--no-state", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--random-shift", type=int, default=0)

    # RNR-DP specific
    parser.add_argument("--rnrdp-sample-t", type=str, default="mix", choices=["linear", "rand", "mix"])
    parser.add_argument("--rnrdp-eval-method", type=str, default="laddering", choices=["laddering", "pure-noise"])
    parser.add_argument("--eval-noise-scheduler", type=str, default="ddpm", choices=["ddim", "ddpm"])
    parser.add_argument("--num-diffusion-iters", type=int, default=100)
    parser.add_argument("--prediction-type", type=str, default="epsilon", choices=["epsilon", "sample"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args, unknown = parser.parse_known_args()
    if args.debug or args.exp_name == "debug":
        args.track = False
        args.eval_freq = 1
        args.exp_name = "dummy_debug"
        args.num_eval_episodes = 1
        args.capture_video = True
    args.algo_name = ALGO_NAME
    args.script = __file__
    args.num_eval_envs = min(args.num_eval_envs, args.num_eval_episodes)
    assert args.num_eval_episodes % args.num_eval_envs == 0
    if args.num_eval_envs == 1:
        args.sync_venv = True
    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, 'Control mode mismatched'
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    if args.image_size:
        args.image_size = tuple(map(int, args.image_size.split('x')))
    # fmt: on
    return args


class SeqActionWrapper(gym.Wrapper):
    def step(self, action_seq):
        rew_sum = 0
        for action in action_seq:
            obs, rew, terminated, truncated, info = self.env.step(action)
            rew_sum += rew
            if terminated or truncated:
                break
        return obs, rew_sum, terminated, truncated, info


class MS2_RGBDObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.get_wrapper_attr("obs_mode") == "rgbd"
        self.state_obs_extractor = self.build_state_obs_extractor(env.unwrapped.spec.id)
        self.observation_space = self.build_obs_space(
            env, np.float32, self.state_obs_extractor
        )
        self.concat_fn = partial(np.concatenate, axis=-1)
        self.transpose_fn = partial(
            np.transpose, axes=(2, 0, 1)
        )  # channel first, (H, W, C) -> (C, H, W)

    def observation(self, obs):
        return self.convert_obs(
            obs, self.concat_fn, self.transpose_fn, self.state_obs_extractor
        )

    @staticmethod
    def build_state_obs_extractor(env_id):
        env_name = env_id.split("-")[0]
        if env_name in ["TurnFaucet_COTPC", "StackCube"]:
            return lambda obs: list(obs["extra"].values())
        elif env_name == "PushChair":
            return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())
        else:
            raise NotImplementedError(
                "Please tune state obs by --state-keys for this env"
            )

    @staticmethod
    def build_obs_space(env, depth_dtype, state_obs_extractor):
        # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
        obs_space = env.observation_space
        # state_dim = 0

        # # Original
        # for k in ['agent', 'extra']:
        #     state_dim += sum([v.shape[0] for v in nested_space_dict_to_flat_space_dict(obs_space[k]).values()])

        # # For StackCube
        # state_dim = sum([v.shape[0] for v in nested_space_dict_to_flat_space_dict(obs_space['extra']).values()])

        # For PushChair
        # for k1 in ['agent', 'extra']:
        #     for k2, v in nested_space_dict_to_flat_space_dict(obs_space[k1]).items():
        #         if k2 in args.state_keys:
        #             state_dim += v.shape[0]

        # Unified version
        state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

        single_img_space = next(iter(env.observation_space["image"].values()))
        h, w, _ = single_img_space["rgb"].shape
        n_images = len(env.observation_space["image"])

        return spaces.Dict({
            "state": spaces.Box(
                -float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32
            ),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
            "depth": spaces.Box(
                -float("inf"),
                float("inf"),
                shape=(n_images, h, w),
                dtype=depth_dtype,
            ),
        })

    @staticmethod
    def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor):
        img_dict = obs["image"]
        new_img_dict = {
            key: transpose_fn(
                concat_fn([v[key] for v in img_dict.values()])
            )  # (C, H, W) or (B, C, H, W)
            for key in ["rgb", "depth"]
        }
        # if isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
        #     new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

        # NOTE: qvel is very harmful in many cases
        # state = np.hstack([
        #     state_dict_to_vector(obs['agent']),
        #     state_dict_to_vector(obs['extra']),
        # ])
        # state = state_dict_to_vector(obs['extra'])

        ######################
        # Debug code:
        ######################
        # x = obs['agent']['base_pose']
        # x = obs['agent']['qpos']
        # x = obs['agent']['qvel']
        # x = obs['extra']['tcp_pose']
        # if x.ndim == 2:
        #     print('demo:\t', end='')
        #     print(np.mean(x, axis=0))
        # else:
        #     print('env:\t', end='')
        #     print(x)
        # import pdb; pdb.set_trace()
        # state = np.hstack([
        #     # obs['agent']['qpos'],
        #     obs['agent']['qvel'],
        #     # obs['agent']['base_pose'],
        #     obs['extra']['tcp_pose'],
        # ])

        ########################
        # For PushChair
        ########################
        # state_to_stack = []
        # for k1 in ['agent', 'extra']:
        #     for k2, v in obs[k1].items():
        #         if k2 in args.state_keys:
        #             state_to_stack.append(v)
        # try:
        #     state = np.hstack(state_to_stack)
        # except: # dirty fix for concat trajectory of states
        #     state = np.column_stack(state_to_stack)
        # state = state.astype(np.float32)

        # Unified version
        states_to_stack = state_obs_extractor(obs)
        for j in range(len(states_to_stack)):
            # FIXED: in turn-simple task, states_to_stack[1] is float, not np type
            if isinstance(states_to_stack[j], float):
                states_to_stack[j] = np.array([states_to_stack[j]], dtype=np.float32)
            if states_to_stack[j].dtype == np.float64:
                states_to_stack[j] = states_to_stack[j].astype(np.float32)
        try:
            state = np.hstack(states_to_stack)
        except Exception:  # dirty fix for concat trajectory of states
            state = np.column_stack(states_to_stack)
        if state.dtype == np.float64:
            for x in states_to_stack:
                print(x.shape, x.dtype)
            import pdb

            pdb.set_trace()

        out_dict = {
            "state": state,
            "rgb": new_img_dict["rgb"],
            "depth": new_img_dict["depth"],
        }
        return out_dict


def to_torch_recursive(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, dict):
        return {key: to_torch_recursive(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_torch_recursive(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_torch_recursive(item) for item in data)
    else:
        return data


class MS3_RGBDObsWrapper(gym.ObservationWrapper):
    """
    Flatten obs dict into a dict with 3 keys: state, rgb, depth
    """

    def __init__(self, env):
        self.base_env = env.unwrapped
        super().__init__(env)
        self.transforms = T.Compose([
            T.Resize((64, 64), antialias=True),  # (128, 128) -> (64, 64)
        ])
        self.transpose_fn = partial(torch.permute, dims=(0, 3, 1, 2))
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    @staticmethod
    def build_state_obs_extractor(env_id):
        env_name = env_id.split("-")[0]
        if env_name in ["TurnFaucet_COTPC", "StackCube"]:
            return lambda obs: list(obs["extra"].values())
        elif env_name == "PushChair":
            return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())
        elif env_name == "PushT":  # NOTE: currently hardcode
            return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())
        else:
            raise NotImplementedError(
                "Please tune state obs by --state-keys for this env"
            )

    @staticmethod
    def convert_obs(observation, transform_fn, transpose_fn, state_obs_extractor):
        # FIXED: AttributeError: 'numpy.ndarray' object has no attribute 'permute', if order is MS3_RGBDObsWrapper<CPUWrapper>, then reset return numpy array by design of CPUWrapper
        observation = to_torch_recursive(
            observation
        )  # NOTE: because this is also used during loading numpy dataset
        # debug(observation)
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]

        # from devtools import debug
        # debug(sensor_data)
        # exit()

        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            # print(cam_data["rgb"].shape)
            # print(cam_data["depth"].shape)
            resized_rgb = transform_fn(
                # cam_data["rgb"].permute(0, 3, 1, 2) # unwrapped: Box(0, 255, (1, 128, 128, 3), uint8)
                transpose_fn(cam_data["rgb"])
            )  # (1, 128, 128, 3) -> (1, 3, 64, 64)
            images_rgb.append(resized_rgb)

            # has to np.float32 to have num_eval_episode=100 to have AsyncVecEnv to work
            depth = (cam_data["depth"].to(torch.float32) / 1024).to(
                torch.float32
            )  # unwrapped: Box(-32768/1024, 32767/1024, (1, 128, 128, 1), int16)
            resized_depth = transform_fn(
                # depth.permute(0, 3, 1, 2)
                transpose_fn(depth)
            )  # (1, 128, 128, 1) -> (1, 1, 64, 64)
            images_depth.append(resized_depth)

        rgb = torch.stack(images_rgb, dim=1)  # (1, num_cams, C, 64, 64), uint8
        depth = torch.stack(images_depth, dim=1)  # (1, num_cams, C, 64, 64), float16

        # rgb = torch.cat(images_rgb, dim=1) # (1, num_cams * C, 64, 64), uint8
        # depth = torch.cat(images_depth, dim=1) # (1, num_cams * C, 64, 64), float16

        # flatten the rest of the data which should just be state data

        # observation = common.flatten_state_dict(observation, use_torch=True)
        observation = state_obs_extractor(observation)

        ret = dict()
        ret["state"] = observation
        ret["rgb"] = rgb
        ret["depth"] = depth

        return ret

    def observation(self, observation):
        return self.convert_obs(
            observation,
            self.transforms,
            self.transpose_fn,
            partial(common.flatten_state_dict, use_torch=True),
        )


class DictFrameStack(FrameStack):
    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
        """
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)


class NewDictFrameStack(FrameStack):
    def observation(self, observation):
        stacked_obs = {}
        for k in self.observation_space.keys():
            stacked_obs[k] = [frame[k] for frame in self.frames]
            stacked_obs[k] = torch.stack(stacked_obs[k]).transpose(0, 1)
        return stacked_obs


class NewNumpyDictFrameStack(gymFrameStack):
    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, num_stack=num_stack, lz4_compress=lz4_compress
        )
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        # new_observation_space = gym.spaces.Dict()
        # # for k, v in self.observation_space.items():

        # for k, v in env.base_env.observation_space.items():
        #     low = np.repeat(v.low[np.newaxis, ...], num_stack, axis=0)
        #     high = np.repeat(v.high[np.newaxis, ...], num_stack, axis=0)
        #     new_observation_space[k] = gym.spaces.Box(low=low, high=high, dtype=v.dtype)
        # self.observation_space = new_observation_space

    def observation(self, observation):
        stacked_obs = {}
        for k in self.observation_space.keys():
            stacked_obs[k] = [frame[k] for frame in self.frames]
            # debug(stacked_obs[k])
            # stacked_obs[k] = torch.stack(stacked_obs[k]).transpose(0, 1)
            # numpy alternative

            stacked_obs[k] = np.stack(stacked_obs[k], axis=0)
            # stacked_obs[k] = rearrange(stacked_obs[k], "To t ... -> t To ...")

        return stacked_obs


def make_vec_env(
    env_id,
    num_envs,
    seed,
    control_mode=None,
    image_size=None,
    video_dir=None,
    other_kwargs={},
    gym_vec_env=True,
):
    assert gym_vec_env or video_dir is None, (
        "Saving video is only supported for gym vec env"
    )
    cam_cfg = {"width": image_size[0], "height": image_size[1]} if image_size else None
    wrappers = [
        gym.wrappers.ClipAction,
        partial(NewDictFrameStack, num_stack=other_kwargs["obs_horizon"]),
        SeqActionWrapper,
    ]
    env_kwargs = (
        {"model_ids": other_kwargs["obj_ids"]}
        if len(other_kwargs["obj_ids"]) > 0
        else {}
    )
    if video_dir:
        render_mode = "rgb_array"
    else:
        render_mode = None
    if gym_vec_env:

        def make_single_env(_seed):
            def thunk():
                env = gym.make(
                    env_id,
                    reward_mode="sparse",
                    obs_mode="rgbd",
                    control_mode=control_mode,
                    render_mode=render_mode,
                    # camera_cfgs=cam_cfg, # NOTE: deprecated in MS3
                    **env_kwargs,
                )

                env = MS3_RGBDObsWrapper(env)
                env = gym.wrappers.ClipAction(env)
                env = NewDictFrameStack(env, num_stack=other_kwargs["obs_horizon"])
                env = SeqActionWrapper(env)

                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                env = RecordEpisode(
                    env, output_dir=video_dir, save_trajectory=False, info_on_video=True
                )

                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        # must use AsyncVectorEnv, so that the renderers will be in different processes
        # but in BC, we can use SyncVectorEnv if we only create one renderer, since we do not have training envs
        VecEnv = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        envs = VecEnv([make_single_env(seed + i) for i in range(num_envs)])
    else:
        raise NotImplementedError()

    return envs


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, data_path, obs_process_fn, obs_space, num_traj):
        if data_path[-4:] == ".pkl":
            raise NotImplementedError()
        else:
            from utils.ms3_data import load_demo_dataset

            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print("Raw trajectory loaded, start to pre-process the observations...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, obs_space
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())

        # Pre-process the actions
        min_len_actions = np.inf
        max_len_actions = -np.inf
        for i in range(len(trajectories["actions"])):
            if trajectories["actions"][i].shape[0] < min_len_actions:
                min_len_actions = trajectories["actions"][i].shape[0]
            if trajectories["actions"][i].shape[0] > max_len_actions:
                max_len_actions = trajectories["actions"][i].shape[0]
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i])
        print(
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )
        print(
            "Min/Max length of actions among all 973 trajs:",
            min_len_actions,
            max_len_actions,
        )
        # breakpoint()
        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in args.control_mode
            or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            self.pad_action_arm = torch.zeros((
                trajectories["actions"][0].shape[1] - 1,
            ))
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            args.obs_horizon,
            args.pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + obs_horizon + pred_horizon - 1)
                for start in range(-pad_before, L - obs_horizon + 1)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][
            start + self.obs_horizon - 1 : end
        ]
        if start < 0:  # pad before the trajectory
            # act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
            # shouldn't pad actions when start < 0
            pass
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        # assert len(env.single_observation_space["state"].shape) == 2 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
            env.single_action_space.low == -1
        ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        num_stack, _, C, H, W = env.single_observation_space["rgb"].shape
        if args.no_state:
            obs_state_dim = 0
        self.args = args

        visual_feature_dim = 256
        CNN_class = PlainConv
        self.visual_encoder = CNN_class(
            in_channels=int(C / 3 * 4), out_dim=visual_feature_dim
        )
        self.model = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = args.num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type=args.prediction_type,
        )

        if args.random_shift > 0:
            from utils.torch_utils import RandomShiftsAug

            self.aug = RandomShiftsAug(args.random_shift)

    def encode_obs(self, obs_seq, eval_mode):
        # NOTE: now obs_seq["rgb"]: (B, To, k, C, H, W)
        # print(obs_seq["state"].shape) # still (B, To, Do)
        obs_seq["rgb"] = rearrange(obs_seq["rgb"], "B To k C H W -> B To (k C) H W")
        obs_seq["depth"] = rearrange(obs_seq["depth"], "B To k C H W -> B To (k C) H W")

        rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
        depth = obs_seq["depth"].float()  # (B, obs_horizon, 1*k, H, W)
        img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            rgb.shape[0], self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        # *z avoid separate eval code not finding args
        if self.args.no_state:
            return visual_feature.flatten(start_dim=1)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B, To, Do = obs_seq["state"].shape
        b, Tp, Da = action_seq.shape
        action_seq = rearrange(action_seq, "B Tp Da -> Tp B Da")  # *z
        # observation as FiLM conditioning
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)  # (B, To * Do)

        # sample noise to add to actions
        noise = torch.randn_like(action_seq, device=device)

        # sample a diffusion iteration for each data point

        assert self.args.pred_horizon == self.num_diffusion_iters
        if self.args.rnrdp_sample_t == "linear":
            timesteps = torch.tensor(
                list(range(0, self.noise_scheduler.config.num_train_timesteps)),
                device=device,  # *z needs Tp noise-levels
            ).long()

        if self.args.rnrdp_sample_t == "rand":
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (self.noise_scheduler.config.num_train_timesteps,),
                device=device,  # *z needs Tp noise-levels
            ).long()
        if self.args.rnrdp_sample_t == "mix":
            if torch.rand(1) >= 0.6:
                # linear
                timesteps = torch.tensor(
                    list(range(0, self.noise_scheduler.config.num_train_timesteps)),
                    device=device,  # *z needs Tp noise-levels
                ).long()
            else:
                # rand
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (self.noise_scheduler.config.num_train_timesteps,),
                    device=device,  # *z needs Tp noise-levels
                ).long()
        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        pred = self.model(
            rearrange(noisy_action_seq, "Tp B Da -> B Tp Da"),
            timesteps,
            global_cond=obs_cond,
        )
        pred = rearrange(pred, "B Tp Da -> Tp B Da")

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = action_seq

        loss = F.mse_loss(pred, target)
        return loss


def collect_episode_info(infos, result=None):
    if result is None:
        result = defaultdict(list)
    if "final_info" in infos:  # infos is a dict
        indices = np.where(infos["_final_info"])[
            0
        ]  # not all envs are done at the same time
        for i in indices:
            info = infos["final_info"][i]  # info is also a dict
            ep = info["episode"]
            print(
                f"global_step={cur_iter}, ep_return={ep['return']:.2f}, ep_len={ep['episode_len']}, success={ep['success_once']}"
            )
            result["return"].append(ep["return"])
            result["len"].append(ep["episode_len"])
            result["success"].append(ep["success_once"])
    return result


def evaluate_rnrdp(n, agent, eval_envs, device, method, receding_horizon):
    # eval_scheduler = agent.noise_scheduler
    if args.eval_noise_scheduler == "ddim":
        eval_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type=args.prediction_type,
        )
        eval_scheduler.set_timesteps(num_inference_steps=args.pred_horizon)
    elif args.eval_noise_scheduler == "ddpm":
        eval_scheduler = DDPMScheduler(
            num_train_timesteps=args.pred_horizon,  # reduce inference diffusion step as Tongzhou suggested
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type=args.prediction_type,
        )
    agent.eval()
    print("======= Evaluation Starts =========")
    result = defaultdict(list)
    obs, info = eval_envs.reset()  # don't seed here

    t = torch.tensor(list(range(0, args.pred_horizon)), device=device).long()
    buffer = torch.randn(
        args.pred_horizon, args.num_eval_envs, agent.act_dim, device=device
    )  # (Tp, B, Da)
    if method != "pure-noise":
        if method == "laddering":
            idle_cnt = args.pred_horizon - 1
        with torch.no_grad():
            for i in range(idle_cnt):
                obs_seq = to_tensor(obs, device)
                B, L_obs_state, C_obs_state = obs_seq["state"].shape
                obs_cond = agent.encode_obs(
                    obs_seq, eval_mode=True
                )  # (B, obs_horizon * obs_dim)
                model_output = agent.model(
                    rearrange(buffer, "Tp B Da -> B Tp Da"),
                    t,
                    global_cond=obs_cond,
                )  # (B, Tp, Da)
                model_output = rearrange(model_output, "B Tp Da -> Tp B Da")
                _buffer = torch.zeros_like(buffer)  # expect buffer (Tp, B, Da)
                for idx, k in enumerate(reversed(eval_scheduler.timesteps)):
                    noisy_frame = eval_scheduler.step(
                        model_output=model_output[idx],
                        timestep=k,
                        sample=buffer[idx],
                    ).prev_sample
                    _buffer[idx] = noisy_frame
                action = _buffer[0].unsqueeze(0)
                buffer = torch.cat(
                    [_buffer[1:], torch.randn_like(action, device=device)], dim=0
                )
    receding_cnt = 0
    while len(result["return"]) < n:
        with torch.no_grad():
            obs_seq = to_tensor(obs, device)

            B, L_obs_state, C_obs_state = obs_seq["state"].shape

            obs_cond = agent.encode_obs(obs_seq, eval_mode=True)  # (B, To * Do)

            model_output = agent.model(
                rearrange(buffer, "Tp B Da -> B Tp Da"),
                t,
                global_cond=obs_cond,
            )  # (B, Tp, Da)
            model_output = rearrange(model_output, "B Tp Da -> Tp B Da")
            _buffer = torch.zeros_like(buffer)  # expect buffer (Tp, B, Da)
            for idx, k in enumerate(reversed(eval_scheduler.timesteps)):
                noisy_frame = eval_scheduler.step(
                    model_output=model_output[idx],
                    timestep=k,
                    sample=buffer[idx],
                ).prev_sample
                _buffer[idx] = noisy_frame
            action = _buffer[0].unsqueeze(0)
            buffer = torch.cat(
                [_buffer[1:], torch.randn_like(action, device=device)], dim=0
            )

        next_obs, rew, terminated, truncated, info = eval_envs.step(
            rearrange(action, "1 num_env dim_a -> num_env 1 dim_a").cpu().numpy()
        )
        collect_episode_info(info, result)
        receding_cnt += 1
        if not (receding_cnt % receding_horizon):
            obs = next_obs
            receding_cnt = 0
    print("======= Evaluation Ends =========")
    agent.train()
    return result


def save_ckpt(tag):
    os.makedirs(f"{log_path}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
        },
        f"{log_path}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = parse_args()
    # breakpoint()

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    tag = "{:s}_{:d}".format(now, args.seed)
    if args.exp_name:
        tag += "_" + args.exp_name
    log_name = os.path.join(args.env_id, ALGO_NAME, tag)
    log_path = os.path.join(args.output_dir, log_name)

    if args.track:
        import wandb

        run_id = wandb.util.generate_id() if not args.wandb_id else args.wandb_id
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=log_name.replace(os.path.sep, "__"),
            monitor_gym=True,
            save_code=True,
            id=run_id,
        )
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import json

    with open(f"{log_path}/args.json", "w") as f:
        args_dict = vars(args)
        args_dict["run_id"] = run_id if args.track else None
        json.dump(args_dict, f, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    eval_envs = make_vec_env(
        args.env_id,
        args.num_eval_envs,
        args.seed + 1000,
        args.control_mode,
        args.image_size,
        video_dir=f"{log_path}/videos" if args.capture_video else None,
        other_kwargs=args.__dict__,
    )
    obs, _ = eval_envs.reset(
        seed=args.seed + 1000
    )  # seed eval_envs here, and no more seeding during evaluation
    envs = eval_envs
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    # dataloader setup
    obs_process_fn = partial(
        MS3_RGBDObsWrapper.convert_obs,
        transform_fn=T.Compose([
            T.Resize((64, 64), antialias=True),  # (128, 128) -> (64, 64)
        ]),
        transpose_fn=partial(torch.permute, dims=(0, 3, 1, 2)),
        state_obs_extractor=partial(common.flatten_state_dict, use_torch=True),
    )
    tmp_env = gym.make(args.env_id, obs_mode="rgbd")
    orignal_obs_space = tmp_env.observation_space
    # breakpoint()
    tmp_env.close()
    dataset = SmallDemoDataset_DiffusionPolicy(
        args.demo_path, obs_process_fn, orignal_obs_space, args.num_demo_traj
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        pin_memory=True,
        persistent_workers=(args.num_dataload_workers > 0),
    )

    # agent setup
    agent = Agent(envs, args).to(device)
    if args.ckpt:
        agent.eval()
        checkpoint = torch.load(args.ckpt)
        agent.load_state_dict(checkpoint["ema_agent"])
        cur_iter = 0
        result = evaluate_rnrdp(
            args.num_eval_episodes,
            agent,
            eval_envs,
            device,
            args.rnrdp_eval_method,
            args.act_horizon,
        )
        sr = np.mean(result["success"])
        print(f"Offline eval {args.num_eval_episodes} SR:\n{sr}\n")
        import sys

        if args.track:
            wandb.finish()
        sys.exit(0)
    optimizer = optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    best_success_rate = -1

    timer = NonOverlappingTimeProfiler()

    for iteration, data_batch in enumerate(train_dataloader):
        cur_iter = iteration + 1
        timer.end("data")

        # # copy data from cpu to gpu
        obs_batch_dict = data_batch["observations"]
        obs_batch_dict = {
            k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()
        }
        act_batch = data_batch["actions"].cuda(non_blocking=True)

        # forward and compute loss
        total_loss = agent.compute_loss(
            obs_seq=obs_batch_dict,  # obs_batch_dict['state'] is (B, Tp, Do)
            action_seq=act_batch,  # (B, Tp, Da)
        )
        timer.end("forward")

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timer.end("backward")

        # update Exponential Moving Average of the model weights
        ema.step(agent.parameters())
        timer.end("EMA")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if cur_iter % args.log_freq == 0:
            print(cur_iter, total_loss.item())
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            timer.dump_to_writer(writer, cur_iter)

        # Evaluation
        if cur_iter % args.eval_freq == 0:
            ema.copy_to(ema_agent.parameters())
            result = evaluate_rnrdp(
                args.num_eval_episodes,
                ema_agent,
                eval_envs,
                device,
                args.rnrdp_eval_method,
                args.act_horizon,
            )
            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), cur_iter)
                writer.add_scalar(
                    f"eval/reced{args.act_horizon}/{k}", np.mean(v), cur_iter
                )
                writer.add_scalar(
                    f"eval/reced{args.act_horizon}/{k}/{args.rnrdp_eval_method}",
                    np.mean(v),
                    cur_iter,
                )

            timer.end("eval")
            sr = np.mean(result["success"])
            if sr > best_success_rate:
                best_success_rate = sr
                save_ckpt("best_eval_success_rate")
                print(f"### Update best success rate: {sr:.4f}")

        # Checkpoint
        if args.save_freq and cur_iter % args.save_freq == 0:
            save_ckpt(str(cur_iter))

    envs.close()
    writer.close()
