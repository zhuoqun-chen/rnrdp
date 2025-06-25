# ruff: noqa: E402

ALGO_NAME = "BC_Diffusion_state_UNet_RNRDP"

import argparse
import datetime
import os
import random
from collections import defaultdict
from distutils.util import strtobool

import gymnasium as gym
import mani_skill.envs  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from einops import rearrange
from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode  # type: ignore
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
    parser.add_argument("--demo-path", type=str, default='data/RollBall-v1/rl/trajectory.state.pd_ee_delta_pos.cpu.h5',
        help="the path of demo dataset (pkl or h5)")
    parser.add_argument("--num-demo-traj", type=int, default=None)
    parser.add_argument("--total-iters", type=int, default=1_000_000, # for easier task, we can train shorter
        help="total timesteps of the experiments")
    parser.add_argument("--batch-size", type=int, default=1024, # 2048 does not further improve
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
    parser.add_argument("--sync-venv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num-dataload-workers", type=int, default=0)
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')
    parser.add_argument("--obj-ids", metavar='N', type=str, nargs='+', default=[])
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
        args.capture_video = False
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


def make_env(env_id, seed, control_mode=None, video_dir=None, other_kwargs={}):
    def thunk():
        env_kwargs = (
            {"model_ids": other_kwargs["obj_ids"]}
            if len(other_kwargs["obj_ids"]) > 0
            else {}
        )
        render_mode = "rgb_array"
        env = gym.make(
            env_id,
            reward_mode="sparse",
            obs_mode="state",
            control_mode=control_mode,
            render_mode=render_mode if video_dir else None,
            **env_kwargs,
        )
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        if video_dir:
            env = RecordEpisode(
                env, output_dir=video_dir, save_trajectory=False, info_on_video=True
            )

        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FrameStack(env, other_kwargs["obs_horizon"])
        env = SeqActionWrapper(env)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, data_path, device, num_traj):
        if data_path[-4:] == ".pkl":
            raise NotImplementedError()
        else:
            from utils.ms2_data import load_demo_dataset

            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)

        for k, v in trajectories.items():
            for i in range(len(v)):
                trajectories[k][i] = torch.Tensor(v[i]).to(device)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in args.control_mode
            or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
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
            assert trajectories["observations"][traj_idx].shape[0] == L + 1
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

        obs_seq = self.trajectories["observations"][traj_idx][
            max(0, start) : start + self.obs_horizon
        ]  # start+self.obs_horizon is at least 1
        act_seq = self.trajectories["actions"][traj_idx][
            start + self.obs_horizon - 1 : end
        ]
        if start < 0:  # pad before the trajectory
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)
            # shouldn't pad actions when start < 0
            pass
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq.shape[0] == self.obs_horizon
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
        assert len(env.single_observation_space.shape) == 2  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
            env.single_action_space.low == -1
        ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        self.args = args
        self.model = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=np.prod(
                env.single_observation_space.shape
            ),  # obs_horizon * obs_dim
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

    def compute_loss(self, obs_seq, action_seq):
        B, To, Do = obs_seq.shape
        b, Tp, Da = action_seq.shape
        action_seq = rearrange(action_seq, "B Tp Da -> Tp B Da")  # *z
        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1)  # (B, To * Do)

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
                obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
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

            B, L_obs_state, C_obs_state = obs_seq.shape

            obs_cond = obs_seq.flatten(start_dim=1)  # (B, To * Do)

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
    VecEnv = (
        gym.vector.SyncVectorEnv
        if args.sync_venv or args.num_eval_envs == 1
        else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
    )
    eval_envs = VecEnv([
        make_env(
            args.env_id,
            args.seed + 1000 + i,
            args.control_mode,
            f"{log_path}/videos" if args.capture_video and i == 0 else None,
            other_kwargs=args.__dict__,
        )
        for i in range(args.num_eval_envs)
    ])
    eval_envs.reset(
        seed=args.seed + 1000
    )  # seed eval_envs here, and no more seeding during evaluation
    envs = eval_envs
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    # dataloader setup
    dataset = SmallDemoDataset_DiffusionPolicy(
        args.demo_path, device, num_traj=args.num_demo_traj
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
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
        # data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        # forward and compute loss
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],  # (B, Tp, Do)
            action_seq=data_batch["actions"],  # (B, Tp, Da)
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
