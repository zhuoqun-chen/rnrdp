import math
from typing import Union

import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x horizon * cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """

        # the cond is now fused_cond, time_embedding + image_state feats
        assert len(cond.shape) == 3, (
            "Expecting a 2D time embedding + image_state feats sequence block (no batch size dim) for my new designed CR Block"
        )

        # 1. get x
        out = self.blocks[0](x)

        # 2. get a, b
        embed = self.cond_encoder(cond)

        # now t is not 1d but 2d
        embed = rearrange(embed, "b t c -> b c t")
        scale, bias = embed.chunk(2, dim=1)

        # 3. a*x + b
        out = scale * out + bias

        out = self.blocks[1](out)

        # 4. residual connection
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ])
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                    ),
                    ConditionalResidualBlock1D(
                        dim_out,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                    ),
                    Downsample1d(dim_out) if not is_last else nn.Identity(),
                    Downsample1d(cond_dim)
                    if not is_last
                    else nn.Identity(),  # down-t is a must-have when t is 2d
                ])
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_out * 2,
                        dim_in,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                    ),
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_in,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                    ),
                    Upsample1d(dim_in) if not is_last else nn.Identity(),
                    Upsample1d(cond_dim)
                    if not is_last
                    else nn.Identity(),  # up-t is must-have
                ])
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B, rnrdp_token_len, rnrdp_buffer_size) (New! shape is 3 dims)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        t_cond = self.diffusion_step_encoder(timesteps)
        t_cond = t_cond.unsqueeze(
            0
        )  # (pred_horizon, dim_t_embed) -> (1, pred_horizon, dim_t_embed)
        t_cond = t_cond.expand(
            global_cond.shape[0], -1, -1
        )  # -> (b, pred_horizon, dim_t_embed)

        assert len(global_cond.shape) == 2, (
            "\n====\nExpect condition shape like (b, c)\n====\n"
        )
        global_cond = rearrange(
            global_cond, "b c -> b 1 c"
        )  # <=> global_cond.unsqueeze(1)
        # manual broadcast, feat_cond is just a ref to global_cond in another view
        # to ensure feat_cond is "saw by each time frame along pred-horizon"
        feat_cond = global_cond.expand(-1, t_cond.shape[1], -1)
        fused_cond = torch.cat(
            [t_cond, feat_cond], dim=2
        )  # -> (b, pred_horizon, dim_t_embed + c)
        x = sample
        h = []
        # down modules
        for idx, (resnet, resnet2, downsample, down_t) in enumerate(self.down_modules):
            # CR1
            x = resnet(x, fused_cond)
            # CR2
            x = resnet2(x, fused_cond)
            h.append(x)
            # DOWN
            x = downsample(x)
            # down fused_cond
            fused_cond = rearrange(fused_cond, "b t c -> b c t")
            fused_cond = down_t(fused_cond)
            fused_cond = rearrange(fused_cond, "b c t -> b t c")

        # write explicitly for mid modules as well
        for idx, (resnet, resnet2) in enumerate(self.mid_modules):
            # CR1
            x = resnet(x, fused_cond)
            # CR2
            x = resnet2(x, fused_cond)

        # up modules
        for idx, (resnet, resnet2, upsample, up_t) in enumerate(self.up_modules):
            # CR1
            pop = h.pop()
            x = torch.cat((x, pop), dim=1)
            x = resnet(x, fused_cond)
            # CR2
            x = resnet2(x, fused_cond)
            # UP
            x = upsample(x)
            # up fused_cond
            fused_cond = rearrange(fused_cond, "b t c -> b c t")
            fused_cond = up_t(fused_cond)
            fused_cond = rearrange(fused_cond, "b c t -> b t c")

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x
