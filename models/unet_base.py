import torch
import torch.nn as nn


def time_embedding(time_steps, t_dim):
    r"""
    Generate sinusoidal time embeddings.
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert t_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=t_dim // 2, dtype=torch.float32, device=time_steps.device) / (t_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, t_dim
    t_embedding = time_steps[:, None].repeat(1, t_dim // 2) / factor
    t_embedding = torch.cat([torch.sin(t_embedding), torch.cos(t_embedding)], dim=-1)
    return t_embedding


class DownBlock(nn.Module):
    r"""
     Downsampling block for U-Net.
    Sequence of following block
    1. Resnet block with time embedding
    2. Attention block
    3. Downsample using 2x2 average pooling
    """
    def __init__(self, in_channels, out_channels, t_dim,
                 down_sample=True, n_heads=4, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.down_sample = down_sample

        # Define ResNet layers
        self.res_blocks_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for i in range(n_layers)
            ]
        )

        # Time embedding projection layers
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_dim, out_channels)
            )
            for _ in range(n_layers)
        ])

        # Second set of ResNet layers
        self.res_blocks_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                )
                for _ in range(n_layers)
            ]
        )

        # Attention layers
        self.attention_norms = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(n_layers)])
        
        self.attentions = nn.ModuleList([nn.MultiheadAttention(out_channels, n_heads, batch_first=True) for _ in range(n_layers)])
        
        # Skip connection layers
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(n_layers)
            ]
        )

        # Downsampling layer
        self.downsample = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x, t_emb):
        out = x
        for i in range(self.n_layers):
            
            # Resnet block 
            resnet_in = out
            out = self.res_blocks_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.res_blocks_second[i](out)
            out = out + self.residual_input_conv[i](resnet_in)
            
            # Attention block 
            batch_size, channels, h, w = out.shape
            attn_in  = out.reshape(batch_size, channels, h * w)
            attn_in  = self.attention_norms[i](attn_in )
            attn_in  = attn_in.transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_in , attn_in , attn_in )
            attn_out = attn_out.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + attn_out
            
        out = self.downsample(out)
        return out


class MidBlock(nn.Module):
    r"""
    Middle block with with alternating ResNet and Attention layers.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    def __init__(self, in_channels, out_channels, t_dim, n_heads=4, n_layers=1):
        super().__init__()
        self.n_layers = n_layers

        self.res_blocks_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(n_layers+1)
            ]
        )

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_dim, out_channels)
            )
            for _ in range(n_layers + 1)
        ])

        self.res_blocks_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(n_layers+1)
            ]
        )
        
        self.attention_norms = nn.ModuleList([nn.GroupNorm(8, out_channels) for _ in range(n_layers)])
        
        self.attentions = nn.ModuleList([nn.MultiheadAttention(out_channels, n_heads, batch_first=True) for _ in range(n_layers)])
       
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(n_layers+1)
            ]
        )
    
    def forward(self, x, t_emb):
        out = x
        
        # First resnet block
        resnet_in = out
        out = self.res_blocks_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.res_blocks_second[0](out)
        out = out + self.residual_input_conv[0](resnet_in)
        
        for i in range(self.n_layers):
            
            # Attention Block
            batch_size, channels, h, w = out.shape
            attn_in  = out.reshape(batch_size, channels, h * w)
            attn_in  = self.attention_norms[i](attn_in )
            attn_in  = attn_in.transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_in , attn_in , attn_in )
            attn_out = attn_out.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + attn_out
            
            # Resnet Block
            resnet_in = out
            out = self.res_blocks_first[i+1](out)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.res_blocks_second[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_in)
        
        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    def __init__(self, in_channels, out_channels, t_dim, up_sample=True, n_heads=4, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.up_sample = up_sample

        self.res_blocks_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(n_layers)
            ]
        )

        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_dim, out_channels)
            )
            for _ in range(n_layers)
        ])

        self.res_blocks_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(n_layers)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(8, out_channels)
                for _ in range(n_layers)
            ]
        )
        
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, n_heads, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(n_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_n_layerslayers):
            resnet_in = out
            out = self.res_blocks_first[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.res_blocks_second[i](out)
            out = out + self.residual_input_conv[i](resnet_in)
            
            batch_size, channels, h, w = out.shape
            attn_in  = out.reshape(batch_size, channels, h * w)
            attn_in  = self.attention_norms[i](attn_in )
            attn_in  = attn_in .transpose(1, 2)
            attn_out, _ = self.attentions[i](attn_in , attn_in , attn_in )
            attn_out = attn_out.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + attn_out

        return out


class Unet(nn.Module):
    r"""
    Unet model comprising Down blocks, Midblocks and Uplocks
    """
    def __init__(self, model_config):
        super().__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        # Validate channel configuration
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        
        # Initial time embedding projection
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_dim, self.t_dim),
            nn.SiLU(),
            nn.Linear(self.t_dim, self.t_dim)
        )

        # Initial input convolution
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        # Downsampling layers        
        self.down_blocks = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.down_blocks.append(DownBlock(self.down_channels[i], self.down_channels[i+1], self.t_dim,
                                        down_sample=self.down_sample[i], n_layers=self.num_down_layers))
        
        # Middle layers
        self.mid_blocks = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mid_blocks.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_dim,
                                      n_layers=self.num_mid_layers))
        
        # Upsampling layers
        self.up_blocks = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.up_blocks.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i-1] if i != 0 else 16,
                                    self.t_dim, up_sample=self.down_sample[i], n_layers=self.num_up_layers))
        
        # Output normalization and convolution
        self.norm_out = nn.GroupNorm(8, 16)
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]

        # Input convolution (B x C x H x W)
        output = self.conv_in(x)
        # B x C1 x H x W
        
        # Time embedding
        # t_emb -> B x t_dim
        t_emb = time_embedding(torch.as_tensor(t).long(), self.t_dim)
        t_emb = self.t_proj(t_emb)
        
        # Store outputs from down blocks
        down_blocks_out = []
        
        for idx, down in enumerate(self.down_blocks):
            down_blocks_out.append(output)
            output = down(output, t_emb)
        # down_blocks_out  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # output B x C4 x H/4 x W/4

        # Middle blocks   
        for mid in self.mid_blocks:
            output = mid(output, t_emb)
        # output B x C3 x H/4 x W/4
        
        # Up blocks
        for up in self.up_blocks:
            down_output = down_blocks_out.pop()
            output = up(output, down_output, t_emb)
            # output [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
       
        # Output layer
        output = self.norm_out(output)
        output = nn.SiLU()(output)
        output = self.conv_out(output)
        # output B x C x H x W
        return output
