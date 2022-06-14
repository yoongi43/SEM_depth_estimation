import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, k, s, p='same', d=1, g=1, actnorm=True):
        super().__init__()
        net = [nn.Conv2d(ic, oc, k, stride=s, padding=p, dilation=d, groups=g)]
        if actnorm:
            net = net + [
                nn.InstanceNorm2d(num_features=oc),
                nn.PReLU(num_parameters=oc)
            ]
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)
    
    
class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    

class FeedForward(nn.Module):
    def __init__(self, channels, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape=channels),
            nn.Linear(channels, channels * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(dropout_p),
            nn.Linear(channels * expansion_factor, channels, bias=True),
            nn.Dropout(dropout_p)
        )
    
    def forward(self, x):
        return self.net(x) * 0.5
    

class Conformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_linear = FeedForward(channels=channels)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True
        )
        self.conv = nn.Sequential(
            nn.LayerNorm(channels),
            Rearrange("bh w c -> bh c w"),
            nn.Conv1d(in_channels=channels, out_channels=channels *2, kernel_size=1),
            nn.GLU(dim=-2),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding='same', groups=channels),
            Rearrange("bh c w -> bh w c"),
            nn.LayerNorm(channels),
            Swish(),
            Rearrange("bh w c -> bh c w"),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.Dropout(0.1),
            Rearrange("bh c w -> bh w c")            
        )
        self.post_linear = FeedForward(channels=channels)
        self.post_layernorm = nn.LayerNorm(normalized_shape=channels)
        
    def forward(self, x):
        x = x + self.pre_linear(x)
        x = x + self.mhsa.forward(query=x, key=x, value=x, need_weights=False)[0]
        x = x + self.conv(x)
        x = x + self.post_linear(x)
        x = self.post_layernorm(x)
        return x
    
    
class TSConformer(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.wconformer = Conformer(channels=channels)
        self.hconformer = Conformer(channels=channels)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = rearrange(x, "b c h w -> (b h) w c")
        x = x + self.wconformer(x)
        x = rearrange(x, "(b h) w c -> (b w) h c", b= batch_size)
        x = x + self.hconformer(x)
        x = rearrange(x, "(b w) h c -> b c h w", b=batch_size)
        return x
    
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, ic, oc, k, s, p, op, d=1, g=1, actnorm=True) -> None:
        super().__init__()
        net = [nn.ConvTranspose2d(ic, oc, k, s, p, output_padding=op, dilation=d, groups=g)]
        
        if actnorm:
            net = net + [
                nn.InstanceNorm2d(num_features=oc),
                nn.PReLU(num_parameters=oc)
            ]
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)