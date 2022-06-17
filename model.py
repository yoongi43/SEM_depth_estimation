import torch
import torch.nn as nn
from modules import Conformer, ConvBlock, TSConformer, ConvTransposeBlock
from einops import rearrange



class DepthEstimation(nn.Module):
    def __init__(self, ic=1, n_conformer=4):
        super().__init__()
            
        self.encoder = nn.Sequential(
            ConvBlock(ic=ic, oc=4, k=3, s=1, p='same'),
            ConvBlock(ic=4, oc=16, k=3, s=1, p='same'),
            ConvBlock(ic=16, oc=32, k=3, s=1, p='same'),
            ConvBlock(ic=32, oc=64, k=3, s=1, p='same')
            # ConvBlock(ic=16, oc=64, k=3, s=2, p=1)

        )
        # self.encoder = nn.Sequential(
        #     ConvBlock(ic=ic, oc=4, k=3, s=1, p='same'),
        #     ConvBlock(ic=4, oc=16, k=3, s=1, p='same'),
        #     # ConvBlock(ic=16, oc=32, k=3, s=1, p='same'),
        #     # ConvBlock(ic=32, oc=64, k=3, s=1, p='same')
        #     ConvBlock(ic=16, oc=64, k=3, s=2, p=1)

        # )
        self.tsconformers = nn.Sequential(
            *[TSConformer(channels=64) for _ in range(n_conformer)]
        )
        self.decoder = nn.Sequential(
            # ConvTransposeBlock(ic=64, oc=16, k=3, s=2, p=1, op=(1, 0)),
            ConvBlock(ic=64, oc=32, k=3, s=1, p='same'),
            ConvBlock(ic=32, oc=16, k=3, s=1, p='same'),
            ConvBlock(ic=16, oc=4, k=3, s=1, p='same'),
            ConvBlock(ic=4, oc=1, k=3, s=1, p='same')
        )
        # self.decoder = nn.Sequential(
        #     ConvTransposeBlock(ic=64, oc=16, k=3, s=2, p=1, op=(1, 0)),
        #     # ConvBlock(ic=64, oc=16, k=3, s=1, p='same'),
        #     # ConvBlock(ic=64, oc=32, k=3, s=1, p='same'),
        #     # ConvBlock(ic=32, oc=16, k=3, s=1, p='same'),
        #     ConvBlock(ic=16, oc=4, k=3, s=1, p='same'),
        #     ConvBlock(ic=4, oc=1, k=3, s=1, p='same')
        # )
        
    def forward(self, x):
        x = rearrange(x, 'b h w -> b 1 h w')  # if cv2.read!
        x = self.encoder(x)
        x = self.tsconformers(x)
        x = self.decoder(x)
        x = rearrange(x, 'b 1 h w -> b h w')  # if cv2.read!!
        return x
    

class DepthEstimation2(nn.Module):
    def __init__(self, ic=1, n_conformer=4):
        super().__init__()
            
        self.enc1 = ConvBlock(ic=ic, oc=4, k=5, s=1, p='same')
        self.enc2 = ConvBlock(ic=4, oc=16, k=5, s=1, p='same')
        self.enc3 = ConvBlock(ic=16, oc=32, k=3, s=1, p='same')
        self.enc4 = ConvBlock(ic=32, oc=64, k=3, s=1, p='same')
        # self.enc4 = ConvBlock(ic=32, oc=64, k=3, s=2, p=1)
        
        self.transformers = nn.Sequential(
            *[TSConformer(channels=64) for _ in range(n_conformer)]
        )
        
        # self.dec4 = ConvTransposeBlock(ic=64, oc=32, k=3, s=2, p=1, op=(1, 0))
        self.dec4 = ConvBlock(ic=64, oc=32, k=3, s=1, p='same')
        self.dec3 = ConvBlock(ic=32, oc=16, k=3, s=1, p='same')
        self.dec2 = ConvBlock(ic=16, oc=4, k=5, s=1, p='same')
        self.dec1 = ConvBlock(ic=4, oc=1, k=5, s=1, p='same')
        
    def forward(self, x):
        # x = rearrange(x, 'b h w -> b 1 h w')  # if cv2.imread
        out_e1 = self.enc1(x)
        out_e2 = self.enc2(out_e1)
        out_e3 = self.enc3(out_e2)
        out_e4 = self.enc4(out_e3)
        
        out_conformer = self.transformers(out_e4)
        
        out_d4 = self.dec4(out_conformer) + out_e3
        out_d3 = self.dec3(out_d4) + out_e2
        out_d2 = self.dec2(out_d3) + out_e1
        out_d1 = self.dec1(out_d2)
        # out_d1 = rearrange(out_d1, 'b 1 h w -> b h w')
        return out_d1
    
    
class SimpleConv(nn.Module):
    """
    For Test
    """
    def __init__(self, ic=1, n_conformer=4):
        super().__init__()
            
        self.encoder = nn.Sequential(
            ConvBlock(ic=ic, oc=4, k=3, s=1, p='same'),
            ConvBlock(ic=4, oc=16, k=3, s=1, p='same'),
            ConvBlock(ic=16, oc=64, k=3, s=2, p=1)
        )
        # self.tsconformers = nn.Sequential(
        #     *[TSConformer(channels=64) for _ in range(n_conformer)]
        # )
        self.decoder = nn.Sequential(
            ConvTransposeBlock(ic=64, oc=16, k=3, s=2, p=1, op=(1, 0)),
            ConvBlock(ic=16, oc=4, k=3, s=1, p='same'),
            ConvBlock(ic=4, oc=1, k=3, s=1, p='same')
        )
        
    def forward(self, x):
        # x = rearrange(x, 'b h w -> b 1 h w')
        x = self.encoder(x)
        # x = self.tsconformers(x)
        x = self.decoder(x)
        # x = rearrange(x, 'b 1 h w -> b h w')
        return x
        
        
if __name__=="__main__":
    from torchinfo import summary
    de = DepthEstimation(ic=1)
    summary(de, input_size=(128,1, 66,45))