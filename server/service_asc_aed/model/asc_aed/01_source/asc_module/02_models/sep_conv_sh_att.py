import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite

import sys
sys.path.append('.')
from src.utils import *

class Stem(nn.Module):
  def __init__(self, in_channels, out_channels,*args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    # 3x64x64 -> 16x64x64 -> 32x64x64
    self.module = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
      nn.ReLU(),
      # nn.Conv2d(out_channels//2, out_channels, kernel_size=, stride=2),
      # nn.ReLU(),
      nn.BatchNorm2d(num_features=out_channels),
      nn.Dropout(p=0.2)
    )

  def forward(self, x):
    return self.module(x)
  
class Splitor(nn.Module):
  def __init__(self, att_channels,*args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.att_channels = att_channels
  
  def forward(self, x):
    x_att = x[:, :self.att_channels, :, :]
    x_cnn = x[:, self.att_channels:, :, :]

    return x_att, x_cnn
  
class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.module = nn.Sequential(
      nn.Conv2d(in_channels, out_channels,
                kernel_size=1, stride=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.Conv2d(out_channels, out_channels, 
                kernel_size=3, stride=1, 
                padding='same', groups=out_channels),
      nn.ReLU()
    )

  def forward(self, x):
    return self.module(x)

class Channel2Token(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def forward(self, x):
    #    bs x c x h x w
    # -> bs x c x (h * w)
    bs, c, h, w = x.shape
    return x.reshape(bs, c, -1)

class Token2Channel(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def forward(self, x):
    #    bs x c x (h * w)
    # -> bs x c x h x w
    bs, c, embed_dim = x.shape
    return x.reshape(bs, c, int(embed_dim**0.5), -1)
  

class SingleHeadAttention(nn.Module):  
  def __init__(self, embed_dim, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.layer_norm = nn.LayerNorm(embed_dim)
    self.q_proj = nn.Linear(embed_dim, embed_dim)
    self.k_proj = nn.Linear(embed_dim, embed_dim)
    self.v_proj = nn.Linear(embed_dim, embed_dim)
    self.scaler = embed_dim ** 0.5

  def forward(self, x_seq):
    # x_seq: bs x seq_len x embed_dim
    x_seq = self.layer_norm(x_seq)
    Q = self.q_proj(x_seq)
    K = self.k_proj(x_seq)
    V = self.v_proj(x_seq)

    att_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scaler
    att_weight = F.softmax(att_scores, dim=-1)

    output = torch.matmul(att_weight, V)
    return output
  
class SepConvSHAttBlock(nn.Module):
  def __init__(self, in_channels, ratio, embed_dim, is_identical=False, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.is_identical = is_identical

    att_channels = int(in_channels*ratio)
    cnn_channels = int(in_channels - att_channels)
    self.splitor = Splitor(att_channels)

    self.conv_block = ConvBlock(in_channels=cnn_channels, out_channels=cnn_channels*4)

    self.channel2token = Channel2Token()
    self.att_block = SingleHeadAttention(embed_dim)
    self.token2channel = Token2Channel()
    self.proj = nn.Sequential(
      nn.Conv2d(in_channels=att_channels + cnn_channels*4, out_channels=in_channels,
                kernel_size=1, stride=1),
      nn.Dropout(p=0.2),
      nn.GELU()
    )

  def forward(self, x):
    x_skip = x
    x_att, x_cnn = self.splitor(x)
    

    if(not self.is_identical):
      x_cnn = self.conv_block(x_cnn)

    x_att = self.channel2token(x_att)
    x_att = self.att_block(x_att)
    x_att = self.token2channel(x_att)

    x = torch.concat((x_att, x_cnn), dim=1)
    x = self.proj(x)
    return x_skip + x

class FeedForward(nn.Module):
    def __init__(self, embed_size, expansion_factor=4, dropout=0.2):
        super(FeedForward, self).__init__()
        self.c2t = Channel2Token()
        self.t2c = Token2Channel()

        self.fc1 = nn.Linear(embed_size, embed_size * expansion_factor)
        self.fc2 = nn.Linear(embed_size * expansion_factor, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        # Apply feedforward layers with skip connection
        residual = x
        x = self.c2t(x)
        x = self.fc1(x)       # First linear layer
        x = self.gelu(x)      # ReLU activation
        x = self.dropout(x)   # Dropout for regularization
        x = self.fc2(x)       # Second linear layer
        x = self.gelu(x) 
        return residual + self.t2c(x)   # Add the skip connection (residual connection)

# class Stage(nn.Module):
#   def __init__(self, h, w, c, is_stage,is_stage_3=False, *args, **kwargs) -> None:
#     super().__init__(*args, **kwargs)
#     self.scsha_block1 = SepConvSHAttBlock(in_channels=c, ratio=0.2, embed_dim=h*w) 
#     self.ffn1 = FeedForward(h*w)

#   def forward(self, x):
#     x = self.scsha_block(x)
#     x = self.ffns(x)
#     return x

class DownSample(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.conv_1 = nn.Conv2d(in_channels, in_channels*4,
                            kernel_size=1, stride=1)
    self.conv_2 = nn.Conv2d(in_channels*4, in_channels*4, 
                            kernel_size=3, stride=2, padding=1,
                            groups=in_channels*4)
    self.se = SqueezeExcite(in_channels*4, .25)
    self.conv_3 = nn.Conv2d(in_channels*4, out_channels,
                            kernel_size=1, stride=1)
  
  def forward(self, x):
    x = self.conv_1(x)
    x = self.conv_2(x)
    x = self.se(x)
    x = self.conv_3(x)
    return x 



class SepConvSHAtt(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # c:3 32 48 64 96
    # hw: 64 -> 16 -> 8 -> 4
    self.stem = Stem(in_channels=3, out_channels=32)

    c1 = 32
    c2 = 64
    c3 = 96
    c4 = 128

    self.c2t = Channel2Token()
    self.t2c = Token2Channel()

    # stage1
    h = w = 32
    self.scsha_block1 = SepConvSHAttBlock(in_channels=c1, ratio=0.2, embed_dim=h*w) 
    # self.ffn1 = FeedForward(h*w)
    self.down1 = DownSample(in_channels=c1, out_channels=c2)

    # # stage 2
    h = w = 16
    self.scsha_block2 = SepConvSHAttBlock(in_channels=c2, ratio=0.2, embed_dim=h*w) 
    # self.ffn2 = FeedForward(h*w)
    self.down2 = DownSample(in_channels=c2, out_channels=c3)

    # stage 3
    h = w = 8
    self.scsha_block3_1 = SepConvSHAttBlock(in_channels=c3, ratio=0.2, embed_dim=h*w) 
    self.scsha_block3_2 = SepConvSHAttBlock(in_channels=c3, ratio=0.2, embed_dim=h*w) 
    self.scsha_block3_3 = SepConvSHAttBlock(in_channels=c3, ratio=0.2, embed_dim=h*w) 

    # self.ffn3 = FeedForward(h*w)
    self.down3 = DownSample(in_channels=c3, out_channels=c4)
    
    h = w = 4
    self.scsha_block4 = SepConvSHAttBlock(in_channels=c4, ratio=0.2, embed_dim=h*w) 
    # self.ffn4 = FeedForward(h*w)

    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(in_features=c4, out_features=2)


  def forward(self, x):
    x = self.stem(x)

    x = self.scsha_block1(x)
    # x = self.ffn1(x)
    x = self.down1(x)

    x = self.scsha_block2(x)
    # x = self.ffn2(x)
    x = self.down2(x)

    x = self.scsha_block3_1(x)
    x = self.scsha_block3_2(x)
    x = self.scsha_block3_3(x)
    # x = self.ffn3(x)
    x = self.down3(x)


    x = self.scsha_block4(x)
    # x = self.ffn4(x)
    x = self.avg_pool(x)
    x = torch.squeeze(x)
    x = self.fc(x)



    return x


def main():
  stem = SepConvSHAtt()
  x = torch.randn(8, 3, 64, 64)
  count_params(stem)
  stem(x)
  
if __name__=='__main__':
  main()