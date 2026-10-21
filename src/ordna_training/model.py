"""Torch-only port of ORDNA triplets, preserving upstream state-dict names.

Original: ordna/src/ORDNA/models/{representation_module,triplets}.py.
No original files modified. einops operations are expressed with reshape/permute.
NUM_TOKENS is int17 (upstream defines 17. as a float).
The objective deliberately matches upstream code, including Frobenius batch norms.
"""
import torch
from torch import nn


class Representation(nn.Module):
    def __init__(self, token_emb_dim=8, seq_len=150, repr_dim=256):
        super().__init__()
        if repr_dim % 4:
            raise ValueError('repr_dim must be divisible by four attention heads')
        self.token_emb_layer=nn.Embedding(17,token_emb_dim)
        self.conv_1=nn.Conv1d(token_emb_dim,2*token_emb_dim,5,padding=2)
        self.conv_2=nn.Conv1d(2*token_emb_dim,4*token_emb_dim,5,dilation=5,padding=10)
        self.seq_mlp=nn.Sequential(nn.Linear(8*token_emb_dim*seq_len,2048),nn.ReLU(),nn.Linear(2048,repr_dim))
        self.sample_emb_layer=nn.Embedding(1,repr_dim)
        for index in range(1,5):
            setattr(self,f'attention_{index}',nn.MultiheadAttention(repr_dim,4,batch_first=True))
        self.layer_norm=nn.LayerNorm(repr_dim)

    def forward(self,x):
        if x.ndim!=4 or x.shape[2]!=2 or x.dtype!=torch.long:
            raise ValueError('Expected LongTensor [batch, reads, 2, sequence_length]')
        emb=self.token_emb_layer(x)
        b,n,_,length,t=emb.shape
        directions=emb.permute(2,0,1,4,3).reshape(2,b*n,t,length)
        forward=self.conv_2(self.conv_1(directions[0]))
        reverse=self.conv_2(self.conv_1(directions[1]))
        # Upstream: rearrange([forward,reverse], 'dir BN T L -> BN (dir T L)')
        combined=torch.stack([forward,reverse]).permute(1,0,2,3).reshape(b*n,-1)
        reads=self.seq_mlp(combined).reshape(b,n,-1)
        token=self.sample_emb_layer(torch.zeros(b,device=x.device,dtype=torch.long)).unsqueeze(1)
        z=torch.cat([token,reads],dim=1)
        for index in range(1,5):
            attn=getattr(self,f'attention_{index}')
            output,_=attn(z,z,z,need_weights=False)
            z=self.layer_norm(output+z)
        return z[:,0]


class TripletEncoder(nn.Module):
    def __init__(self,token_emb_dim=8,seq_len=150,sample_repr_dim=256,sample_emb_dim=64):
        super().__init__()
        self.repr_module=Representation(token_emb_dim,seq_len,sample_repr_dim)
        self.linear=nn.Linear(sample_repr_dim,sample_emb_dim)

    def forward(self,x):
        representation=self.repr_module(x)
        return self.linear(representation),representation


def reference_objective(anchor,positive,negative,margin=.3,l2_max_weight=.1):
    triplet=nn.functional.triplet_margin_loss(anchor,positive,negative,margin=margin)
    eps=.001
    weight=torch.clamp((margin-triplet.detach().float()+eps)/(margin+eps),min=0)*l2_max_weight
    # Keep upstream implementation: norm over entire [batch,dim] tensor, no /3.
    norm=sum(torch.linalg.norm(z) for z in [anchor,positive,negative])
    return triplet+weight*norm,{'triplet':triplet.detach(),'norm':norm.detach(),'norm_weight':weight}
