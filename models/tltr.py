from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)
    
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

# TLTR
class ATModel(nn.Module):
    def __init__(self, n_layer=32, rep_dim=1280, mode='tl_down_tr_512_1_8'):
        super().__init__()
        self.n_layer = n_layer
        self.rep_dim = rep_dim

        self.num_tatt_head = 1
        self.num_latt_head = 8
        self.time_tr = ResidualAttentionBlock(self.rep_dim, self.num_tatt_head)
        self.layer_tr = ResidualAttentionBlock(self.rep_dim, self.num_latt_head)
        
        if self.rep_dim == 512 :
            self.down_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, 512))
            
    def forward(self, audio_rep, time_resolution=10):
        # time resolution in seconds
        B, num_layer, audio_len, rep_dim = audio_rep.shape[0], audio_rep.shape[1], audio_rep.shape[2], audio_rep.shape[3]
        audio_rep = audio_rep.reshape([B * num_layer, audio_len, rep_dim])  # [B*32, 25, 1280]
        
        if self.rep_dim == 512 :
            audio_rep = self.down_layer(audio_rep.float())
        
        audio_rep = self.time_tr(audio_rep)  # [B*32, 25, 1280]
        audio_rep = audio_rep.reshape([B, num_layer, audio_len, rep_dim]) # [B, 32, 25, 1280]
        audio_rep = audio_rep.permute([0, 2, 1, 3]) # [B, 25, 32, 1280]
        audio_rep = audio_rep.reshape([B * audio_len, num_layer, rep_dim]) # [B*25, 32, 1280]
        audio_rep = self.layer_tr(audio_rep)  # [B*25, 32, 1280]
        audio_rep = torch.mean(audio_rep, dim=1)  # [B*25, 1280]
        audio_rep = audio_rep.reshape([B, audio_len, rep_dim])
        return audio_rep
