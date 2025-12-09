import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Dict, Any, List


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones(dims)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None: mid_size = out_size
        self.linear1 = nn.Linear(frequency_embedding_size, mid_size)
        self.linear2 = nn.Linear(mid_size, out_size)
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t):
        half = self.frequency_embedding_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(0, half, dtype=mx.float32) / half)
        args = (t[:, None].astype(mx.float32) * freqs[None, :])
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.frequency_embedding_size % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=1)
        return self.linear2(nn.silu(self.linear1(embedding)))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, dim: int, nheads: int, rope_theta: float = 256.0, eps: float = 1e-5):
        super().__init__()
        self.nheads = nheads
        self.head_dim = dim // nheads
        self.scale = self.head_dim ** -0.5
        self.rope_theta = rope_theta

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

    def __call__(self, x, mask=None, positions=None):
        B, L, D = x.shape
        q = self.to_q(x).reshape(B, L, self.nheads, self.head_dim)
        k = self.to_k(x).reshape(B, L, self.nheads, self.head_dim)
        v = self.to_v(x).reshape(B, L, self.nheads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if positions is not None:
            split1 = 32
            split2 = 32 + 48
            q_splits = [q[..., :split1], q[..., split1:split2], q[..., split2:]]
            k_splits = [k[..., :split1], k[..., split1:split2], k[..., split2:]]
            dims_list = [32, 48, 48]

            for i in range(3):
                pos = positions[..., i].astype(mx.int32)

                # Interleaved RoPE
                def manual_rope(x, dims, offset, base=256.0, scale=1.0):
                    half = dims // 2
                    freqs = mx.exp(-mx.log(base) * mx.arange(0, half, dtype=mx.float32) / half)
                    args = offset[..., None].astype(mx.float32) * freqs[None, :] * scale
                    cos = mx.cos(args)
                    sin = mx.sin(args)
                    x1 = x[..., 0::2]
                    x2 = x[..., 1::2]
                    cos = cos[..., None, :]
                    sin = sin[..., None, :]
                    out1 = x1 * cos - x2 * sin
                    out2 = x1 * sin + x2 * cos
                    return mx.stack([out1, out2], axis=-1).reshape(x.shape)

                q_splits[i] = manual_rope(q_splits[i], dims_list[i], pos, base=self.rope_theta)
                k_splits[i] = manual_rope(k_splits[i], dims_list[i], pos, base=self.rope_theta)

            q = mx.concatenate(q_splits, axis=-1)
            k = mx.concatenate(k_splits, axis=-1)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = q @ k.transpose(0, 1, 3, 2) * self.scale

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.to_out(out)


class ZImageTransformerBlock(nn.Module):
    def __init__(self, config: Dict[str, Any], layer_id: int, modulation: bool = True):
        super().__init__()
        dim = config['dim']
        nheads = config['nheads']
        norm_eps = config.get('norm_eps', 1e-6)
        qk_norm_eps = 1e-5
        rope_theta = config.get('rope_theta', 256.0)

        self.modulation = modulation
        self.attention = Attention(dim, nheads, rope_theta=rope_theta, eps=qk_norm_eps)
        self.feed_forward = FeedForward(dim, int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, norm_eps)
        self.ffn_norm1 = RMSNorm(dim, norm_eps)
        self.attention_norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)

        if modulation:
            self.adaLN_modulation = nn.Linear(256, 4 * dim, bias=True)

    def __call__(self, x, mask, positions, adaln_input=None):
        if self.modulation:
            chunks = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(chunks, 4, axis=-1)
            scale_msa, gate_msa = scale_msa[..., None, :], gate_msa[..., None, :]
            scale_mlp, gate_mlp = scale_mlp[..., None, :], gate_mlp[..., None, :]

            norm_x = self.attention_norm1(x) * (1 + scale_msa)
            attn_out = self.attention(norm_x, mask, positions)
            x = x + mx.tanh(gate_msa) * self.attention_norm2(attn_out)

            norm_ffn = self.ffn_norm1(x) * (1 + scale_mlp)
            ffn_out = self.feed_forward(norm_ffn)
            x = x + mx.tanh(gate_mlp) * self.ffn_norm2(ffn_out)
        else:
            attn_out = self.attention(self.attention_norm1(x), mask, positions)
            x = x + self.attention_norm2(attn_out)
            ffn_out = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(ffn_out)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, dim, bias=True)
        )

    def __call__(self, x, c):
        modulation = self.adaLN_modulation.layers[0](c)
        scale = self.adaLN_modulation.layers[1](modulation)
        x = self.norm_final(x) * (1 + scale[:, None, :])
        return self.linear(x)


class ZImageTransformerMLX(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        dim = config['dim']
        eps = config.get('norm_eps', 1e-6)
        self.t_scale = config.get('t_scale', 1000.0)

        self.t_embedder = TimestepEmbedder(256, mid_size=1024)
        embed_dim = config['in_channels'] * (2 * 2)
        self.x_embedder = nn.Linear(embed_dim, dim, bias=True)

        self.cap_embedder = nn.Sequential(
            RMSNorm(config['cap_feat_dim'], eps),
            nn.Linear(config['cap_feat_dim'], dim, bias=True)
        )

        self.final_layer = FinalLayer(dim, embed_dim)
        self.x_pad_token = mx.zeros((1, dim))
        self.cap_pad_token = mx.zeros((1, dim))

        n_refiner = config['n_refiner_layers']
        n_layers = config['n_layers']

        self.noise_refiner = [ZImageTransformerBlock(config, i, True) for i in range(n_refiner)]
        self.context_refiner = [ZImageTransformerBlock(config, i, False) for i in range(n_refiner)]
        self.layers = [ZImageTransformerBlock(config, i, True) for i in range(n_layers)]

    def __call__(self, x, t, cap_feats, x_pos, cap_pos, x_mask=None, cap_mask=None):
        t = t * self.t_scale
        temb = self.t_embedder(t)
        x = self.x_embedder(x)

        if x_mask is not None:
            x = mx.where(x_mask[..., None], self.x_pad_token, x)

        cap_feats = self.cap_embedder.layers[0](cap_feats)
        cap_feats = self.cap_embedder.layers[1](cap_feats)

        if cap_mask is not None:
            cap_feats = mx.where(cap_mask[..., None], self.cap_pad_token, cap_feats)

        x_attn_mask = None
        cap_attn_mask = None

        for layer in self.noise_refiner: x = layer(x, x_attn_mask, x_pos, temb)
        for layer in self.context_refiner: cap_feats = layer(cap_feats, cap_attn_mask, cap_pos, None)

        unified_mask = None

        img_len = x.shape[1]
        unified = mx.concatenate([x, cap_feats], axis=1)
        unified_pos = mx.concatenate([x_pos, cap_pos], axis=1)

        for layer in self.layers: unified = layer(unified, unified_mask, unified_pos, temb)

        x_out = unified[:, :img_len, :]
        x_out = self.final_layer(x_out, temb)
        return x_out