"""Neural network building blocks.

Placeholder for the modules migrated from cs336: RoPE, multi-head
self-attention, transformer blocks, RMSNorm, SwiGLU, embeddings, etc.
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

# functions

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values # get the max value along the specified dim
    exp_x = torch.exp(x - x_max) # subtract the max value for numerical stability
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True) # sum the exponentials along the specified dim
    return exp_x / sum_exp_x # divide by the sum

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    x_max = inputs.max(dim=-1, keepdim=True).values # get the max value along the last dim
    log_sum_exp = x_max.squeeze(-1) + torch.log(torch.exp(inputs - x_max).sum(dim=-1)) # compute log-sum-exp
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # gather the logits corresponding to the targets
    return (log_sum_exp - target_logits).mean() # compute the cross-entropy loss

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor| None = None) -> torch.Tensor:
    # Q @ K^T / sqrt(d_k)
    d_k = K.shape[-1]
    attn_scores = torch.einsum("...nd, ...md -> ...nm", Q, K) / math.sqrt(d_k) # compute the attention scores
    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf")) # apply the mask
    weights = softmax(attn_scores, dim=-1) # compute the attention weights
    return torch.einsum("...nm, ...md -> ...nd", weights, V) # compute the adjusted values

# classes

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = (2.0 / (in_features + out_features)) ** 0.5
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i, oi -> ...o", x, self.weight)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=0.0,
                std=1.0,
                a=-3.0,
                b=3.0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # index into vocab and get the row
        return self.weight[x]

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm_x = x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt() * self.weight
        return norm_x.to(in_dtype)


class RoPE(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        if (d_k % 2) != 0:
            raise ValueError("d_k must be even for RoPE.")
        double_idx = torch.arange(0, self.d_k, 2)
        ## build positional vecor m
        m = torch.arange(0, self.max_seq_len, 1)
        theta_i = 1 / (self.theta ** (double_idx / self.d_k))
        ## outer prodcut -> angle table
        angle_table = torch.outer(m, theta_i)
        self.register_buffer("cos_table", torch.cos(angle_table))
        self.register_buffer("sin_table", torch.sin(angle_table))


    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[positions]
        sin = self.sin_table[positions]
        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd),
                                    dim=-1)
        return x_rotated.reshape(*x.shape)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads, rope: RoPE | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # set up out WQ, WK, WV, WO matrices
        self.WQ = Linear(d_model, d_model)
        self.WK = Linear(d_model, d_model)
        self.WV = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        # forward passes
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        # separate our heads into their own slices
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # apply rope to Q & K 
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # apply single head attn (for each head)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        # unslice
        attn_output = rearrange(attn_output, "... h seq d -> ... seq (h d)")

        # output forward pass
        return self.output_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE | None = None) -> None:
        super().__init__()
        # x -> norm -> attn  + x -> norm -> FF + x
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        # x -> norm -> attn + x
        x = x + self.attn(self.ln1(x), token_positions)
        # x -> norm -> FF + x
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers, num_heads, d_ff, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_model // num_heads
        self.rope = RoPE(self.d_k, context_length, theta)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, self.rope)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        token_positions = torch.arange(x.shape[-2], device=x.device)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
