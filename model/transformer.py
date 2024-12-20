import torch
import torch.nn as nn

from typing import Callable


def process_inputs(x: torch.Tensor, f: Callable, norm: torch.nn.Module, order: str = "pre") -> torch.Tensor:
    assert order in ["pre", "post"], "Order must be either 'pre' or 'post'"
    if order == "pre":
        xn = norm(x)
        xf = f(xn)
        x = x + xf
    else:
        xf = f(x)
        x = norm(x + xf)
    return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_order: str = "post"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_order = norm_order

    def forward(self, src, src_mask=None):
        # Transpose input from [batch, seq, dim] to [seq, batch, dim]
        src = src.transpose(0, 1)
        
        x = process_inputs(src, 
                          lambda x_: self.self_attn(x_, x_, x_, attn_mask=src_mask)[0], 
                          self.norm1, 
                          order=self.norm_order)
        
        x = process_inputs(x, self.feed_forward, self.norm2, order=self.norm_order)
        
        # Transpose back to [batch, seq, dim]
        return x.transpose(0, 1)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1, norm_order: str = "post", **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, norm_order) for _ in range(n_layers)])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_order: str = "post"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_order = norm_order

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Transpose inputs from [batch, seq, dim] to [seq, batch, dim]
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        
        x = process_inputs(tgt, 
                          lambda x_: self.self_attn(x_, x_, x_, attn_mask=tgt_mask)[0], 
                          self.norm1, 
                          order=self.norm_order)
        
        x = process_inputs(x, 
                          lambda x_: self.cross_attn(x_, memory, memory, attn_mask=memory_mask)[0], 
                          self.norm2, 
                          order=self.norm_order)
        
        x = process_inputs(x, self.feed_forward, self.norm3, order=self.norm_order)
        
        # Transpose back to [batch, seq, dim]
        return x.transpose(0, 1)
    

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1, norm_order: str = "post", **kwargs):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, norm_order) for _ in range(n_layers)])

    def create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        seq_len = tgt.size(1)  # Get the sequence length
        tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).to(tgt.device)
        return tgt_mask

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return output