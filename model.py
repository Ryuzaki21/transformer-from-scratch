import torch
import torch.nn as nn
import math


# positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            position = torch.tensor(pos, dtype=torch.float32)
            for i in range((d_model + 1) // 2):
                div_term = torch.tensor(
                    10000 ** (2 * i / d_model),
                    dtype=torch.float32
                )

                positional_encoding[pos][2 * i] = torch.sin(position / div_term)

                if 2 * i + 1 < d_model:
                    positional_encoding[pos][2 * i + 1] = torch.cos(position / div_term)

        self.register_buffer(
            "positional_encoding",
            positional_encoding.unsqueeze(0)
        )

    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1)]
        return self.dropout(x)


# layernorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):  # (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# feedforward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):  # (batch_size, seq_len, d_model)
        return self.net(x)


# multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        assert d_model % heads == 0

        self.heads = heads
        self.d_k = d_model // heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = q

        batch_size, query_len, _ = q.size()
        key_len = k.size(1)

        Q = self.w_q(q).view(batch_size, query_len, self.heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, key_len, self.heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, key_len, self.heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = attention @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        return self.w_o(out)


# encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, encoder_mask):
        x = self.norm1(x + self.self_attn(x, mask=encoder_mask))
        x = self.norm2(x + self.ffn(x))
        return x


# decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, decoder_mask, encoder_mask):
        x = self.norm1(x + self.self_attn(x, mask=decoder_mask))
        x = self.norm2(
            x + self.cross_attn(x, encoder_output, encoder_output, encoder_mask)
        )
        x = self.norm3(x + self.ffn(x))
        return x


# transformer
class Transformer(nn.Module):
    def __init__(
        self,
        encoder_vocab_size,
        decoder_vocab_size,
        seq_len,
        d_model=512,
        N=6,
        heads=8,
        d_ff=2048,
        dropout=0.1
    ):
        super().__init__()

        self.encoder_embed = nn.Embedding(encoder_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(decoder_vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, seq_len, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)]
        )

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)]
        )

        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, decoder_vocab_size)

    def encode(self, encoder_input, encoder_mask):
        x = self.pos(self.encoder_embed(encoder_input))
        for layer in self.encoder_layers:
            x = layer(x, encoder_mask)
        return self.norm(x)

    def decode(self, decoder_input, encoder_output, decoder_mask, encoder_mask):
        x = self.pos(self.decoder_embed(decoder_input))
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, decoder_mask, encoder_mask)
        return self.norm(x)

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(
            decoder_input, encoder_output, decoder_mask, encoder_mask
        )
        return self.proj(decoder_output)
