import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, seq_length, head_hidden_dim, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.seq_length = seq_length
        self.head_hidden_dim = head_hidden_dim
        self.hidden_dim = num_heads * head_hidden_dim
        self.dropout = dropout

        self.to_QKV = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.to_QKV(x).split(self.hidden_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)
        return attention


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.feed_forward(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, num_heads, seq_length, hidden_dim, ff_dim, dropout):
        super().__init__()

        self.num_heads = num_heads
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.head_hidden_dim = hidden_dim // num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.multi_head_attention = MultiHeadAttention(self.num_heads,
                                                       self.seq_length,
                                                       self.head_hidden_dim,
                                                       self.dropout)
        self.feed_forward = FeedForward(self.hidden_dim, self.ff_dim, self.dropout)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

    def forward(self, x):
        attention = self.multi_head_attention(self.norm1(x))
        feed_forward = self.feed_forward(self.norm2(attention + x))
        output = feed_forward + x
        return output


class GPT(nn.Module):
    def __init__(self, num_layers, vocab_size, num_heads, seq_length, hidden_dim, ff_dim, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(seq_length, hidden_dim)
        self.blocks = nn.Sequential(
            *[Block(num_heads, seq_length, hidden_dim, ff_dim, dropout) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        device = x.device

        tok_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(torch.arange(T, device=device))

        embed = tok_embed + pos_embed
        context_embeds = self.blocks(embed)
        normalized_embeds = self.norm(context_embeds)
        logits = self.linear_head(normalized_embeds)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, max_new_tokens):
        for i in range(max_new_tokens):
            chunk = x[:, -self.seq_length:]
            logits, _ = self(chunk)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, new_token), dim=1)
        return x

    def generate_generator(self, x, max_new_tokens):
        for i in range(max_new_tokens):
            chunk = x[:, -self.seq_length:]
            logits, _ = self(chunk)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, new_token), dim=1)
            yield new_token