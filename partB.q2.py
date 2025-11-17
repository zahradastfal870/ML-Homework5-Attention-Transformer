import torch
import torch.nn as nn

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=8, dim_ff=256):
        super(SimpleTransformerEncoder, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


# ---------- TEST SHAPE ----------
if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    d_model = 128
    n_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    encoder = SimpleTransformerEncoder(d_model=d_model, n_heads=n_heads)

    output, attn_weights = encoder(x)

    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)
