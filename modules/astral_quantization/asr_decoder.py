import torch
import torch.nn as nn
import torch.nn.functional as F

class ASRDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        block_size: int = 4096,
        in_channels: int = 512,
        n_vocab: int = 51866,
        bos_id: int = 50528,
        eos_id: int = 50527,
        dropout_rate: float = 0.0,
        attn_dropout_rate: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.depth = depth
        self.block_size = block_size
        self.in_channels = in_channels
        self.n_vocab = n_vocab
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, n_vocab)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, block_size, hidden_dim))
        
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len, in_channels)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        if seq_len <= self.block_size:
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            x = x + self.pos_embedding[:, :self.block_size, :]
        
        # Create attention mask if lengths provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
            mask = mask >= lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        else:
            mask = None
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)
        
        return logits
