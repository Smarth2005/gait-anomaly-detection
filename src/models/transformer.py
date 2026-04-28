import torch
import torch.nn as nn
import math

from src.config import (
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD, DROPOUT, BOTTLENECK_DIM, PATCH_SIZE
)

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding to inject sequence order info
    into the transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return x


class PatchTransformerAutoencoder(nn.Module):
    """
    Advanced Patch-based Time-Series Transformer Autoencoder (2024 architecture).
    
    Instead of processing data point-by-point (which is noisy and slow for Transformers),
    this model slices the time series into non-overlapping "patches", embeds them, 
    and applies self-attention. This dramatically improves context capturing and efficiency.
    """

    def __init__(
        self,
        input_dim: int = 38,
        seq_len: int = 128,
        patch_size: int = PATCH_SIZE,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        bottleneck_dim: int = BOTTLENECK_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        
        # ---------------------------------------------------------
        # ENCODER
        # ---------------------------------------------------------
        # Linear layer to embed each patch into d_model dimensions
        self.patch_embedding = nn.Linear(input_dim * patch_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.num_patches)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Flatten patches and project to bottleneck
        self.to_bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_patches * d_model, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim)
        )
        
        # ---------------------------------------------------------
        # DECODER
        # ---------------------------------------------------------
        # Project bottleneck back to patched d_model sequence
        self.from_bottleneck = nn.Linear(bottleneck_dim, self.num_patches * d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        # Using a TransformerEncoder as a decoder block since we just need to process the latent sequence
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Project embedded patches back to raw sensor data
        self.patch_reconstruction = nn.Linear(d_model, input_dim * patch_size)

    def encode(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        batch_size = x.size(0)
        
        # 1. Create Patches: (batch_size, num_patches, patch_size, input_dim)
        x_patched = x.reshape(batch_size, self.num_patches, self.patch_size, self.input_dim)
        
        # 2. Flatten each patch: (batch_size, num_patches, patch_size * input_dim)
        x_patched = x_patched.reshape(batch_size, self.num_patches, -1)
        
        # 3. Embed Patches
        embedded = self.patch_embedding(x_patched)  # (batch_size, num_patches, d_model)
        
        # 4. Positional Encoding (Transformer expects seq_len, batch, d_model if batch_first=False, 
        # but our PE is designed for seq_len first, so we transpose, apply, and transpose back)
        embedded = embedded.transpose(0, 1)  # (num_patches, batch_size, d_model)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)  # (batch_size, num_patches, d_model)
        
        # 5. Transformer Encoder
        encoded = self.transformer_encoder(embedded)  # (batch_size, num_patches, d_model)
        
        # 6. Bottleneck
        z = self.to_bottleneck(encoded)  # (batch_size, bottleneck_dim)
        return z

    def decode(self, z):
        """
        z: (batch_size, bottleneck_dim)
        """
        batch_size = z.size(0)
        
        # 1. Expand from bottleneck
        h = self.from_bottleneck(z)  # (batch_size, num_patches * d_model)
        h = h.reshape(batch_size, self.num_patches, self.d_model)  # (batch_size, num_patches, d_model)
        
        # 2. Positional Encoding
        h = h.transpose(0, 1)
        h = self.pos_encoder(h)
        h = h.transpose(0, 1)
        
        # 3. Transformer Decoder
        decoded = self.transformer_decoder(h)  # (batch_size, num_patches, d_model)
        
        # 4. Reconstruct Patches
        reconstructed_patches = self.patch_reconstruction(decoded)  # (batch_size, num_patches, patch_size * input_dim)
        
        # 5. Reshape back to raw sequence
        reconstruction = reconstructed_patches.reshape(batch_size, self.seq_len, self.input_dim)
        return reconstruction

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z

    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary."""
        print(f"Patch-based Transformer Autoencoder (2024 Architecture)")
        print(f"  Input:      ({self.seq_len}, {self.input_dim})")
        print(f"  Patch Size: {self.patch_size} -> {self.num_patches} patches")
        print(f"  D_Model:    {self.d_model}")
        print(f"  Bottleneck: {self.bottleneck_dim}")
        print(f"  Parameters: {self.get_num_params():,}")
