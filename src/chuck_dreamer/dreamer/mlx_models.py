"""Example MLX-based neural network models for robotics applications."""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron using MLX."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 128, 64),
                 output_dim: int = 1,
                 activation: str = "relu",
                 dropout_rate: float = 0.1):
        """
        Initialize the MLP.

        Args:
            input_dim: Input dimension
            hidden_dims: Tuple of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function ("relu", "gelu", "silu")
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        # Build layers
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())

            # Add dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = layers

    def __call__(self, x):
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    """Basic Transformer block using MLX."""

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout_rate: float = 0.1):
        """
        Initialize transformer block.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dimension to model dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, num_heads)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )

    def __call__(self, x, mask=None):
        """Forward pass with residual connections."""
        # Self-attention with residual connection
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class WorldModelEncoder(nn.Module):
    """Example world model encoder for robotics applications."""

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 latent_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8):
        """
        Initialize the world model encoder.

        Args:
            observation_dim: Dimension of observations
            action_dim: Dimension of actions
            latent_dim: Latent representation dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()

        # Input projections
        self.obs_proj = nn.Linear(observation_dim, latent_dim)
        self.action_proj = nn.Linear(action_dim, latent_dim)

        # Positional embedding (learned)
        self.pos_embedding = mx.random.normal(
            (1000, latent_dim))  # Max sequence length

        # Transformer layers
        self.layers = [
            TransformerBlock(latent_dim, num_heads)
            for _ in range(num_layers)
        ]

        # Output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def __call__(
            self,
            observations,
            actions,
            sequence_length: Optional[int] = None):
        """
        Encode observation-action sequences.

        Args:
            observations: Observation tensor [batch, seq_len, obs_dim]
            actions: Action tensor [batch, seq_len, action_dim]
            sequence_length: Optional sequence length for padding mask

        Returns:
            Encoded representations [batch, seq_len, latent_dim]
        """
        batch_size, seq_len = observations.shape[:2]

        # Project inputs
        obs_tokens = self.obs_proj(observations)
        action_tokens = self.action_proj(actions)

        # Combine observations and actions
        # (simple concatenation along sequence)
        # Alternative: interleave or use different combination strategies
        tokens = obs_tokens + action_tokens  # Element-wise addition

        # Add positional embeddings
        pos_emb = self.pos_embedding[:seq_len]
        tokens = tokens + pos_emb

        # Apply transformer layers
        x = tokens
        for layer in self.layers:
            x = layer(x)

        # Final projection
        x = self.output_proj(x)

        return x


# Example loss functions
def mse_loss(predictions, targets):
    """Mean squared error loss."""
    return mx.mean((predictions - targets) ** 2)


def cross_entropy_loss(logits, targets):
    """Cross entropy loss."""
    return mx.mean(nn.losses.cross_entropy(logits, targets))


# Example training step
def train_step(model, optimizer, batch):
    """Example training step."""

    def loss_fn(model):
        observations, actions, targets = batch
        predictions = model(observations, actions)
        return mse_loss(predictions, targets)

    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss


if __name__ == "__main__":
    # Example usage
    print("MLX World Model Example")

    # Create a simple MLP
    mlp = SimpleMLP(input_dim=10, hidden_dims=(64, 32), output_dim=5)

    # Test forward pass
    x = mx.random.normal((32, 10))  # Batch of 32, input dim 10
    output = mlp(x)
    print(f"MLP output shape: {output.shape}")

    # Create a world model encoder
    encoder = WorldModelEncoder(
        observation_dim=84,  # Example: flattened 84x84 image or state vector
        action_dim=6,        # Example: 6-DOF robot actions
        latent_dim=256
    )

    # Test forward pass
    obs = mx.random.normal((8, 10, 84))    # Batch=8, seq_len=10, obs_dim=84
    actions = mx.random.normal((8, 10, 6))  # Batch=8, seq_len=10, action_dim=6
    encoded = encoder(obs, actions)
    print(f"Encoder output shape: {encoded.shape}")

    print("Models created successfully!")
