# Graph Attention Network (GAT)

This module implements the Graph Attention Network (GAT) using JAX and Flax (NNX).

## Usage

```python
import jax
import jax.numpy as jnp
from flax import nnx
from bonsai.models.gat import GATConfig, GATModel

# Define configuration
config = GATConfig(
    num_nodes=100,
    in_channels=16,
    hidden_dim=8,
    num_heads=8,
    dropout_prob=0.6,
    alpha=0.2,
    num_layers=2,
    out_channels=7,
    concat_last=False
)

# Initialize model
rngs = nnx.Rngs(0)
model = GATModel(config, rngs=rngs)

# Dummy data
key1, key2 = jax.random.split(jax.random.key(0))
x = jax.random.normal(key1, (100, 16)) # [num_nodes, in_channels]
# Random adjacency matrix (binary)
adj = jax.random.bernoulli(key2, 0.1, (100, 100)).astype(jnp.float32)

# Forward pass
output = model(x, adj, rngs=rngs)
print(output.shape) # Should be (100, 7)
```
