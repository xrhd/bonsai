import time

import jax
import jax.numpy as jnp
from flax import nnx

from bonsai.models.gat import modeling as model_lib


def run_model():
    # Config
    config = model_lib.GATConfig.gat_test()

    # Initialize model
    rngs = nnx.Rngs(0)
    model = model_lib.GATModel(config, rngs=rngs)

    # Dummy data
    num_nodes = config.num_nodes
    in_channels = config.in_channels

    key1, key2 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(key1, (num_nodes, in_channels))
    # Random sparse adjacency (10% density)
    adj = jax.random.bernoulli(key2, 0.1, (num_nodes, num_nodes)).astype(jnp.float32)
    # Ensure self-loops
    adj = adj.at[jnp.diag_indices(num_nodes)].set(1.0)

    @nnx.jit
    def forward(model, x, adj):
        return model(x, adj)

    # Warmup
    print("Compiling...")
    _ = forward(model, x, adj)
    print("Compilation done.")

    # Profile
    t0 = time.perf_counter()
    for _ in range(100):
        out = forward(model, x, adj)
        out.block_until_ready()
    print(f"Step time: {(time.perf_counter() - t0) / 100:.6f} s")

    print("Output shape:", out.shape)
    assert out.shape == (num_nodes, config.out_channels)


if __name__ == "__main__":
    run_model()
