import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from bonsai.models.gat.modeling import GATConfig, GATModel


class TestGAT(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.config = GATConfig(
            num_nodes=20,
            in_channels=8,
            hidden_dim=4,
            num_heads=2,
            dropout_prob=0.0,  # Deterministic
            alpha=0.2,
            num_layers=2,
            out_channels=3,
            concat_last=False,
        )
        self.rngs = nnx.Rngs(0)
        self.model = GATModel(self.config, rngs=self.rngs)

    def test_shape(self):
        key = jax.random.key(0)
        x = jax.random.normal(key, (self.config.num_nodes, self.config.in_channels))
        adj = jnp.eye(self.config.num_nodes)  # Identity adjacency

        output = self.model(x, adj, rngs=self.rngs)
        self.assertEqual(output.shape, (self.config.num_nodes, self.config.out_channels))

    def test_permutation_invariance(self):
        # GAT should be permutation equivariant if adj is permuted accordingly
        # But if adj is identity (no edges except self), it operates like a shared MLP per node (mostly).
        # Let's test if we permute nodes and adjacency, output is permuted.

        key = jax.random.key(2)
        x = jax.random.normal(key, (self.config.num_nodes, self.config.in_channels))
        # Random adj
        adj = jax.random.bernoulli(key, 0.3, (self.config.num_nodes, self.config.num_nodes)).astype(jnp.float32)

        # Forward original
        out1 = self.model(x, adj, rngs=self.rngs)

        # Permute
        perm = jax.random.permutation(jax.random.key(3), self.config.num_nodes)
        x_perm = x[perm]
        adj_perm = adj[perm][:, perm]

        out2 = self.model(x_perm, adj_perm, rngs=self.rngs)

        # Undo permutation on output
        inv_perm = jnp.argsort(perm)
        out2_restored = out2[inv_perm]

        # Check close
        # Note: floating point differences might occur, but should be small
        np.testing.assert_allclose(out1, out2_restored, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    absltest.main()
