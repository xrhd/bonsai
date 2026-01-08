import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx


@dataclasses.dataclass(frozen=True)
class GATConfig:
    num_nodes: int
    in_channels: int
    hidden_dim: int
    num_heads: int
    dropout_prob: float
    alpha: float  # LeakyReLU negative slope
    num_layers: int
    out_channels: int
    concat_last: bool = False  # Whether to concat or average the last layer heads
    add_skip_connection: bool = True
    bias: bool = True

    @classmethod
    def gat_test(cls):
        return cls(
            num_nodes=100,
            in_channels=16,
            hidden_dim=8,
            num_heads=8,
            dropout_prob=0.6,
            alpha=0.2,
            num_layers=2,
            out_channels=7,  # e.g. Cora classes
            concat_last=False,
            add_skip_connection=True,
            bias=True,
        )


class GATLayer(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        alpha: float,
        dropout_prob: float,
        concat: bool = True,
        add_skip_connection: bool = True,
        bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        self.alpha = alpha
        self.dropout_prob = dropout_prob
        self.add_skip_connection = add_skip_connection

        # Projection: [in_dim] -> [num_heads * out_dim]
        self.proj = nnx.Linear(in_dim, num_heads * out_dim, use_bias=False, rngs=rngs)
        
        # Skip connection projection if dims don't match
        self.skip_proj = None
        if add_skip_connection:
            if in_dim != num_heads * out_dim:
                 self.skip_proj = nnx.Linear(in_dim, num_heads * out_dim, use_bias=False, rngs=rngs)

        # Attention parameters: [1, num_heads, out_dim]
        # In Imp3: scoring_fn_target and scoring_fn_source
        self.a_src = nnx.Param(jax.random.normal(rngs.params(), (1, num_heads, out_dim)))
        self.a_dst = nnx.Param(jax.random.normal(rngs.params(), (1, num_heads, out_dim)))
        
        if bias:
            self.bias = nnx.Param(jnp.zeros((num_heads * out_dim if concat else out_dim,)))
        else:
            self.bias = None
        
        self.dropout = nnx.Dropout(dropout_prob, rngs=rngs)

    def __call__(self, h: jnp.ndarray, adj: jnp.ndarray, *, rngs: nnx.Rngs | None = None) -> jnp.ndarray:
        # h: [N, in_dim]
        N = h.shape[0]

        # Dropout on input features (as per paper/reference)
        h_dropped = self.dropout(h, rngs=rngs)

        # 1. Linear Projection
        # [N, in_dim] -> [N, num_heads * out_dim] -> [N, num_heads, out_dim]
        h_prime = self.proj(h_dropped).reshape(N, self.num_heads, self.out_dim)

        # 2. Attention Mechanism
        # Calculate scores per head: src and dst
        # [N, NH, F] * [1, NH, F] -> [N, NH, F] --sum--> [N, NH]
        scores_src = (h_prime * self.a_src).sum(axis=-1)
        scores_dst = (h_prime * self.a_dst).sum(axis=-1)

        # Broadcast add: [N, 1, NH] + [1, N, NH] -> [N, N, NH]
        # score_ij = score_src_i + score_dst_j
        scores = scores_src[:, None, :] + scores_dst[None, :, :]
        
        scores = jax.nn.leaky_relu(scores, negative_slope=self.alpha)

        # Masking
        # adj is [N, N]. We expand to [N, N, 1] for broadcasting over heads
        mask = jnp.where(adj > 0, 0.0, -9e15)
        scores = scores + mask[..., None]

        # Softmax over neighbors (dim 1)
        attn_weights = jax.nn.softmax(scores, axis=1) # [N, N, NH]
        attn_weights = self.dropout(attn_weights, rngs=rngs)

        # 3. Aggregation
        # h'_i = sum_j alpha_ij Wh_j
        # [N, N, NH] @ [N, NH, F] -> wait, we need per-head matmul
        # einsum 'ijh,jhf->ihf'
        h_out = jnp.einsum('ijh,jhf->ihf', attn_weights, h_prime)

        # 4. Concat or Mean + Skip + Bias
        if self.concat:
            # [N, NH, F] -> [N, NH * F]
            h_out = h_out.reshape(N, self.num_heads * self.out_dim)
        else:
            # [N, NH, F] -> [N, F]
            h_out = h_out.mean(axis=1)

        if self.add_skip_connection:
             if self.skip_proj:
                 skip = self.skip_proj(h) # Original input (not dropped) commonly used for skip? Reference uses input to layer. References uses 'in_nodes_features' which was dropped? 
                 # Reference: in_nodes_features = self.dropout(in_nodes_features) (L664)
                 # Then logic uses in_nodes_features for skip (L827). So it uses the dropped version?
                 # Wait, in reference 'forward' L657 unpacks data. L664 drops it.
                 # L827 (skip) uses 'in_nodes_features'. 
                 # So yes, it uses the DO-ed input.
                 # Let's match reference: use h_dropped for skip?
                 # Actually in reference GATLayer L664: in_nodes_features = self.dropout(in_nodes_features)
                 # Then L819 skip_concat_bias takes in_nodes_features.
                 # So yes, it is the dropped features.
                 # BUT, normally skip connections bypass the noise. 
                 # In GAT reference, 'in_nodes_features' variable is overwritten by dropout result.
                 pass 
             
             # Re-reading reference: 
             # L664: in_nodes_features = self.dropout(in_nodes_features)
             # ...
             # L704: out... = aggregate... (..., in_nodes_features, ...) // wait aggregate uses it? No, just for device/dtype
             # L710: out... = skip_concat_bias(..., in_nodes_features, out...)
             # So yes, the skip connection adds the DROPPED input.
             
             if self.skip_proj:
                skip_val = self.skip_proj(h_dropped)
                if self.concat:
                    skip_val = skip_val.reshape(N, self.num_heads * self.out_dim)
                else:
                    skip_val = skip_val.mean(axis=1) # If we are not concatenating, skip projection might be huge?
                    # StartLine 633: skip_proj = Linear(num_in... , num_heads * num_out...)
                    # L831: out = skip_proj(in).view(...)
                    # L838: out = out.mean(dim=head)
                    # So yes, we project to full head dim then mean if not concat.
             else:
                 # Check dims
                 if h_dropped.shape[-1] == (self.num_heads * self.out_dim if self.concat else self.out_dim):
                      skip_val = h_dropped
                 else:
                      # If dims mismatch and no skip_proj, we can't add. 
                      # But constructor handles creation of skip_proj if in != out (account for heads).
                      # Wait, if not concat, output is out_dim. Input is in_dim. 
                      # In reference L824: if out.shape[-1] == in.shape[-1].
                      # So if sizes match, just add.
                      skip_val = h_dropped

             # Wait, strict reference implementation:
             # If dims match (L824), unsqueeze and add. 
             # For us, h_out is already shaped.
             
             # Let's implement simpler logic matching shapes:
             # We need to add skip_val to h_out.
             # h_out shape is [N, NH*F] (concat) or [N, F] (mean).
             
             if self.skip_proj:
                 # Project: [N, NH*F]
                 skip_val = self.skip_proj(h_dropped)
                 if not self.concat:
                     skip_val = skip_val.reshape(N, self.num_heads, self.out_dim).mean(axis=1)
                 h_out = h_out + skip_val
             else:
                 if h_dropped.shape[-1] == h_out.shape[-1]:
                     h_out = h_out + h_dropped
                 # else: can't add, but init should have covered this.

        if self.bias is not None:
            h_out = h_out + self.bias[...]
        
        return h_out


class GATModel(nnx.Module):
    def __init__(self, cfg: GATConfig, *, rngs: nnx.Rngs):
        layers = []
        self.num_layers = cfg.num_layers

        # Hidden layers
        for i in range(cfg.num_layers - 1):
            layers.append(
                GATLayer(
                    in_dim=cfg.in_channels if i == 0 else cfg.hidden_dim * cfg.num_heads,
                    out_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    alpha=cfg.alpha,
                    dropout_prob=cfg.dropout_prob,
                    concat=True,
                    add_skip_connection=cfg.add_skip_connection,
                    bias=cfg.bias,
                    rngs=rngs,
                )
            )

        # Output layer
        # If concat_last is False, we average heads and output out_channels
        layers.append(
            GATLayer(
                in_dim=cfg.in_channels if cfg.num_layers == 1 else cfg.hidden_dim * cfg.num_heads,
                out_dim=cfg.out_channels,
                num_heads=cfg.num_heads,  # Usually output layer also has heads, but we might average them
                alpha=cfg.alpha,
                dropout_prob=cfg.dropout_prob,
                concat=cfg.concat_last,
                add_skip_connection=cfg.add_skip_connection,
                bias=cfg.bias,
                rngs=rngs,
            )
        )

        self.layers = nnx.List(layers)
        self.dropout_prob = cfg.dropout_prob
        self.activation = jax.nn.elu  # GAT uses ELU between layers usually

    def __call__(self, h: jnp.ndarray, adj: jnp.ndarray, *, rngs: nnx.Rngs | None = None) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            h = layer(h, adj, rngs=rngs)
            if i < self.num_layers - 1:
                h = self.activation(h)
                # Note: Dropout on features is often applied here too in some impls,
                # but GATLayer already has dropout on attention coefficients.
                # Standard GAT (Velickovic et al) applies dropout to input features of each layer.
                # We can add a dropout here if needed.

        # Final output (usually softmax is applied externally for classification)
        return h
