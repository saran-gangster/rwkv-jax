import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Dict, Any
from functools import partial

class GroupNorm(nn.Module):
    """Applies Group Normalization over a mini-batch of inputs."""
    num_groups: int
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x):
        batch, seq_len, channels = x.shape
        x = x.reshape(batch * seq_len, self.num_groups, -1)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.epsilon)
        x = x.reshape(batch, seq_len, channels)
        return self.param('scale', nn.initializers.ones, (channels,)) * x + self.param('bias', nn.initializers.zeros, (channels,))

class RWKVBlock(nn.Module):
    """Represents a single block in the RWKV model."""
    config: Dict[str, Any]
    layer_id: int

    def setup(self):
        args = self.config
        self.head_size = args['head_size_a']
        self.n_head = args['dim_att'] // self.head_size
        assert args['dim_att'] % self.n_head == 0

        ratio_0_to_1 = self.layer_id / (args['n_layer'] - 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_id / args['n_layer'])

        def init_time_maa(name, ratio):
            def init(key, shape):
                ddd = jnp.arange(args['n_embd']) / args['n_embd']
                return 1.0 - jnp.power(ddd, ratio).reshape(1, 1, -1)
            return self.param(name, init, (1, 1, args['n_embd']))

        self.time_maa_x = init_time_maa('time_maa_x', ratio_1_to_almost0)
        self.time_maa_w = init_time_maa('time_maa_w', ratio_1_to_almost0)
        self.time_maa_k = init_time_maa('time_maa_k', ratio_1_to_almost0)
        self.time_maa_v = init_time_maa('time_maa_v', ratio_1_to_almost0 + 0.3 * ratio_0_to_1)
        self.time_maa_r = init_time_maa('time_maa_r', 0.5 * ratio_1_to_almost0)
        self.time_maa_g = init_time_maa('time_maa_g', 0.5 * ratio_1_to_almost0)

        def init_time_decay(key, shape):
            ddd = jnp.arange(self.n_head) / self.n_head
            return jnp.expand_dims(-5 + 8 * (ddd ** (0.7 + 1.3 * ratio_0_to_1)), axis=(0, 2, 3))
        self.time_decay = self.param('time_decay', init_time_decay, (1, self.n_head, 1, 1))

        def init_time_first(key, shape):
            return jnp.full(shape, -3.0)
        self.time_first = self.param('time_first', init_time_first, (1, self.n_head, 1, 1))

        self.time_faaaa = self.param('time_faaaa', nn.initializers.normal(0.01), (self.n_head, self.head_size))

        self.receptance = nn.Dense(args['dim_att'], use_bias=False)
        self.key = nn.Dense(args['dim_att'], use_bias=False)
        self.value = nn.Dense(args['dim_att'], use_bias=False)
        self.output = nn.Dense(args['n_embd'], use_bias=False)
        self.gate = nn.Dense(args['dim_att'], use_bias=False)
        self.ln_x = GroupNorm(num_groups=self.n_head)

        # Channel mixing setup
        self.time_maa_k_channel = init_time_maa('time_maa_k_channel', ratio_1_to_almost0)
        self.time_maa_r_channel = init_time_maa('time_maa_r_channel', ratio_1_to_almost0)

        self.key_channel = nn.Dense(args['dim_ffn'], use_bias=False)
        self.receptance_channel = nn.Dense(args['n_embd'], use_bias=False)
        self.value_channel = nn.Dense(args['n_embd'], use_bias=False)

        self.ln1 = nn.LayerNorm(epsilon=self.config['layer_norm_epsilon'])
        self.ln2 = nn.LayerNorm(epsilon=self.config['layer_norm_epsilon'])

    def time_shift(self, x):
        """Shift tensor in time dimension for time mixing."""
        return jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)

    def time_mixing(self, x, state):
        """Performs time mixing."""
        x_shape = x.shape
        if len(x_shape) == 2:
            B, C = x_shape
            T = 1
            x = x.reshape(B, T, C)
        elif len(x_shape) == 3:
            B, T, C = x_shape
        else:
            raise ValueError(f"Unexpected input shape: {x_shape}")
    
        H = self.n_head
        S = self.head_size

        xx = self.time_shift(x)
        sx = xx - x
        xk = x + sx * self.time_maa_k
        xv = x + sx * self.time_maa_v
        xr = x + sx * self.time_maa_r
        xg = x + sx * self.time_maa_g

        r = self.receptance(xr).reshape(B, T, H, S)
        k = self.key(xk).reshape(B, T, H, S)
        v = self.value(xv).reshape(B, T, H, S)
        g = jax.nn.silu(self.gate(xg))

        w = jnp.exp(self.time_decay).reshape(1, H, 1, 1)
        u = jnp.exp(self.time_first).reshape(1, H, 1, 1)
  
        kv = (k * v)

        # Ensure state has the correct shape
        state = state.reshape(B, H, S, S)

        # Update state
        new_state = state
        for t in range(T):
            new_state = new_state * w + kv[:, t].reshape(B, H, S, 1) @ kv[:, t].reshape(B, H, 1, S)
        state = new_state

        # Calculate y
        y = jnp.zeros((B, T, H, S))
        for t in range(T):
            y_t = (state @ kv[:, t].reshape(B, H, S, 1)).squeeze(-1) * u.squeeze(-1)
            y_t = y_t + kv[:, t]
            y_t = y_t * r[:, t]
            y = y.at[:, t].set(y_t)

        y = y.reshape(B, T, H * S)

        y = self.ln_x(y)

        y = self.output(y * g)

        return y, state

    def channel_mixing(self, x):
        """Performs channel mixing."""
        xx = self.time_shift(x)
        sx = xx - x
        xk = x + sx * self.time_maa_k_channel
        xr = x + sx * self.time_maa_r_channel

        k = jax.nn.relu(self.key_channel(xk)) ** 2
        kv = self.value_channel(k)
        return jax.nn.sigmoid(self.receptance_channel(xr)) * kv

    def __call__(self, x, state, deterministic=True):
        """Defines the forward pass of the block."""
        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.reshape(orig_shape[0], 1, orig_shape[1])
        
        x_attn, new_state = self.time_mixing(self.ln1(x), state)
        x = x + x_attn
        x = x + self.channel_mixing(self.ln2(x))
        
        if not deterministic:
            key = self.make_rng('dropout')
            x = nn.Dropout(rate=self.config['dropout'], deterministic=deterministic)(x, key)
        
        if len(orig_shape) == 2:
            x = x.reshape(orig_shape)
        
        return x, new_state

@nn.compact
class RWKV(nn.Module):
    """Represents the full RWKV model."""
    vocab_size: int
    n_layer: int
    n_embd: int
    dim_att: int
    dim_ffn: int
    head_size_a: int
    n_head: int
    head_size_divisor: int
    dropout: float
    layer_norm_epsilon: float

    def setup(self):
        self.emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        self.blocks = [RWKVBlock({
            'vocab_size': self.vocab_size,
            'n_layer': self.n_layer,
            'n_embd': self.n_embd,
            'dim_att': self.dim_att,
            'dim_ffn': self.dim_ffn,
            'head_size_a': self.head_size_a,
            'n_head': self.n_head,
            'head_size_divisor': self.head_size_divisor,
            'dropout': self.dropout,
            'layer_norm_epsilon': self.layer_norm_epsilon
        }, i) for i in range(self.n_layer)]
        self.ln_out = nn.LayerNorm(epsilon=self.layer_norm_epsilon)
        self.head = nn.Dense(self.vocab_size, use_bias=False)
        
    def __call__(self, idx, state, deterministic=True):
        x = self.emb(idx)
        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.reshape(orig_shape[0], 1, orig_shape[1])
        
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state[:, i], deterministic=deterministic)
            new_states.append(new_state)
        
        x = self.ln_out(x)
        x = self.head(x)
        
        if len(orig_shape) == 2:
            x = x.reshape(orig_shape[0], -1)
        
        return x, jnp.stack(new_states, axis=1)

    
    @classmethod
    def init_state(cls, config):
        return lambda batch_size: jnp.zeros((batch_size, config['n_layer'], config['n_head'], config['head_size_a'], config['head_size_a']))

    def get_init_state(self, batch_size):
        return jnp.zeros((batch_size, self.n_layer, self.n_head, self.head_size_a, self.head_size_a))
    
    def forward_parallel(self, idx, state, deterministic=True):
        """Defines the forward pass with parallel computation."""
        x = self.emb(idx)
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state[:, i], deterministic=deterministic)
            new_states.append(new_state)
        x = self.ln_out(x)
        x = self.head(x)
        return x, jnp.stack(new_states, axis=1)
        
def create_model(config):
    """Creates and initializes the RWKV model."""
    model = RWKV(**config)
    key = random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 16), dtype=jnp.int32)
    
    @jax.jit
    def init_fn(key):
        dummy_state = RWKV.init_state(config)(1)
        return model.init(key, dummy_input, dummy_state)

    params = init_fn(key)
    return model, params

@partial(jax.jit, static_argnums=(0,))
def model_forward(model, params, idx, state, deterministic=True):
    """Forward pass for the model."""
    return model.apply(params, idx, state, deterministic=deterministic, rngs={'dropout': random.PRNGKey(0)})

@partial(jax.jit, static_argnums=(0,))
def model_forward_parallel(model, params, idx, state, deterministic=True):
    """Parallel forward pass for the model."""
    return model.apply(params, idx, state, method=model.forward_parallel, deterministic=deterministic, rngs={'dropout': random.PRNGKey(0)})

if __name__ == "__main__":
    config = {
        'vocab_size': 50277,
        'n_layer': 12,
        'n_embd': 768,
        'dim_att': 768,
        'dim_ffn': 2688,
        'head_size_a': 64,
        'n_head': 12,
        'head_size_divisor': 8,
        'dropout': 0.1,
        'layer_norm_epsilon': 1e-5,
    }

    # Initialize model and parameters
    model, params = create_model(config)

    # Initialize state
    batch_size = 1
    init_state_fn = model.init_state(config)
    state = init_state_fn(batch_size)

    # Prepare input
    input_ids = jnp.array([[1, 2, 3]])  # Example input

    # Forward pass
    output, new_state = model_forward(model, params, input_ids, state)

    print("Model initialized successfully!")
    print(f"Output shape: {output.shape}")
    print(f"New state shape: {new_state.shape}")

    # Parallel processing example
    parallel_input_ids = jnp.array([[1, 2, 3, 4, 5]])  # Longer sequence
    parallel_output, parallel_new_state = model_forward_parallel(model, params, parallel_input_ids, state)

    print("\nParallel processing:")
    print(f"Parallel output shape: {parallel_output.shape}")
    print(f"Parallel new state shape: {parallel_new_state.shape}")
