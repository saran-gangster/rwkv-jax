import jax
import jax.numpy as jnp
from jax import random
import jax.lax as lax
from functools import partial
from typing import Dict, Any, Optional, Tuple
from src.tokenizer import RWKVTokenizer

def sample_logits(logits: jnp.ndarray, temperature: float = 1.0, top_p: float = 0.8) -> jnp.ndarray:
    """
    Sample from the logits tensor produced by the model.

    Args:
        logits (jnp.ndarray): Logits tensor from the model, shape [batch_size, vocab_size].
        temperature (float): Temperature parameter for controlling the diversity of sampling.
        top_p (float): Top-p truncation parameter for controlling the sampling probability distribution.

    Returns:
        jnp.ndarray: Sampled token indices, shape [batch_size].
    """
    if top_p == 0.0:
        # Deterministically select the most likely token
        return jnp.argmax(logits, axis=-1)

    if temperature != 1.0:
        logits = logits / temperature

    sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    
    cutoff = jnp.take_along_axis(sorted_logits, jnp.argmax(cumulative_probs > top_p, axis=-1, keepdims=True), axis=-1)
    logits = jnp.where(logits < cutoff, -jnp.inf, logits)

    return jax.random.categorical(random.PRNGKey(0), logits)

def apply_penalties(logits: jnp.ndarray, 
                    temperature: float, 
                    top_p: float, 
                    presence_penalty: float, 
                    frequency_penalty: float, 
                    token: Optional[jnp.ndarray] = None, 
                    freq_dict: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Apply penalties to the logits tensor and sample from it.
    """
    if freq_dict is None:
        freq_dict = jnp.zeros(logits.shape[-1], dtype=jnp.int32)

    if token is not None:
        # Create a mask for tokens that have appeared in the generated sequence
        mask = jnp.zeros_like(logits, dtype=jnp.bool_)
        mask = mask.at[jnp.arange(logits.shape[0]), token].set(True)

        # Apply presence penalty
        logits = jnp.where(mask, logits - presence_penalty, logits)
       
        # Apply frequency penalty based on token occurrence
        freq_penalties = freq_dict / jnp.maximum(jnp.sum(freq_dict), 1)
        freq_penalties = jnp.clip(freq_penalties, 0, 1)
        logits -= frequency_penalty * freq_penalties

    token_sampled = sample_logits(logits, temperature=temperature, top_p=top_p)
    
    # Update frequency dictionary
    freq_dict = freq_dict.at[token_sampled].add(1)

    if token is not None:
        token = jnp.append(token, token_sampled)
    else:
        token = token_sampled

    return token_sampled, token, freq_dict

@partial(jax.jit, static_argnums=(0, 2, 3, 4, 5, 6, 7))
def generate_tokens(model: Any, 
                    params: Dict, 
                    vocab_size: int,
                    num_tokens: int, 
                    temperature: float = 1.0, 
                    top_p: float = 0.8, 
                    presence_penalty: float = 0.0, 
                    frequency_penalty: float = 0.0, 
                    initial_tokens: Optional[jnp.ndarray] = None, 
                    initial_state: Optional[jnp.ndarray] = None):
    """
    Generate tokens using the RWKV model.
    """
    def body_fn(carry, _):
        tokens, current_idx, state, freq_dict = carry
        current_token = lax.dynamic_slice(tokens, (current_idx - 1,), (1,))
        logits, new_state = model.apply(params, current_token.reshape(1, -1), state)
        token_sampled, _, new_freq_dict = apply_penalties(
            logits[:, -1],
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            current_token,
            freq_dict
        )
        new_tokens = lax.dynamic_update_slice(tokens, token_sampled, (current_idx,))
        return (new_tokens, current_idx + 1, new_state, new_freq_dict), token_sampled

    if initial_tokens is None:
        initial_tokens = jnp.array([0], dtype=jnp.int32)
    if initial_state is None:
        initial_state = model.get_init_state(1)

    initial_freq_dict = jnp.zeros(vocab_size, dtype=jnp.int32)
    total_tokens = initial_tokens.shape[0] + num_tokens
    padded_tokens = jnp.pad(initial_tokens, (0, num_tokens), constant_values=0)
    initial_idx = jnp.array(initial_tokens.shape[0], dtype=jnp.int32)

    (final_tokens, _, final_state, _), _ = lax.scan(
        body_fn, (padded_tokens, initial_idx, initial_state, initial_freq_dict), None, length=num_tokens
    )

    return final_tokens, final_state

if __name__ == "__main__":
    from model import create_model, model_forward

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

    model, params = create_model(config)
    
    tokenizer = RWKVTokenizer("/home/sarangangster/Desktop/rwkv_jax/rwkv_vocab_v20230424.txt")

    input_text = "Once upon a time,"
    num_tokens_to_generate = 20

    encoded_input = tokenizer.encode_numpy(input_text)
    initial_tokens = jnp.array(encoded_input[0], dtype=jnp.int32)

    initial_state = model.init_state(config)(1)

    generated_tokens, final_state = generate_tokens(
        model, 
        params, 
        vocab_size=config['vocab_size'],
        num_tokens=num_tokens_to_generate, 
        temperature=0.8, 
        top_p=0.9, 
        initial_tokens=initial_tokens, 
        initial_state=initial_state
    )

    decoded_text = tokenizer.decode_numpy(generated_tokens.reshape(1, -1))[0]

    print("Input text:", input_text)
    print("Generated text:", decoded_text)
    print("Final state shape:", final_state.shape)
