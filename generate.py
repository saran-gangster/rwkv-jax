import os
import pickle
import yaml
import jax
import jax.numpy as jnp
from functools import partial
import argparse
from src.model import RWKV, RWKVConfig
from src.tokenizer import RWKVTokenizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config):
    return RWKV(config)

def load_pretrained_weights(model_path, model):
    with open(model_path, 'rb') as f:
        initial_params = pickle.load(f)
    
    def reshape_params(loaded, current):
        if loaded.shape != current.shape:
            if len(loaded.shape) > len(current.shape):
                slices = tuple(slice(None) for _ in range(len(current.shape)))
                loaded = loaded[slices]
            return jnp.resize(loaded, current.shape)
        return loaded

    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    dummy_state = RWKV.get_init_state(rwkv_config, 1)
    variables = jax.jit(model.init)(jax.random.PRNGKey(0), dummy_input, dummy_state)
    
    reshaped_params = jax.tree_util.tree_map(
        reshape_params,
        initial_params,
        variables['params']
    )
    
    return reshaped_params

@partial(jax.jit, static_argnums=(0, 1))
def infer_single_token(model, params, token, state):
    token_array = jnp.array([[token]])
    logits, new_state = model.apply({'params': params}, token_array, state)
    return logits[0, 0], new_state

def generate_text(model, params, config, tokenizer, initial_prompt, max_length=100, temperature=0.8):
    tokens = tokenizer.encode(initial_prompt)[0]
    state = RWKV.get_init_state(config, batch_size=1)
    generated_tokens = []
    
    for token in tokens:
        logits, state = infer_single_token(model, params, token, state)
    
    for _ in range(max_length):
        logits /= temperature
        token = jax.random.categorical(jax.random.PRNGKey(int.from_bytes(os.urandom(4), byteorder='little')), logits)
        generated_tokens.append(token.item())
        logits, state = infer_single_token(model, params, token.item(), state)
    
    generated_text = tokenizer.decode([tokens + generated_tokens])[0]
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RWKV V6 Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model weights")
    parser.add_argument("--tokenizer_path", type=str, default='rwkv_vocab_v20230424.txt', help="Path to the tokenizer file")
    parser.add_argument("--initial_prompt", type=str, default="Once upon a time, ", help="Prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length for generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config['model']

    if model_config['min_clamp'] is None:
        model_config['min_clamp'] = 10 ** (-74 / model_config['chunk_size'])

    rwkv_config = RWKVConfig(**model_config)

    model = create_model(rwkv_config)
    params = load_pretrained_weights(args.model_path, model)

    tokenizer = RWKVTokenizer(args.tokenizer_path)

    generated_text = generate_text(model, params, rwkv_config, tokenizer, 
                                   args.initial_prompt, args.max_length, args.temperature)
    print(generated_text)
