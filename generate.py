import os
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from src.model import RWKV, model_forward
from src.tokenizer import RWKVTokenizer
from src.sampler import generate_tokens
from train import load_checkpoint

MODEL_PATH = './weight/RWKV-x060.pkl'
TOKENIZER_PATH = "rwkv_vocab_v20230424.txt"
CHECKPOINT_DIR = os.path.abspath("rwkv-pretrain-checkpoint")
CONFIG = {
    'vocab_size': 65536,
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

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint_')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1]))
    return os.path.join(checkpoint_dir, latest_checkpoint, "checkpoint")


def main():
    tokenizer = RWKVTokenizer(TOKENIZER_PATH)

    latest_checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint_path is None:
        print("No checkpoint found. Using initial model.")
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
    else:
        checkpoint = load_checkpoint(latest_checkpoint_path)
        params = checkpoint['params']

    model = RWKV(**CONFIG)

    prompt = "Once upon a time, in a land far away,"
    input_ids = jnp.array(tokenizer.encode(prompt)[0])

    num_tokens_to_generate = 50
    temperature = 0.8
    top_p = 0.9

    generated_tokens, _ = generate_tokens(
        model,
        params,
        CONFIG['vocab_size'],
        num_tokens_to_generate,
        temperature=temperature,
        top_p=top_p,
        initial_tokens=input_ids,
    )
    generated_tokens_list = np.array(generated_tokens).flatten().tolist()
    generated_text = tokenizer.decode([generated_tokens_list])[0]

    print("Input prompt:", prompt)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()
