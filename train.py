import os
import re
import jax
import jax.numpy as jnp
from jax import random, pmap, local_device_count, jit, tree_util
import optax
from flax.training import train_state
from flax.training import checkpoints
from functools import partial
import numpy as np
from tqdm import tqdm
import pickle
import time
from src.model import create_model, RWKV, model_forward
from src.tokenizer import RWKVTokenizer
from src.binidx import MMapIndexedDataset

# Configurations
MODEL_PATH = './weight/RWKV-x060.pkl'
TOKENIZER_PATH = "rwkv_vocab_v20230424.txt"
DATA_PATH = 'data/minipile'
SAVE_PATH = os.path.abspath("rwkv-pretrain-checkpoint")  # Make this an absolute path

config = {
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

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SEQ_LEN = 10
EPOCHS = 10
SAVE_EVERY = 1000  # Save checkpoint every 1000 steps

# Multi-GPU setup
devices = jax.devices()
num_devices = len(devices)
print(f"Number of devices: {num_devices}")

BATCH_SIZE_PER_DEVICE = BATCH_SIZE // num_devices
assert BATCH_SIZE % num_devices == 0, f"Batch size must be divisible by the number of devices. Got {BATCH_SIZE} and {num_devices} devices."

# Initialize tokenizer
tokenizer = RWKVTokenizer(TOKENIZER_PATH)

# Load or create model
def init_or_load_model(config, model_path):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        model = RWKV(**config)
    else:
        print("Creating new model")
        model, params = create_model(config)
        with open(model_path, 'wb') as f:
            pickle.dump(params, f)
    return model, params

def init_state(config):
    return lambda batch_size: jnp.zeros((batch_size, config['n_layer'], config['n_head'], config['head_size_a'], config['head_size_a']))

model, params = init_or_load_model(config, MODEL_PATH)


def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Checkpoint loaded, step: {checkpoint['step']}")
    return checkpoint

def save_checkpoint(train_state, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_file = os.path.join(save_dir, "checkpoint")
    
    # Extract the raw parameters from the TrainState
    raw_params = jax.device_get(train_state.params)

    # Create the checkpoint dictionary
    checkpoint = {
        'params': raw_params,
        'step': step
    }

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Checkpoint saved at step {step}")

def create_train_state(params, learning_rate):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint_')]
    if not checkpoints:
        return None
    
    def extract_step(checkpoint_name):
        match = re.search(r'checkpoint_(\d+)', checkpoint_name)
        return int(match.group(1)) if match else -1
    
    latest_checkpoint = max(checkpoints, key=extract_step)
    return os.path.join(checkpoint_dir, latest_checkpoint)

def extract_step(checkpoint_name):
    match = re.search(r'checkpoint_(\d+)', checkpoint_name)
    return int(match.group(1)) if match else 0

# Load dataset
dataset = MMapIndexedDataset(DATA_PATH)

# Padding function
def pad_sequences(sequences, max_len, pad_value=0):
    padded = []
    for seq in sequences:
        if isinstance(seq, list):  # If seq is a list of arrays
            flat_seq = np.concatenate([arr for arr in seq if len(arr) > 0])
        else:  # If seq is already a flat array
            flat_seq = seq
        if len(flat_seq) > max_len:
            padded.append(flat_seq[:max_len])
        else:
            padded.append(np.pad(flat_seq, (0, max_len - len(flat_seq)), constant_values=pad_value))
    return np.array(padded, dtype=np.int32)

# Create mask for valid tokens
def create_mask(padded_sequences, max_len):
    return np.array([[1 if i < np.sum(seq != 0) else 0 for i in range(max_len)] for seq in padded_sequences])

@jax.pmap
def train_step(state, batch, mask, init_state):
    def loss_fn(params):
        logits, _ = model.apply(params, batch[:, :-1], init_state)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, config['vocab_size']),
            batch[:, 1:].reshape(-1)
        )
        loss = (loss * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def train():
    print("Initializing training state...")
    
    # Check for existing checkpoints
    latest_checkpoint = find_latest_checkpoint(SAVE_PATH)
    if latest_checkpoint:
        checkpoint_file = os.path.join(latest_checkpoint, "checkpoint")
        print(f"Loading checkpoint from {checkpoint_file}")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                loaded_state = pickle.load(f)
            train_state = create_train_state(loaded_state['params'], LEARNING_RATE)
            start_step = loaded_state['step']
        else:
            print(f"Checkpoint file not found: {checkpoint_file}")
            print("Starting from scratch.")
            train_state = create_train_state(params, LEARNING_RATE)
            start_step = 0
    else:
        print("No checkpoint found. Starting from scratch.")
        train_state = create_train_state(params, LEARNING_RATE)
        start_step = 0
    
    # Replicate the train state for each device
    train_state = jax.device_put_replicated(train_state, devices)
    
    print("Training state initialized.")

    # Prepare dummy data for compilation
    dummy_batch = jnp.ones((num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN), dtype=jnp.int32)
    dummy_mask = jnp.ones((num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN), dtype=jnp.int32)
    dummy_init_state = jnp.zeros((num_devices, BATCH_SIZE_PER_DEVICE, config['n_layer'], config['n_head'], config['head_size_a'], config['head_size_a']))

    # Compile the training step
    print("Compiling training step...")
    train_step(train_state, dummy_batch, dummy_mask, dummy_init_state)
    print("Compilation done.")

    total_steps = (len(dataset) // (BATCH_SIZE * SEQ_LEN)) * EPOCHS

    for epoch in range(EPOCHS):
        total_steps = len(dataset) // (BATCH_SIZE * SEQ_LEN)
        with tqdm(total=total_steps, initial=start_step, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for step in range(start_step, total_steps):
                rng = jax.random.PRNGKey(epoch * total_steps + step)

                # Sample a batch
                idxs = jax.random.randint(rng, (BATCH_SIZE,), 0, len(dataset) - SEQ_LEN)
                sequences = [dataset[idx:idx+SEQ_LEN] for idx in idxs]

                # Pad sequences and create mask
                padded_sequences = pad_sequences(sequences, SEQ_LEN)
                mask = create_mask(padded_sequences, SEQ_LEN)

                # Reshape for multiple devices
                padded_sequences = padded_sequences.reshape(num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN)
                mask = mask.reshape(num_devices, BATCH_SIZE_PER_DEVICE, SEQ_LEN)

                # Convert to jax arrays
                padded_sequences = jnp.array(padded_sequences)
                mask = jnp.array(mask)

                # Initialize state
                init_state = RWKV.init_state(config)(BATCH_SIZE)
                init_state = init_state.reshape(num_devices, BATCH_SIZE_PER_DEVICE, config['n_layer'], config['n_head'], config['head_size_a'], config['head_size_a'])

                # Perform training step
                train_state, loss = train_step(train_state, padded_sequences, mask, init_state)

                pbar.update(1)

                if (step + 1) % 100 == 0:
                    loss = jax.device_get(loss)
                    print(f"Step {step+1}, Loss: {loss.mean():.4f}")

                if (step + 1) % SAVE_EVERY == 0:
                    save_dir = os.path.join(SAVE_PATH, f"checkpoint_{step + 1}")
                    os.makedirs(save_dir, exist_ok=True)
                    # Gather params from all devices and save only one copy
                    flat_params = jax.tree_util.tree_map(lambda x: x[0], jax.device_get(train_state.params))
                    checkpoint_file = os.path.join(save_dir, "checkpoint")
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'params': flat_params, 'step': step + 1}, f)

        # Reset start_step after first epoch
        start_step = 0

    # Save final model
    final_save_dir = os.path.join(SAVE_PATH, "checkpoint_final")
    os.makedirs(final_save_dir, exist_ok=True)
    flat_params = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x, jax.device_get(train_state.params))
    final_checkpoint_file = os.path.join(final_save_dir, "checkpoint")
    with open(final_checkpoint_file, 'wb') as f:
        pickle.dump({'params': flat_params, 'step': total_steps}, f)
        
# Run training
if __name__ == "__main__":
    train()
