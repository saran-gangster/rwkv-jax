import os
import pickle
import yaml
import jax
import optax
import orbax
from flax.training import train_state
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from src.model import RWKV, RWKVConfig
from src.tokenizer import RWKVTokenizer
from src.binidx import MMapIndexedDataset
from flax.training import orbax_utils

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            else:
                self[key] = value

with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

config = DotDict(config_dict)

devices = jax.local_devices()
num_devices = len(devices)
batch_size_per_device = config.training.batch_size // num_devices
assert config.training.batch_size % num_devices == 0, f"Batch size must be divisible by the number of devices. Got {config.training.batch_size} and {num_devices} devices."

config.model.min_clamp = config.model.min_clamp if config.model.min_clamp==str(None) else 10 ** (-74 / config.model.chunk_size)
rwkv_config = RWKVConfig(**config.model)

global_step = 0

tokenizer = RWKVTokenizer(config.paths.tokenizer_path)

def create_checkpoint_manager(directory, max_to_keep=config.checkpoint.max_checkpoints_to_keep):
    directory = os.path.abspath(directory)
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=max_to_keep, create=True)
    return orbax.checkpoint.CheckpointManager(
        directory, orbax.checkpoint.PyTreeCheckpointer(), options)

def create_learning_rate_schedule():
    def schedule(step):
        warmup_factor = jnp.minimum(step / config.training.warmup_steps, 1.0)
        warmup_lr = config.training.initial_learning_rate + (config.training.max_learning_rate - config.training.initial_learning_rate) * warmup_factor
        decay_factor = jnp.maximum(0.0, 1.0 - (step - config.training.warmup_steps) / config.training.decay_steps)
        decay_lr = config.training.max_learning_rate * decay_factor
        return jnp.where(step < config.training.warmup_steps, warmup_lr, decay_lr)
    return schedule

def create_model(config):
    return RWKV(config)

def init_model_variables(model, rwkv_config):
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
    dummy_state = RWKV.get_init_state(rwkv_config, 1)
    variables = model.init(key, dummy_input, dummy_state)
    return variables

def reinitialize_optimizer_state(tx, params):
    opt_state = tx.init(params)
    return opt_state

def reshape_optimizer_state(old_state, new_params):
    new_state = jax.tree_util.tree_map(
        lambda old, new: jnp.resize(old, new.shape) if old.shape != new.shape else old,
        old_state, new_params
    )
    return new_state

def create_train_state(model, variables, learning_rate_schedule):
    tx = optax.chain(
        optax.clip_by_global_norm(config.training.grad_clip_norm),
        optax.clip(config.training.grad_clip_value),
        optax.adamw(learning_rate_schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

def save_checkpoint(checkpoint_manager, train_state, step):
    save_args = orbax_utils.save_args_from_target(train_state)
    checkpoint_manager.save(step, train_state, save_kwargs={'save_args': save_args})
    print(f"\nCheckpoint saved at step {step}\n")

def load_checkpoint(checkpoint_manager, train_state):
    step = checkpoint_manager.latest_step()
    if step is not None:
        loaded_state = checkpoint_manager.restore(step, items=train_state)
        
        updated_params = jax.tree_util.tree_map(
            lambda current, loaded: jnp.resize(loaded, current.shape) if current.shape != loaded.shape else loaded,
            train_state.params,
            loaded_state.params
        )
        
        new_opt_state = reinitialize_optimizer_state(train_state.tx, updated_params)
        reshaped_opt_state = reshape_optimizer_state(loaded_state.opt_state, new_opt_state)
        
        loaded_state = train_state.replace(params=updated_params, opt_state=reshaped_opt_state)
        return loaded_state, step
    return None, 0

def save_model_weights(params, path):
    with open(path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model weights saved to {path}")

dataset = MMapIndexedDataset(config.paths.data_path)

def pad_sequences(sequences, max_len, pad_value=0):
    padded = []
    for seq in sequences:
        if isinstance(seq, (list, np.ndarray)):
            flat_seq = np.concatenate(seq) if isinstance(seq, list) else seq
        else:
            flat_seq = np.array(seq)
        if len(flat_seq) > max_len:
            padded.append(flat_seq[:max_len])
        else:
            padded.append(np.pad(flat_seq, (0, max_len - len(flat_seq)), constant_values=pad_value))
    return np.array(padded, dtype=np.int32)

def create_mask(padded_sequences, max_len):
    return np.array([[1 if i < np.sum(seq != 0) else 0 for i in range(max_len)] for seq in padded_sequences])

def compute_loss(logits, labels, mask):
    num_classes = logits.shape[-1]
    smooth_positives = 1.0 - config.training.label_smoothing
    smooth_negatives = config.training.label_smoothing / num_classes
    onehot_labels = jax.nn.one_hot(labels, num_classes)
    smooth_labels = onehot_labels * smooth_positives + smooth_negatives
    loss = -jnp.sum(smooth_labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    loss = (loss * mask).sum() / (mask.sum())
    return loss

@partial(jax.pmap, axis_name='batch')
def train_step(state, batch, mask, init_state, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
    def loss_fn(params):
        logits, new_state = state.apply_fn(
            {'params': params}, batch[:, :-1], init_state, 
            deterministic=False, 
            rngs={'dropout': dropout_rng}
        )
        loss = compute_loss(logits, batch[:, 1:], mask[:, 1:])
        return loss, (logits, new_state)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (_, new_state)), grads = grad_fn(state.params)
    grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isnan(g), jnp.zeros_like(g), g), grads)
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    max_grad = jax.lax.pmean(jnp.max(jnp.abs(jax.tree_util.tree_leaves(grads)[0])), axis_name='batch')
    return new_state, loss, new_state, max_grad, jnp.isnan(loss).any(), new_dropout_rng

def train():
    global global_step

    total_tokens = len(dataset)
    tokens_per_step = config.training.batch_size * config.training.seq_len
    steps_per_epoch = total_tokens // tokens_per_step
    total_steps = steps_per_epoch * config.training.epochs
    learning_rate_schedule = create_learning_rate_schedule()
    
    model = create_model(rwkv_config)
    
    variables = init_model_variables(model, rwkv_config)
    train_state = create_train_state(model, variables, learning_rate_schedule)
    checkpoint_manager = create_checkpoint_manager(config.paths.save_path)
    
    loaded_state, start_step = load_checkpoint(checkpoint_manager, train_state)
    if loaded_state is not None:
        train_state = loaded_state
        global_step = start_step
        print(f"Loaded checkpoint at step {global_step}")
    else:
        if os.path.exists(config.paths.model_path):
            with open(config.paths.model_path, 'rb') as f:
                initial_params = pickle.load(f)
            
            def reshape_params(loaded, current):
                if loaded.shape != current.shape:
                    if len(loaded.shape) > len(current.shape):
                        slices = tuple(slice(None) for _ in range(len(current.shape)))
                        loaded = loaded[slices]
                    return jnp.resize(loaded, current.shape)
                return loaded

            reshaped_params = jax.tree_util.tree_map(
                reshape_params,
                initial_params,
                train_state.params
            )
            
            train_state = train_state.replace(params=reshaped_params)
            global_step = 0
            print("Loaded Pre-trained weights")
        else:
            global_step = 0
            print("No checkpoint or Pre-trained weights found. Starting from scratch.")
    
    if global_step >= total_steps:
        print("Training already completed. Increase epochs if you want to train more.")
        return
    
    train_state = jax.device_put_replicated(train_state, devices)

    dummy_batch = jnp.ones((num_devices, batch_size_per_device, config.training.seq_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((num_devices, batch_size_per_device, config.training.seq_len), dtype=jnp.int32)
    dummy_init_state = RWKV.get_init_state(rwkv_config, batch_size_per_device)
    dummy_init_state = jnp.repeat(dummy_init_state[jnp.newaxis, ...], num_devices, axis=0)
    dropout_rng = jax.random.PRNGKey(0)
    dropout_rng = jax.random.split(dropout_rng, num_devices)
    
    train_state, _, _, _, _, dropout_rng = train_step(train_state, dummy_batch, dummy_mask, dummy_init_state, dropout_rng)

    start_epoch = global_step // steps_per_epoch
    for epoch in range(start_epoch, config.training.epochs):
        epoch_loss = 0
        epoch_max_grad = 0
        with tqdm(total=steps_per_epoch, initial=global_step % steps_per_epoch, 
                  desc=f"Training (Epoch {epoch + 1}/{config.training.epochs})") as pbar:
            for step in range(global_step % steps_per_epoch, steps_per_epoch):
                rng = jax.random.PRNGKey(global_step)
                
                max_start_idx = total_tokens - config.training.seq_len * config.training.batch_size
                start_idx = jax.random.randint(rng, (1,), 0, max_start_idx)[0]
                sequences = [dataset[start_idx + i*config.training.seq_len : start_idx + (i+1)*config.training.seq_len] for i in range(config.training.batch_size)]

                padded_sequences = pad_sequences(sequences, config.training.seq_len)
                mask = create_mask(padded_sequences, config.training.seq_len)

                padded_sequences = padded_sequences.reshape(num_devices, batch_size_per_device, config.training.seq_len)
                mask = mask.reshape(num_devices, batch_size_per_device, config.training.seq_len)

                padded_sequences, mask = jnp.array(padded_sequences), jnp.array(mask)

                init_state = RWKV.get_init_state(rwkv_config, batch_size_per_device)
                init_state = jnp.repeat(init_state[jnp.newaxis, ...], num_devices, axis=0)

                train_state, loss, _, max_grad, is_nan, dropout_rng = train_step(
                    train_state, padded_sequences, mask, init_state, dropout_rng
                )

                loss = jax.device_get(loss)
                max_grad = jax.device_get(max_grad)
                epoch_loss += jnp.mean(loss)
                epoch_max_grad = max(epoch_max_grad, jnp.max(max_grad))

                is_nan = jax.device_get(is_nan)
                if np.any(is_nan):
                    print(f"\nWarning: NaN detected at step {global_step}")
                elif np.isinf(np.mean(loss)):
                    print(f"\nWarning: Inf loss detected at step {global_step}")

                global_step += 1
                pbar.update(1)

                if config.checkpoint.save_every_steps > 0 and global_step % config.checkpoint.save_every_steps == 0:
                    save_checkpoint(checkpoint_manager, jax.device_get(train_state), global_step)

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"Avg Loss: {avg_epoch_loss:.4f}, Max grad: {epoch_max_grad:.4f}")

        if config.checkpoint.save_every_epochs > 0 and (epoch + 1) % config.checkpoint.save_every_epochs == 0:
            save_checkpoint(checkpoint_manager, jax.device_get(train_state), global_step)

    print("Training completed.")
    save_checkpoint(checkpoint_manager, jax.device_get(train_state), global_step)
    
    final_params = jax.device_get(train_state.params)
    save_model_weights(final_params, config.paths.model_path)

if __name__ == "__main__":
    train()
