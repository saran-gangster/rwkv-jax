# RWKV-JAX

## Overview

This project is a JAX implementation of the RWKV (Receptance Weighted Key Value) language model. RWKV is an novel architecture that combines the efficiency of RNNs with the powerful expressiveness of Transformers. The long-term goal of this JAX implementation is to become as fast or faster than the official implementation, While maintaining Good quality codebase.

⚠️ **Note: This project is in its very early stages of development.** Many optimizations and features are yet to be implemented. Contributions and feedback are welcome!

## Installation

-  pip install all the packages
-  Get .idx and .bin of your data from the official implementation
-  Edit the config and start training


## Usage

### Training

```bash
python train.py
```
### Testing

```bash
python generate.py
```


## Current Limitations and Future Improvements

- Implemented only Data parallesim for multi node training.
- Long sequences training intialization is extremely slow.
- Boosting model speed by Jax primitives/custom cuda kernels.

## Contributing

Bug fixes, feature additions, or performance improvements, your input is valuable. Please feel free to open issues or submit pull requests, Thanks.

## Acknowledgments

This implementation is based on the original RWKV model developed by [BlinkDL](https://github.com/BlinkDL/RWKV-LM).
