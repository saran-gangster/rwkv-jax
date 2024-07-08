# RWKV-JAX

## Overview

This project is a JAX implementation of the RWKV (Receptance Weighted Key Value) language model. RWKV is an novel architecture that combines the efficiency of RNNs with the powerful expressiveness of Transformers. The long-term goal of this JAX implementation is to become as fast or faster than the official implementation.

⚠️ **Note: This is yet to be fully optimized on GPU, Official implementation uses cuda kernel for wkv computation, Hopefully this gap will be bridged in future!** 
Currently Great for TPU training and good for running medium scale pretraining/finetuning/experiments.

## Installation

```bash
pip install -r requirements.txt
```

### Data Preparation

Head to [json2binidx](https://github.com/saran-gangster/rwkv-jax/json2binidx_tool)

Thanks to [Howard-Hou json2binidx_tool](https://github.com/howard-hou/json2binidx_tool/tree/main) (Orginal)

## Usage

### Training

Just Edit the config and start training/finetuning.
```bash
python train.py (after editing the config)
```

### Testing

```bash
python generate.py --model_path your/path/model.rwkv
```
By default generate.py will use config.yaml model-config to generate, but you specify it manually by '--config yourconfig.yaml' , Also Checkout the other args in generate.py

## Current Limitations and Future Improvements

- Implemented only Data parallesim for multi node training.
- Implement custom cuda kernel for time mixing for gpu training.
- Implement mixed precision training(not sure currently).
- Add the other time mixing versions.
- Implement State tuning.

## Contributing

- Write a conversion script for weights (.rwkv to .pth) to make it compatible with other RWKV projects.
- Implement State tuning.

Bug fixes, feature additions, or performance improvements, your input is valuable. Please feel free to open issues or submit pull requests, Thanks.

## Acknowledgments

This implementation is based on the original RWKV model developed by [BlinkDL](https://github.com/BlinkDL/RWKV-LM).
