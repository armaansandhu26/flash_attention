# Flash Attention Experiments

This repository contains notebook-based experiments and benchmark visualizations for Flash Attention implementations, including Triton and Pallas variants.

## Notebook Overview

### `trition_attn.ipynb` (Triton FlashAttention walkthrough)

This notebook is a full systems-style build of a Triton FlashAttention implementation, developed in stages and validated along the way:

- starts from tiled forward-pass scaffolding and incrementally builds the full kernel (Q/K/V tile loading, score computation, online softmax, output writeback),
- benchmarks forward pass throughput and then analyzes arithmetic intensity and roofline position,
- implements a custom backward kernel, checks correctness against PyTorch autograd gradients, and wraps everything in a `torch.autograd.Function`,
- benchmarks backward pass performance, estimates occupancy, and includes profiling/roofline efficiency decomposition.

It is the "from-kernel-construction-to-performance-analysis" notebook for the Triton path.

### `pallas_attn_gpt2.ipynb` (Pallas attention in GPT-2 training context)

This notebook focuses on integrating and evaluating Pallas-based attention in a GPT-2 style workload rather than only isolated kernel snippets:

- compares attention variants (including a naive XLA path and a Pallas Flash-style path),
- measures end-to-end training speed for GPT-2 124M settings,
- runs kernel-level roofline analysis for naive vs Pallas Flash implementations,
- uses shared utility helpers for throughput and roofline calculations to keep comparisons consistent.

It is the "model-training + kernel-roofline" notebook for the Pallas path.

## Repository Layout

- `utils/` - plotting, roofline, and timing helper utilities.
- `results/` - generated benchmark figures and roofline plots.

## Quick Start

1. Create and activate a Python environment.
2. Install your required JAX/Triton/plotting dependencies.
3. Open the notebooks and run cells to reproduce experiments.
