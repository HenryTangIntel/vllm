# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
- Install development dependencies: `pip install -r requirements/dev.txt` (includes lint.txt and test.txt)
- Set up pre-commit hooks: `pip install -r requirements/lint.txt && pre-commit install`
- Build from source: `pip install -e .` (with cmake>=3.26.1, ninja, torch==2.7.1)

### Code Quality and Testing
- **Linting/Formatting**: The project uses pre-commit hooks. Run `pre-commit run --all-files` or let hooks run on commit
  - YAPF for Python formatting
  - Ruff for linting and additional formatting 
  - Typos checker, isort for imports
  - clang-format for C++/CUDA files
  - MyPy for type checking
- **Type checking**: `bash tools/mypy.sh` (runs mypy on core modules)
- **Testing**: `python -m pytest` with specific test paths (e.g., `pytest tests/models/test_*.py`)
  - Use `-v -s` for verbose output
  - Tests are organized by functionality (models, engine, core, distributed, etc.)
- **Single test execution**: `pytest tests/path/to/specific_test.py::test_function_name`

### Build and Development
- The project uses setuptools with pyproject.toml configuration
- CMake build system for CUDA/C++ kernels in `csrc/`
- Requirements are managed in `requirements/` directory with separate files for different purposes

## Architecture Overview

### Core Components

**Engine Layer** (`vllm/engine/`):
- `LLMEngine`: Main inference engine managing the full pipeline
- `AsyncLLMEngine`: Async wrapper providing non-blocking inference
- Output processors handle multi-step decoding and response formatting
- Request scheduling and batching logic

**Model Executor** (`vllm/model_executor/`):
- Model loading and weight management (`model_loader/`)
- Layer implementations (`layers/`) including attention, activation, quantization
- Model implementations (`models/`) for 100+ supported architectures
- Parallel execution support (tensor, pipeline, data parallelism)

**Attention Backends** (`vllm/attention/`):
- PagedAttention: Core innovation for memory-efficient KV cache management
- Multiple backends: FlashAttention, Triton, XFormers, custom implementations
- Platform-specific optimizations (CUDA, ROCm, TPU, CPU)

**Core Scheduling** (`vllm/core/`):
- Block manager for memory allocation
- Scheduler for request batching and execution planning
- KV cache management and eviction policies

**Distributed Execution** (`vllm/distributed/`):
- Multi-GPU and multi-node support
- NCCL integration for collective operations
- Expert parallelism for MoE models
- KV transfer system for disaggregated serving

### Key Architectural Concepts

**PagedAttention**: Memory is allocated in fixed-size blocks, enabling dynamic KV cache growth and efficient memory sharing between sequences with common prefixes.

**Continuous Batching**: Requests are dynamically batched and processed together, with new requests joining and completed requests leaving the batch seamlessly.

**Multi-Step Decoding**: Supports speculative decoding and multi-token generation for improved throughput.

**Quantization Support**: Extensive quantization backends (GPTQ, AWQ, FP8, INT8, etc.) in `vllm/model_executor/layers/quantization/`

**Multimodal Processing** (`vllm/multimodal/`): Unified pipeline for text, image, audio, and video inputs with model-specific processors.

### Entry Points

- **Python API**: `vllm.LLM` class for offline inference
- **OpenAI-Compatible Server**: `vllm.entrypoints.openai.api_server` 
- **CLI**: `vllm serve`, `vllm benchmark` commands in `vllm/entrypoints/cli/`

### Testing Strategy

Tests are organized by component:
- `tests/models/` - Model-specific correctness tests
- `tests/engine/` - Engine and scheduling tests  
- `tests/distributed/` - Multi-GPU/multi-node tests
- `tests/quantization/` - Quantization accuracy tests
- `tests/kernels/` - Kernel performance and correctness
- Hardware-specific tests in `.buildkite/scripts/hardware_ci/`

### Platform Support

The codebase supports NVIDIA GPUs (primary), AMD GPUs (ROCm), Intel CPUs/GPUs, TPU, and AWS Neuron with platform-specific implementations in `vllm/platforms/` and corresponding worker implementations.

## Important Implementation Notes

- All Python files must include SPDX license headers
- Use lazy imports in `__init__.py` files to reduce startup time
- Follow existing patterns for new model implementations (see `vllm/model_executor/models/`)
- CUDA kernels in `csrc/` require careful memory management and optimization
- Configuration classes use Pydantic dataclasses for validation (`vllm/config.py`)
- The project uses regex library instead of re for better Unicode support