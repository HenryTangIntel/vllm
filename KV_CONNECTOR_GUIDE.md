# vLLM KV Connector Guide

## What is a KV Connector?

A **KV connector** in vLLM is a component that enables **disaggregated prefilling** - a technique that separates the prefill and decode phases of LLM inference into different vLLM instances.

### Problem Solved
In standard LLM inference, prefill (processing the input prompt) and decode (generating output tokens) happen in the same instance, which can cause:
- Difficulty tuning TTFT (time-to-first-token) and ITL (inter-token-latency) separately
- Unpredictable tail latency when prefill jobs interrupt decoding

### Solution: Disaggregated Architecture
Run separate vLLM instances:
- **Prefill instance**: Processes input prompts and generates KV caches
- **Decode instance**: Receives KV caches and generates output tokens
- **KV connector**: Transfers the KV caches between instances

## Architecture Clarification

**KV connector is NOT a standalone instance**. It's a **component/module** that runs **inside** the prefill and decode instances to enable communication between them.

```
┌─────────────────────┐    KV Cache Transfer    ┌─────────────────────┐
│   Prefill Instance  │ ◄─────────────────────► │   Decode Instance   │
│                     │                         │                     │
│ ┌─────────────────┐ │                         │ ┌─────────────────┐ │
│ │  LLM Engine     │ │                         │ │  LLM Engine     │ │
│ │                 │ │                         │ │                 │ │
│ │ ┌─────────────┐ │ │                         │ │ ┌─────────────┐ │ │
│ │ │KV Connector │ │ │                         │ │ │KV Connector │ │ │
│ │ │(Producer)   │ │ │                         │ │ │(Consumer)   │ │ │
│ │ └─────────────┘ │ │                         │ │ └─────────────┘ │ │
│ └─────────────────┘ │                         │ └─────────────────┘ │
└─────────────────────┘                         └─────────────────────┘
     GPU 0/Process 1                                GPU 1/Process 2
```

**What Actually Runs:**
- **Two vLLM instances**: Complete vLLM servers with embedded KV connectors
- **KV Connector is**: A software component within each instance handling transfer protocol
- **Not 3 separate things**: You have 2 instances (prefill + decode), not 3 (prefill + decode + connector)

## Available KV Connectors (6 Types)

vLLM supports **6 different KV connectors** for disaggregated prefilling, each optimized for different use cases:

### 1. **SharedStorageConnector** 
- **Type**: File system-based storage
- **Use Case**: Development, testing, simple setups
- **Performance**: High latency, low throughput
- **Setup Complexity**: Simple
- **Best For**: Local development and testing

### 2. **PyNcclConnector** (Legacy/V0)
- **Type**: Traditional NCCL-based transfers  
- **Use Case**: Legacy support, simpler NCCL setup
- **Performance**: Medium latency, high throughput
- **Setup Complexity**: Medium
- **Best For**: Simple GPU-to-GPU transfers

### 3. **P2pNcclConnector**
- **Type**: Point-to-point NCCL with proxy coordination
- **Use Case**: Production NCCL deployments
- **Performance**: Medium latency, high throughput
- **Setup Complexity**: High (requires proxy server)
- **Best For**: Production environments with NCCL infrastructure

### 4. **LMCacheConnectorV1**
- **Type**: LMCache integration with multiple backends
- **Use Case**: Advanced caching with CPU offload, cross-instance sharing
- **Performance**: Low latency, very high throughput
- **Setup Complexity**: High (requires LMCache installation)
- **Best For**: Production workloads with shared prefixes

### 5. **NixlConnector**
- **Type**: NVIDIA NIXL high-performance networking
- **Use Case**: Ultra-low latency GPU-to-GPU via RDMA
- **Performance**: Very low latency, very high throughput  
- **Setup Complexity**: High (requires UCX/NIXL installation)
- **Best For**: Ultra-performance requirements with RDMA hardware

### 6. **MultiConnector**
- **Type**: Composite connector combining multiple backends
- **Use Case**: Fallback mechanisms and hybrid approaches
- **Performance**: Variable (depends on combination)
- **Setup Complexity**: High (combines multiple connectors)
- **Best For**: Fault tolerance and reliability

## Key Components

- **Connector**: Main interface for KV cache transfer between producer/consumer
- **LookupBuffer**: Provides SQL-like `insert` and `drop_select` APIs for KV cache management
- **Pipe**: Single-direction FIFO for tensor transmission with `send_tensor`/`recv_tensor`

## How to Launch KV Connector Instances

### Method 1: Python API (Offline Inference)

#### Basic NCCL Setup
```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Prefill instance (KV producer)
prefill_config = KVTransferConfig(
    kv_connector="PyNcclConnector",
    kv_role="kv_producer",
    kv_rank=0,
    kv_parallel_size=2
)

llm_prefill = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    kv_transfer_config=prefill_config,
    gpu_memory_utilization=0.8
)

# Decode instance (KV consumer) 
decode_config = KVTransferConfig(
    kv_connector="PyNcclConnector", 
    kv_role="kv_consumer",
    kv_rank=1,
    kv_parallel_size=2
)

llm_decode = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    kv_transfer_config=decode_config,
    gpu_memory_utilization=0.8
)
```

#### Shared Storage Connector (Simpler)
```python
# Both instances can use the same config
shared_config = KVTransferConfig(
    kv_connector="SharedStorageConnector",
    kv_role="kv_both",  # Can be producer and consumer
    kv_connector_extra_config={"shared_storage_path": "local_storage"}
)

llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    kv_transfer_config=shared_config,
    gpu_memory_utilization=0.8
)
```

#### Complete Example with Process Coordination
```python
import os
from multiprocessing import Event, Process

def run_prefill(prefill_done):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    prompts = ["Hello, my name is", "Hi, your name is"]
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
    )
    
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        kv_transfer_config=ktc,
        gpu_memory_utilization=0.8,
    )
    
    llm.generate(prompts, sampling_params)
    prefill_done.set()

def run_decode(prefill_done):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    prompts = ["Hello, my name is", "Hi, your name is"]
    sampling_params = SamplingParams(temperature=0, top_p=0.95)
    
    ktc = KVTransferConfig(
        kv_connector="PyNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
    )
    
    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        kv_transfer_config=ktc,
        gpu_memory_utilization=0.8,
    )
    
    prefill_done.wait()  # Wait for prefill to complete
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Prompt: {output.prompt}, Generated: {output.outputs[0].text}")

# Launch both processes
prefill_done = Event()
prefill_process = Process(target=run_prefill, args=(prefill_done,))
decode_process = Process(target=run_decode, args=(prefill_done,))

prefill_process.start()
decode_process.start()

decode_process.join()
prefill_process.terminate()
```

### Method 2: Server API (Online Serving)

#### Launch Prefill Server
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8100 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
```

#### Launch Decode Server  
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
```

#### Complete Server Script
```bash
#!/bin/bash

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Launch prefill server
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
    --port 8100 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' &

# Launch decode server
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL_NAME \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' &

# Wait for servers to be ready
wait_for_server() {
  timeout 1200 bash -c "
    until curl -s localhost:$1/v1/completions > /dev/null; do
      sleep 1
    done"
}

wait_for_server 8100
wait_for_server 8200

echo "Both servers are ready!"
```

## Detailed Connector Configurations

### 1. SharedStorageConnector
```python
KVTransferConfig(
    kv_connector="SharedStorageConnector",
    kv_role="kv_both",  # Can be producer and consumer
    kv_connector_extra_config={
        "shared_storage_path": "local_storage"  # Directory path
    }
)
```

### 2. PyNcclConnector (Legacy)
```python 
KVTransferConfig(
    kv_connector="PyNcclConnector",
    kv_role="kv_producer",  # or "kv_consumer"
    kv_rank=0,  # 0 for producer, 1 for consumer
    kv_parallel_size=2  # Total instances
)
```

### 3. P2pNcclConnector
```python
KVTransferConfig(
    kv_connector="P2pNcclConnector",
    kv_role="kv_producer",
    kv_buffer_size="8e9",  # 8GB buffer
    kv_port="21001",
    kv_connector_extra_config={
        "proxy_ip": "10.0.1.1",
        "proxy_port": "30001", 
        "http_port": "20001"
    }
)
```

### 4. LMCacheConnectorV1
```python
KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_connector_extra_config={
        "config_path": "lmcache-config.yaml"
    }
)

# lmcache-config.yaml example:
# enable_nixl: True
# nixl_role: "sender"  # or "receiver"
# nixl_peer_host: "localhost"
# nixl_peer_port: 55555
# nixl_buffer_size: 1073741824  # 1GB
```

### 5. NixlConnector
```python
KVTransferConfig(
    kv_connector="NixlConnector",
    kv_role="kv_both"
    # NIXL configuration handled internally
)
```

### 6. MultiConnector
```python
KVTransferConfig(
    kv_connector="MultiConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "connectors": [
            {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both"
            },
            {
                "kv_connector": "SharedStorageConnector", 
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "shared_storage_path": "local_storage"
                }
            }
        ]
    }
)
```

## Performance & Use Case Matrix

| Connector | Latency | Throughput | Complexity | Hardware Req | Best For |
|-----------|---------|------------|------------|--------------|----------|
| **SharedStorage** | High | Low | Simple | Any | Development/Testing |
| **PyNcclConnector** | Medium | High | Medium | CUDA GPUs | Simple GPU-GPU |
| **P2pNcclConnector** | Medium | High | High | NCCL + Proxy | Production NCCL |  
| **LMCacheConnectorV1** | Low | Very High | High | LMCache + Backends | Advanced caching |
| **NixlConnector** | Very Low | Very High | High | UCX + RDMA | Ultra-performance |
| **MultiConnector** | Variable | Variable | High | Combined | Fault tolerance |

## Selection Guidelines

- **Development/Testing**: `SharedStorageConnector` (simplest setup)
- **Simple Production**: `PyNcclConnector` (good balance of simplicity and performance)
- **Advanced Production**: `LMCacheConnectorV1` (best caching and flexibility)
- **Ultra-Performance**: `NixlConnector` (lowest latency with RDMA)
- **High Availability**: `MultiConnector` (fallback mechanisms)
- **Legacy Compatibility**: `PyNcclConnector` (V0 support)

## Configuration Parameters

### Core Parameters
- `kv_connector`: Connector type (one of the 6 types above)
- `kv_role`: Instance role
  - `"kv_producer"`: Generates and sends KV caches (prefill instance)  
  - `"kv_consumer"`: Receives and uses KV caches (decode instance)
  - `"kv_both"`: Can both produce and consume (for shared storage)
- `kv_rank`: Instance rank (0 for producer, 1 for consumer in NCCL-based connectors)
- `kv_parallel_size`: Total number of instances (usually 2)

### Advanced Parameters
- `kv_buffer_size`: Buffer size for transfers (connector-dependent)
- `kv_port`: Communication port (for network-based connectors)
- `kv_connector_extra_config`: Connector-specific configuration dictionary

## Process Flow

1. **Prefill instance**: 
   - Receives prompts
   - Processes prefill phase
   - Generates KV caches
   - Transfers KV caches via connector
   - Returns minimal response (often just 1 token)

2. **Decode instance**: 
   - Receives same prompts
   - Waits for/retrieves KV caches via connector
   - Continues generation from where prefill left off
   - Returns full completion

3. **Coordination**: 
   - Use multiprocessing Events for offline inference
   - Use proxy servers for online serving
   - Ensure proper timing between instances

## Benefits

- **Separate optimization**: Tune prefill (TTFT) and decode (ITL) independently
- **Predictable latency**: Eliminate decode interruption by prefill jobs  
- **Resource allocation**: Assign different parallel strategies to each phase
- **Tail latency control**: More reliable than chunked prefill

## Limitations

- **No throughput improvement**: Disaggregation focuses on latency control
- **Complexity**: Requires coordination between multiple instances
- **Resource overhead**: Running two instances instead of one
- **Experimental**: Feature is still evolving and subject to change

## Use Cases

Ideal for scenarios where you need:
- Strict ITL guarantees without prefill interference
- Different optimization strategies for prefill vs decode
- Fine-grained control over inference latency characteristics
- High-priority decode workloads that cannot tolerate interruption