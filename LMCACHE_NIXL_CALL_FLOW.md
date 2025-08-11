# LMCache + NIXL Call Flow Diagram

## System Architecture Overview

```
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│        Prefill Instance         │  NIXL   │        Decode Instance          │
│          (GPU 0)                │ Transfer│          (GPU 1)                │
├─────────────────────────────────┤ ◄─────► ├─────────────────────────────────┤
│                                 │         │                                 │
│ ┌─────────────────────────────┐ │         │ ┌─────────────────────────────┐ │
│ │        vLLM Engine          │ │         │ │        vLLM Engine          │ │
│ │                             │ │         │ │                             │ │
│ │ ┌─────────────────────────┐ │ │         │ │ ┌─────────────────────────┐ │ │
│ │ │     LMCacheConnectorV1  │ │ │         │ │ │     LMCacheConnectorV1  │ │ │
│ │ │                         │ │ │         │ │ │                         │ │ │
│ │ │ ┌─────────────────────┐ │ │ │         │ │ │ ┌─────────────────────┐ │ │ │
│ │ │ │      LMCache        │ │ │ │         │ │ │ │      LMCache        │ │ │ │
│ │ │ │                     │ │ │ │         │ │ │ │                     │ │ │ │
│ │ │ │ ┌─────────────────┐ │ │ │ │         │ │ │ │ ┌─────────────────┐ │ │ │ │
│ │ │ │ │  NIXL Backend   │ │ │ │ │ ◄─────► │ │ │ │ │  NIXL Backend   │ │ │ │ │
│ │ │ │ │   (Sender)      │ │ │ │ │         │ │ │ │ │  (Receiver)     │ │ │ │ │
│ │ │ │ └─────────────────┘ │ │ │ │         │ │ │ │ └─────────────────┘ │ │ │ │
│ │ │ └─────────────────────┘ │ │ │         │ │ │ └─────────────────────┘ │ │ │
│ │ └─────────────────────────┘ │ │         │ │ └─────────────────────────┘ │ │
│ └─────────────────────────────┘ │         │ └─────────────────────────────┘ │
└─────────────────────────────────┘         └─────────────────────────────────┘
            │                                           │
            ▼                                           ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│         NIXL Stack              │         │         NIXL Stack              │
│                                 │         │                                 │
│ ┌─────────────────────────────┐ │         │ ┌─────────────────────────────┐ │
│ │         NIXL API            │ │         │ │         NIXL API            │ │
│ └─────────────────────────────┘ │         │ └─────────────────────────────┘ │
│ ┌─────────────────────────────┐ │         │ ┌─────────────────────────────┐ │
│ │           UCX               │ │  RDMA   │ │           UCX               │ │
│ └─────────────────────────────┘ │ ◄─────► │ └─────────────────────────────┘ │
│ ┌─────────────────────────────┐ │         │ ┌─────────────────────────────┐ │
│ │     GPU Direct/CUDA         │ │         │ │     GPU Direct/CUDA         │ │
│ └─────────────────────────────┘ │         │ └─────────────────────────────┘ │
└─────────────────────────────────┘         └─────────────────────────────────┘
```

## Detailed Call Flow Sequence

### Phase 1: System Initialization

```mermaid
sequenceDiagram
    participant Client
    participant PrefillEngine as Prefill vLLM Engine
    participant LMCache_P as LMCache (Prefill)
    participant NIXL_P as NIXL (Prefill)
    participant UCX_P as UCX (Prefill)
    participant UCX_D as UCX (Decode)
    participant NIXL_D as NIXL (Decode)
    participant LMCache_D as LMCache (Decode)
    participant DecodeEngine as Decode vLLM Engine

    Note over Client, DecodeEngine: System Initialization Phase
    
    Client->>PrefillEngine: Initialize LLM with LMCacheConnectorV1
    PrefillEngine->>LMCache_P: Create LMCache instance
    LMCache_P->>NIXL_P: Initialize NIXL backend (role: sender)
    NIXL_P->>UCX_P: Initialize UCX with GPU Direct
    UCX_P->>UCX_P: Setup RDMA endpoints
    UCX_P->>UCX_P: Register GPU memory regions
    
    Note over UCX_P, UCX_D: Peer Discovery & Connection Setup
    
    Client->>DecodeEngine: Initialize LLM with LMCacheConnectorV1
    DecodeEngine->>LMCache_D: Create LMCache instance
    LMCache_D->>NIXL_D: Initialize NIXL backend (role: receiver)
    NIXL_D->>UCX_D: Initialize UCX with GPU Direct
    UCX_D->>UCX_D: Setup RDMA endpoints
    UCX_D->>UCX_D: Register GPU memory regions
    
    NIXL_P->>NIXL_D: Establish NIXL connection (host:55555)
    UCX_P-->>UCX_D: RDMA connection handshake
    NIXL_D-->>NIXL_P: Connection ACK
```

### Phase 2: Request Processing with KV Cache Transfer

```mermaid
sequenceDiagram
    participant Client
    participant PrefillEngine as Prefill Engine
    participant PrefillAttn as Prefill Attention
    participant LMCache_P as LMCache (Prefill)
    participant NIXL_P as NIXL (Prefill)
    participant UCX_P as UCX (Prefill)
    participant GPU_P as GPU Memory (Prefill)
    participant Network as RDMA Network
    participant GPU_D as GPU Memory (Decode)
    participant UCX_D as UCX (Decode)
    participant NIXL_D as NIXL (Decode)
    participant LMCache_D as LMCache (Decode)
    participant DecodeAttn as Decode Attention
    participant DecodeEngine as Decode Engine

    Note over Client, DecodeEngine: Request Processing Flow
    
    Client->>PrefillEngine: generate(prompts, max_tokens=1)
    PrefillEngine->>PrefillEngine: Start model forward pass
    
    loop For each transformer layer
        PrefillEngine->>PrefillAttn: forward(hidden_states, layer_idx)
        PrefillAttn->>PrefillAttn: compute_attention(q, k, v)
        PrefillAttn->>PrefillAttn: generate KV cache tensors
        PrefillAttn->>LMCache_P: store_kv_cache(req_id, layer_idx, kv_cache)
        
        LMCache_P->>NIXL_P: async_send(kv_cache_tensor, req_metadata)
        NIXL_P->>UCX_P: ucp_put_nbi(gpu_src_ptr, gpu_dst_ptr, size)
        UCX_P->>GPU_P: pin GPU memory region
        GPU_P->>Network: Direct GPU-to-GPU RDMA transfer
        Network->>GPU_D: KV cache data arrives
        UCX_D->>NIXL_D: transfer_complete_callback(req_id, layer_idx)
        NIXL_D->>LMCache_D: cache_received(req_id, layer_idx, kv_cache)
    end
    
    PrefillEngine-->>Client: Prefill complete (1 token)
    
    Note over Network: All KV caches transferred via NIXL
    
    Client->>DecodeEngine: generate(prompts, max_tokens=N)
    DecodeEngine->>DecodeEngine: Start model forward pass
    
    loop For each transformer layer
        DecodeEngine->>DecodeAttn: forward(hidden_states, layer_idx)
        DecodeAttn->>LMCache_D: retrieve_kv_cache(req_id, layer_idx)
        
        alt Cache Hit (already received)
            LMCache_D-->>DecodeAttn: kv_cache_tensor (immediate)
        else Cache Miss (still transferring)
            LMCache_D->>NIXL_D: wait_for_cache(req_id, layer_idx)
            NIXL_D->>NIXL_D: block until transfer complete
            NIXL_D-->>LMCache_D: kv_cache_tensor
            LMCache_D-->>DecodeAttn: kv_cache_tensor
        end
        
        DecodeAttn->>DecodeAttn: compute_attention(q, k_cached, v_cached)
    end
    
    DecodeEngine-->>Client: Full generation complete (N tokens)
```

## Layer-by-Layer KV Transfer Detail

```
Prefill Instance (GPU 0)                           Decode Instance (GPU 1)
┌─────────────────────────┐                       ┌─────────────────────────┐
│                         │     NIXL Transfer     │                         │
│  ┌─────────────────┐    │     (Layer 0 KV)      │    ┌─────────────────┐  │
│  │   Attention L0  │    │ ◄──────────────────── │    │   Attention L0  │  │
│  │   KV: [B,H,S,D] │◄───┤                       │    │   KV: [B,H,S,D] │  │
│  └─────────────────┘    │                       │    └─────────────────┘  │
│           ↓              │                       │             ↓           │
│  ┌─────────────────┐    │     NIXL Transfer     │    ┌─────────────────┐  │
│  │   Attention L1  │    │     (Layer 1 KV)      │    │   Attention L1  │  │
│  │   KV: [B,H,S,D] │◄───┤ ◄──────────────────── │    │   KV: [B,H,S,D] │  │
│  └─────────────────┘    │                       │    └─────────────────┘  │
│           ↓              │                       │             ↓           │
│         ...              │         ...           │           ...           │
│           ↓              │                       │             ↓           │
│  ┌─────────────────┐    │     NIXL Transfer     │    ┌─────────────────┐  │
│  │   Attention LN  │    │     (Layer N KV)      │    │   Attention LN  │  │
│  │   KV: [B,H,S,D] │◄───┤ ◄──────────────────── │    │   KV: [B,H,S,D] │  │
│  └─────────────────┘    │                       │    └─────────────────┘  │
│           ↓              │                       │             ↓           │
│      LM Head             │                       │        LM Head          │
│    (1 token out)         │                       │     (N tokens out)      │
└─────────────────────────┘                       └─────────────────────────┘

Each layer transfers:
- K cache: [batch_size, num_heads, seq_len, head_dim]  
- V cache: [batch_size, num_heads, seq_len, head_dim]
- Metadata: request_id, layer_id, tensor shapes
- Total per layer: ~2 * B * H * S * D * sizeof(dtype) bytes
```

## NIXL Internal Transfer Flow

### Sender (Prefill Instance) Detail
```
┌─────────────────────────────────────────────────────────────┐
│                    NIXL Sender Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LMCache.store_kv_cache(req_id, layer_id, kv_tensor)      │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              NIXL Backend                           │   │
│  │                                                     │   │
│  │  1. Serialize metadata (req_id, layer_id, shape)   │   │
│  │  2. Get GPU buffer from pool (1GB allocated)       │   │
│  │  3. Copy KV tensor to NIXL buffer                  │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                UCX Layer                            │   │
│  │                                                     │   │
│  │  ucp_put_nbi(                                      │   │
│  │    src_ptr  = nixl_gpu_buffer,                     │   │
│  │    dst_ptr  = remote_gpu_buffer,                   │   │
│  │    size     = kv_tensor_size,                      │   │
│  │    callback = transfer_complete                     │   │
│  │  )                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            GPU Direct / RDMA                        │   │
│  │                                                     │   │
│  │  - Pin source GPU memory                           │   │
│  │  - Setup RDMA descriptor                           │   │
│  │  - Direct GPU-to-GPU transfer                      │   │
│  │  - No CPU involvement                              │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  Asynchronous completion → callback → buffer cleanup       │
└─────────────────────────────────────────────────────────────┘
```

### Receiver (Decode Instance) Detail
```
┌─────────────────────────────────────────────────────────────┐
│                   NIXL Receiver Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  LMCache.retrieve_kv_cache(req_id, layer_id)              │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              NIXL Backend                           │   │
│  │                                                     │   │
│  │  1. Check local cache first                        │   │
│  │  2. If not found, wait for RDMA completion         │   │
│  │  3. Deserialize metadata when received             │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                UCX Layer                            │   │
│  │                                                     │   │
│  │  RDMA completion handler:                           │   │
│  │    1. Notification of data arrival                 │   │
│  │    2. Validate transfer integrity                  │   │
│  │    3. Signal NIXL completion                       │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            GPU Direct / RDMA                        │   │
│  │                                                     │   │
│  │  - Data arrives directly in GPU buffer             │   │
│  │  - No CPU memcpy required                          │   │
│  │  - Memory already in attention-ready format        │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  Return kv_tensor pointer → Attention Layer                │
└─────────────────────────────────────────────────────────────┘
```

## Configuration-Driven Connection Setup

### Prefiller Initialization
```yaml
# lmcache-prefiller-config.yaml processing
enable_nixl: True              # Enable NIXL backend
nixl_role: "sender"           # Set as KV producer
nixl_peer_host: "localhost"   # Target decode instance  
nixl_peer_port: 55555        # Communication port
nixl_buffer_size: 1073741824 # 1GB GPU buffer pool
nixl_buffer_device: "cuda"   # Use GPU memory
```

### Decoder Initialization  
```yaml
# lmcache-decoder-config.yaml processing
enable_nixl: True              # Enable NIXL backend
nixl_role: "receiver"         # Set as KV consumer
nixl_peer_host: "localhost"   # Source prefill instance
nixl_peer_port: 55555        # Communication port  
nixl_buffer_size: 1073741824 # 1GB GPU buffer pool
nixl_buffer_device: "cuda"   # Use GPU memory
```

## Error Handling and Recovery

```mermaid
sequenceDiagram
    participant LMCache_P as LMCache (Prefill)
    participant NIXL_P as NIXL (Prefill)
    participant Network as RDMA Network
    participant NIXL_D as NIXL (Decode)
    participant LMCache_D as LMCache (Decode)

    Note over LMCache_P, LMCache_D: Error Scenarios & Recovery
    
    LMCache_P->>NIXL_P: async_send(kv_cache)
    
    alt Network Failure
        NIXL_P->>Network: RDMA transfer
        Network-->>NIXL_P: Connection timeout
        NIXL_P->>NIXL_P: Retry with exponential backoff
        NIXL_P->>Network: Retry RDMA transfer
    else Buffer Exhaustion
        NIXL_P->>NIXL_P: GPU buffer pool full
        NIXL_P->>NIXL_P: Wait for buffer cleanup/GC
        NIXL_P->>NIXL_P: Retry allocation
    else Peer Unavailable
        NIXL_P->>NIXL_D: Connection attempt
        NIXL_D-->>NIXL_P: No response
        NIXL_P->>NIXL_P: Mark peer offline, fallback mode
    end
    
    LMCache_D->>NIXL_D: retrieve_kv_cache(req_id)
    
    alt Transfer Incomplete
        NIXL_D->>NIXL_D: Cache not yet received
        NIXL_D->>NIXL_D: Block with timeout
        NIXL_D->>Network: Check transfer status
    else Corrupted Data
        NIXL_D->>NIXL_D: Checksum validation failed
        NIXL_D->>NIXL_P: Request retransmission
    else Timeout
        NIXL_D->>NIXL_D: Transfer timeout exceeded
        NIXL_D->>LMCache_D: Return cache_miss, trigger local prefill
    end
```

## Performance Timeline

```
Transfer Timeline for Multi-Layer Model (32 layers):

Time:     0ms   10ms   20ms   30ms   40ms   50ms   60ms
         ├──────┼──────┼──────┼──────┼──────┼──────┼──────→

Prefill: ████████████████████████████│                    
Layer 0: ███│                        │                    
Layer 1:    ███│                     │                    
Layer 2:       ███│                  │                    
...            ...│                  │                    
Layer N:         ████│               │                    
                     │               │                    
NIXL:    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ (Overlapped)      
Transfer:║║║║║║║║║║║║║║║║║║║║║║║║║║║║║│                    
         ↑                           ↑                    
    Start Layer 0              All Transfers              
    Transfer                    Complete                   

Decode:                               │████████████████████
Wait:                                 │                    
Layer 0:                              │███│                
Layer 1:                              │   ███│             
...                                   │      ...│          
Layer N:                              │         ████│      

Key:
████ = Computation
░░░░ = NIXL Transfer (async)  
║║║║ = RDMA Transfer
───  = Waiting/Idle
```

## Memory Layout and Buffer Management

```
GPU Memory Layout in NIXL:

┌─────────────────────────────────────────────────────────┐
│                    GPU Memory Space                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  Model Weights  │  │     NIXL Buffer Pool        │   │
│  │  (Static)       │  │     (1GB allocated)         │   │
│  │                 │  │                             │   │
│  │  - Transformer  │  │  ┌───────────────────────┐   │   │
│  │    layers       │  │  │ Buffer 0 (256MB)      │   │   │
│  │  - Embeddings   │  │  │ - Layer 0-7 KV cache │   │   │
│  │  - LM head      │  │  └───────────────────────┘   │   │
│  └─────────────────┘  │  ┌───────────────────────┐   │   │
│                       │  │ Buffer 1 (256MB)      │   │   │
│  ┌─────────────────┐  │  │ - Layer 8-15 KV cache│   │   │
│  │ Active KV Cache │  │  └───────────────────────┘   │   │
│  │ (Current batch) │  │  ┌───────────────────────┐   │   │
│  │                 │  │  │ Buffer 2 (256MB)      │   │   │
│  │ - Key tensors   │  │  │ - Layer 16-23 KV     │   │   │
│  │ - Value tensors │  │  └───────────────────────┘   │   │
│  └─────────────────┘  │  ┌───────────────────────┐   │   │
│                       │  │ Buffer 3 (256MB)      │   │   │
│  ┌─────────────────┐  │  │ - Layer 24-31 KV     │   │   │
│  │ Workspace       │  │  └───────────────────────┘   │   │
│  │ (Activations)   │  └─────────────────────────────┘   │
│  └─────────────────┘                                    │
└─────────────────────────────────────────────────────────┘

Buffer Management:
- Ring buffer allocation for continuous transfer
- Garbage collection when buffers freed
- Direct pointer sharing (zero-copy within GPU)
- Memory-mapped RDMA regions for cross-GPU access
```

This comprehensive call flow shows how LMCache and NIXL work together to provide ultra-high-performance KV cache transfer between disaggregated vLLM instances, leveraging direct GPU-to-GPU RDMA for minimal latency and maximum throughput.