**SenseNova-Loop-9B** is an autoregressive multimodal decoder model (accepting text and images, generating text) that achieves greater effective reasoning depth without inflating stored parameter count. It stores approximately **9B total parameters** while reaching roughly **14.6B inference-equivalent parameters** by combining a **shared-weight looped middle block**, **hybrid linear/full attention**, and **two-stream manifold-constrained hyper-connections (mHC)**, supporting context lengths up to **512K tokens**.

---

### Core Specifications

| Specification | Value |
| --- | --- |
| **Architecture** | Autoregressive multimodal decoder; looped block with two-stream mHC |
| **Parameter Scale** | ~9B total stored parameters; ~14.6B inference-equivalent parameters |
| **Modalities** | Text + Image input $\rightarrow$ Text output |
| **Decoder Layers** | 32 distinct layers; **56 layer executions** per forward pass |
| **Looped Block & Passes** | Layers `[4, 28)` (24 layers) executed **2 times** with shared weights |
| **Hidden / FFN Dimension** | 4,096 hidden dimension; 12,288 feed-forward dimension |
| **Vocabulary Size** | 248,320 |
| **Full-Attention Heads** | 16 query heads; 4 key-value (KV) heads |
| **mHC Configuration** | 2 residual streams; 20 Sinkhorn iterations |
| **Context Window** | Up to 512K tokens (trained up to 262,144 / 256K tokens) |
| **Precision** | BF16 backbone; FP32 mHC routing parameters (also released in 4-bit AWQ and GGUF formats) |

---

### 1. Looped Decoder (Weight-Shared Depth)

Instead of passing representations through each decoder layer once, SenseNova-Loop-9B executes a contiguous middle slice twice with shared weights:

* **Entry layers (`0–3`):** Process the initial input representations once.
* **Looped block (`4–27`):** A 24-layer block ($B$) runs sequentially twice—conceptually $B(B(h))$—where the second pass takes the output of the first pass as its input.
* **Exit layers (`28–31` & head):** Finalize representations and produce output tokens.

This yields $4 + (24 \times 2) + 4 = 56$ decoder-layer executions per forward pass while storing only 32 sets of decoder-layer weights.

---

### 2. Hybrid Attention Pattern

To balance long-context efficiency with exact token-to-token recall, the decoder interleaves **linear attention** and **full attention**:

* **4-layer cadence:** Every group of four decoder layers consists of **three linear-attention layers** followed by **one full-attention layer** (with 16 attention heads and 4 KV heads in full-attention layers).
* **Loop alignment:** The loop boundaries (`[4, 28)`) align with complete four-layer groups so the alternating linear/full attention structure stays consistent across both passes of the repeated block.

---

### 3. Two-Stream Manifold-Constrained Hyper-Connections (mHC)

Reusing a layer block across multiple passes can risk signal degradation or instability if residual pathways are not carefully regulated. SenseNova-Loop-9B addresses this with **two-stream manifold-constrained hyper-connections (mHC)**:

* **Dual residual streams:** Information flows across **2 residual streams** in the looped section rather than a single unbranched residual path.
* **Learned routing & manifold constraint:** Learned mappings route and combine information between the two streams, using **20 Sinkhorn iterations** to constrain routing matrices and keep information flow stable across repeated passes.
* **Mixed precision:** While the backbone runs in BF16, the mHC routing parameters are kept in **FP32** to preserve numerical stability during routing and multi-pass execution.

---

### 4. Long-Context & Inference Characteristics

* **Long-horizon multimodal & agentic focus:** Trained across long sequences (up to 262,144 tokens in documented recipes, supporting up to 512K context) combining text, images, code, and tool/agent trajectories.
* **Memory footprint:** Because weights are shared across the looped block, 4-bit weight quantization takes roughly **4.9 GB** for weights, and quantized KV caches keep long-context memory modest (for example, ~15 GiB at 256K context with 4-bit weights and 8-bit KV cache, or ~12 GiB with 4-bit KV cache).
* **Acceleration status:** Multi-Token Prediction (MTP) is supported for inference acceleration, while additional acceleration methods (such as Dspark and Dflash2) have been listed as in training/development.