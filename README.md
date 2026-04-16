## LaCo Pruning for Qwen3-8B

READ OUR FULL RESEARCH LOG AND FINDINGS HERE: [LaCO Research Log](https://www.mercity.ai/blog-post/laco-layer-pruning-for-qwen3-8b-our-research-log/)

This repo contains a script to **prune/compress the Qwen3-8B model using LaCo-style layer merging**, targeting an A6000 (48 GB) GPU setup.  
It is a practical implementation inspired by **LaCo: Large Language Model Pruning via Layer Collapse** \([https://arxiv.org/abs/2507.02279](https://arxiv.org/pdf/2402.11187))\].

For reference, see also the **official LaCo implementation for LLMs**:  
- GitHub: [`yangyifei729/LaCo`](https://github.com/yangyifei729/LaCo)

The main script is `compress_qwen3_8b.py`, which:

- **Loads three copies of the model**
  - **Original model (CPU)**: used as a fixed reference for similarity comparison.
  - **Working model (GPU)**: progressively pruned in-place.
  - **Candidate model (GPU)**: used to test a proposed merge before applying it to the working model.
- **Performs structured layer merging**
  - Merges multiple transformer layers into a base layer (`merge_layers_inplace`).
  - Supports Qwen3-specific components (e.g., `q_norm`, `k_norm`).
  - Updates `num_hidden_layers` after merges.
- **Uses last-layer hidden state similarity as a pruning criterion**
  - Computes cosine similarity between original (CPU) and candidate (GPU) hidden states on a small set of **calibration sentences**.
  - If similarity is above a configurable **threshold**, the merge is **accepted** and applied to the working model.
  - Otherwise, the merge is **rejected** and the script moves to the next layer position.
- **Runs an optional quality check**
  - Generates outputs from both the original and pruned models for a few sample prompts.
- **Saves the pruned model**
  - Outputs to `qwen3-8b-laco-pruned/`.

### Current Configuration

Key hyperparameters in `compress_qwen3_8b.py`:

- **MODEL_NAME**: `Qwen/Qwen3-8B`
- **INTERVAL**: `2` (step size when moving the merge base layer after a successful merge)
- **MERGE_LAYERS**: `3` (number of layers involved in each merge block)
- **HIGHEST_LAY**: `35`
- **LOWEST_LAY**: `10` (layers below this index are protected from pruning)
- **THRESHOLD**: `0.65` (minimum cosine similarity to accept a merge)

These values reflect the **current configuration** and are expected to change as you tune the trade‑off between compression and quality.

### Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Current requirements:

- `torch`
- `transformers`
- `accelerate`
- `numpy`

You also need:

- **NVIDIA GPU with ~48 GB VRAM** (script is tuned for an A6000, loading two GPU models plus overhead).
- Recent CUDA-compatible PyTorch build.

### How to Run

1. **Log in to Hugging Face Hub** (if the Qwen model requires auth):

```bash
huggingface-cli login
```

2. **Run the pruning script**:

```bash
python compress_qwen3_8b.py
```

The script will:

- Print configuration and VRAM usage.
- Iterate over layers from `HIGHEST_LAY` down to `LOWEST_LAY`, attempting merges.
- Report accepted/rejected merges and running VRAM stats.
- Show a brief quality check.
- Save the pruned model to `qwen3-8b-laco-pruned/`.

### Notes and Limitations (Current Stage)

- This is **research code**; error handling and logging are minimal.
- The script assumes the Qwen3-8B architecture with:
  - `model.model.layers`
  - `layer.mlp.{gate_proj, up_proj, down_proj}`
  - `layer.self_attn.{q_proj, k_proj, v_proj, o_proj}` and optional `{q_norm, k_norm}`.
- Hyperparameters (`INTERVAL`, `MERGE_LAYERS`, `THRESHOLD`, layer range) are **not yet exposed via CLI**; edit them directly in `compress_qwen3_8b.py`.
- For smaller GPUs, you will likely need to:
  - Reduce the number of simultaneously loaded models, or
  - Switch to 4-bit/8-bit loading (e.g., `load_in_4bit`) and update the script accordingly.


