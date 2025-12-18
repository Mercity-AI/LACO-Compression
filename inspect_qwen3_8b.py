import torch
from transformers import AutoConfig, AutoModelForCausalLM

def inspect_qwen3_7b():
    """
    Inspect Qwen3-7B architecture WITHOUT downloading weights.
    No GPU needed, runs on CPU, almost instant.
    """
    
    model_name = "Qwen/Qwen3-8B"  # Qwen3 series (closest to 7B)
    # Alternative: "Qwen/Qwen2.5-7B" if Qwen3 not available
    
    print(f"\n{'='*60}")
    print(f"INSPECTING: {model_name}")
    print(f"{'='*60}")
    
    # ============ STEP 1: Load Config ============
    print("\n📋 Loading config (tiny ~1MB download)...")
    
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Qwen3 not found, trying Qwen2.5-7B...")
        model_name = "Qwen/Qwen2.5-7B"
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"\n{'─'*60}")
    print(f"1️⃣  BASIC INFO (from config)")
    print(f"{'─'*60}")
    print(f"   Model: {model_name}")
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Attention heads: {config.num_attention_heads}")
    print(f"   KV heads: {getattr(config, 'num_key_value_heads', 'same')}")
    print(f"   Intermediate size: {getattr(config, 'intermediate_size', 'N/A')}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Max position: {getattr(config, 'max_position_embeddings', 'N/A')}")
    
    # ============ STEP 2: Create Empty Model ============
    print(f"\n{'─'*60}")
    print(f"2️⃣  CREATING MODEL STRUCTURE (no weights, no memory)")
    print(f"{'─'*60}")
    
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    print(f"   ✅ Model created on meta device (0 bytes used)")
    
    # ============ STEP 3: Find Layers ============
    print(f"\n{'─'*60}")
    print(f"3️⃣  LAYERS PATH")
    print(f"{'─'*60}")
    
    # Try common paths
    layers = None
    layers_path = None
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        layers_path = "model.model.layers"
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        layers_path = "model.transformer.h"
    
    if layers:
        print(f"   ✅ Path: {layers_path}")
        print(f"   ✅ Number of layers: {len(layers)}")
        print(f"   → HIGHEST_LAY = {len(layers) - 1}")
    else:
        print(f"   ❌ Could not find layers automatically")
        print(f"   Printing model structure:")
        print(model)
        return
    
    # ============ STEP 4: Layer Components ============
    print(f"\n{'─'*60}")
    print(f"4️⃣  SINGLE LAYER COMPONENTS")
    print(f"{'─'*60}")
    
    layer = layers[0]
    print(f"   Layer type: {type(layer).__name__}")
    print(f"\n   Components:")
    for name, child in layer.named_children():
        print(f"   ├── {name}: {type(child).__name__}")
    
    # ============ STEP 5: MLP Structure ============
    print(f"\n{'─'*60}")
    print(f"5️⃣  MLP STRUCTURE")
    print(f"{'─'*60}")
    
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        print(f"   MLP type: {type(mlp).__name__}")
        print(f"\n   Weights:")
        
        for name, child in mlp.named_children():
            if hasattr(child, 'weight'):
                shape = list(child.weight.shape)
                print(f"   ├── mlp.{name}.weight: {shape}")
                if hasattr(child, 'bias') and child.bias is not None:
                    print(f"   │   └── mlp.{name}.bias: {list(child.bias.shape)}")
        
        # Determine MLP style
        print(f"\n   Style detection:")
        if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'):
            print(f"   ✅ LLaMA-style (separate gate_proj, up_proj, down_proj)")
        elif hasattr(mlp, 'gate_up_proj'):
            print(f"   ✅ Phi-style (fused gate_up_proj, down_proj)")
        elif hasattr(mlp, 'fc1'):
            print(f"   ✅ GPT-style (fc1, fc2)")
        else:
            print(f"   ❓ Unknown style")
    
    # ============ STEP 6: Attention Structure ============
    print(f"\n{'─'*60}")
    print(f"6️⃣  ATTENTION STRUCTURE")
    print(f"{'─'*60}")
    
    attn = None
    attn_name = None
    
    for name in ['self_attn', 'attention', 'attn', 'self_attention']:
        if hasattr(layer, name):
            attn = getattr(layer, name)
            attn_name = name
            break
    
    if attn:
        print(f"   Attention type: {type(attn).__name__}")
        print(f"   Attribute name: {attn_name}")
        print(f"\n   Weights:")
        
        for name, child in attn.named_children():
            if hasattr(child, 'weight'):
                shape = list(child.weight.shape)
                print(f"   ├── {attn_name}.{name}.weight: {shape}")
                if hasattr(child, 'bias') and child.bias is not None:
                    print(f"   │   └── {attn_name}.{name}.bias: {list(child.bias.shape)}")
        
        # Determine attention style
        print(f"\n   Style detection:")
        if hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
            print(f"   ✅ Separate Q/K/V projections (q_proj, k_proj, v_proj, o_proj)")
        elif hasattr(attn, 'qkv_proj'):
            print(f"   ✅ Fused QKV (qkv_proj, o_proj)")
        elif hasattr(attn, 'W_pack'):
            print(f"   ✅ Baichuan-style (W_pack, o_proj)")
        elif hasattr(attn, 'c_attn'):
            print(f"   ✅ GPT-2 style (c_attn, c_proj)")
        else:
            print(f"   ❓ Unknown style")
    
    # ============ STEP 7: Bias Check ============
    print(f"\n{'─'*60}")
    print(f"7️⃣  BIAS CHECK")
    print(f"{'─'*60}")
    
    has_bias = False
    for name, module in layer.named_modules():
        if hasattr(module, 'bias') and module.bias is not None:
            print(f"   ✅ {name} has bias: {list(module.bias.shape)}")
            has_bias = True
    
    if not has_bias:
        print(f"   ❌ No biases found (all layers use bias=False)")
    
    # ============ STEP 8: LayerNorm Check ============
    print(f"\n{'─'*60}")
    print(f"8️⃣  NORMALIZATION LAYERS")
    print(f"{'─'*60}")
    
    for name, child in layer.named_children():
        if 'norm' in name.lower() or 'ln' in name.lower():
            print(f"   ├── {name}: {type(child).__name__}")
            if hasattr(child, 'weight'):
                print(f"   │   └── weight: {list(child.weight.shape)}")
    
    # ============ STEP 9: Generate Merge Code ============
    print(f"\n{'─'*60}")
    print(f"9️⃣  GENERATED MERGE CODE")
    print(f"{'─'*60}")
    
    print(f"""
```python
from copy import deepcopy

def merge_layers_qwen3(model, merge_base_lay, merge_layer_num):
    \"\"\"
    Layer merging function for {model_name}
    Generated from architecture inspection
    \"\"\"
    
    layers = model.model.layers
    merge_layer_num = min(merge_layer_num, len(layers) - merge_base_lay - 1)
    
    model_copy = deepcopy(model)
    layers_copy = model_copy.model.layers
    
    for diff_lay in range(merge_base_lay + 1, merge_base_lay + 1 + merge_layer_num):
        """)
    
    # MLP weights
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
        if hasattr(mlp, 'gate_proj'):
            print(f"""        # MLP weights (LLaMA-style)
        layers_copy[merge_base_lay].mlp.gate_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.gate_proj.weight.data -
            layers_copy[merge_base_lay].mlp.gate_proj.weight.data
        )
        layers_copy[merge_base_lay].mlp.up_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.up_proj.weight.data -
            layers_copy[merge_base_lay].mlp.up_proj.weight.data
        )
        layers_copy[merge_base_lay].mlp.down_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.down_proj.weight.data -
            layers_copy[merge_base_lay].mlp.down_proj.weight.data
        )""")
    
    # Attention weights
    if attn:
        if hasattr(attn, 'q_proj'):
            print(f"""
        # Attention weights (separate Q/K/V)
        layers_copy[merge_base_lay].{attn_name}.q_proj.weight.data.add_(
            model.model.layers[diff_lay].{attn_name}.q_proj.weight.data -
            layers_copy[merge_base_lay].{attn_name}.q_proj.weight.data
        )
        layers_copy[merge_base_lay].{attn_name}.k_proj.weight.data.add_(
            model.model.layers[diff_lay].{attn_name}.k_proj.weight.data -
            layers_copy[merge_base_lay].{attn_name}.k_proj.weight.data
        )
        layers_copy[merge_base_lay].{attn_name}.v_proj.weight.data.add_(
            model.model.layers[diff_lay].{attn_name}.v_proj.weight.data -
            layers_copy[merge_base_lay].{attn_name}.v_proj.weight.data
        )
        layers_copy[merge_base_lay].{attn_name}.o_proj.weight.data.add_(
            model.model.layers[diff_lay].{attn_name}.o_proj.weight.data -
            layers_copy[merge_base_lay].{attn_name}.o_proj.weight.data
        )""")
        elif hasattr(attn, 'qkv_proj'):
            print(f"""
        # Attention weights (fused QKV)
        layers_copy[merge_base_lay].{attn_name}.qkv_proj.weight.data.add_(
            model.model.layers[diff_lay].{attn_name}.qkv_proj.weight.data -
            layers_copy[merge_base_lay].{attn_name}.qkv_proj.weight.data
        )
        layers_copy[merge_base_lay].{attn_name}.o_proj.weight.data.add_(
            model.model.layers[diff_lay].{attn_name}.o_proj.weight.data -
            layers_copy[merge_base_lay].{attn_name}.o_proj.weight.data
        )""")
    
    print(f"""
    # Delete merged layers (in reverse order)
    for diff_lay in range(merge_base_lay + merge_layer_num, merge_base_lay, -1):
        del layers_copy[diff_lay]
    
    return model_copy


# Configuration for {model_name}
HIGHEST_LAY = {len(layers) - 1}
LOWEST_LAY = 0
MERGE_LAYERS = 5  # Adjust based on testing
THRESHOLD = 0.5   # Adjust based on quality needs
INTERVAL = 1
```
""")
    
    # ============ SUMMARY ============
    print(f"\n{'='*60}")
    print(f"✅ INSPECTION COMPLETE")
    print(f"{'='*60}")
    print(f"""
SUMMARY FOR {model_name}:

📍 Layers path: {layers_path}
📊 Total layers: {len(layers)}
🔧 MLP style: {'LLaMA (gate/up/down)' if hasattr(layer.mlp, 'gate_proj') else 'Other'}
👁️ Attention style: {'Separate Q/K/V/O' if hasattr(attn, 'q_proj') else 'Other'}
⚖️ Has biases: {'Yes' if has_bias else 'No'}

You now have everything needed to write the merge code!
""")


# ============ RUN ============
if __name__ == "__main__":
    inspect_qwen3_7b()