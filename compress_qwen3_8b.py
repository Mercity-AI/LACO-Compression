import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
import numpy as np
import gc

# =====================================================
# QWEN3-8B CONFIGURATION
# =====================================================
MODEL_NAME = "Qwen/Qwen3-8B"
INTERVAL = 2
MERGE_LAYERS = 3
HIGHEST_LAY = 35
LOWEST_LAY = 10
THRESHOLD = 0.65

print("🚀 LaCo Pruning for A6000 (48GB) - CPU original, 2 GPU models")
print(f"Config: INTERVAL={INTERVAL}, MERGE_LAYERS={MERGE_LAYERS}")
print(f"        THRESHOLD={THRESHOLD}, Protected layers: 0-{LOWEST_LAY}")

# =====================================================
# LOAD ORIGINAL MODEL (CPU reference ONLY)
# =====================================================
print("\n📥 Loading ORIGINAL model (reference on CPU)...")
original_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Original model loaded on CPU: {len(original_model.model.layers)} layers")

# =====================================================
# LOAD WORKING + CANDIDATE MODELS (GPU)
# =====================================================
print("\n📥 Loading WORKING model (to be pruned) on GPU...")
model_to_prune = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
)
print(f"✅ Working model loaded on GPU: {len(model_to_prune.model.layers)} layers")

print("\n📥 Allocating CANDIDATE model (reused) on GPU...")
candidate_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
)
print(f"✅ Candidate model allocated on GPU: {len(candidate_model.model.layers)} layers")

if torch.cuda.is_available():
    print(f"\n💾 VRAM Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB / 48 GB")

# =====================================================
# MERGE FUNCTION (IN-PLACE on given model)
# =====================================================
def merge_layers_inplace(model, merge_base_lay, merge_layer_num):
    """Merge layers into base layer IN-PLACE on given model"""
    layers = model.model.layers
    merge_layer_num = min(merge_layer_num, len(layers) - merge_base_lay - 1)
    
    if merge_layer_num <= 0:
        print(f"    ⚠️ Nothing to merge at layer {merge_base_lay}")
        return 0
    
    print(f"    Merging layers {merge_base_lay+1} to {merge_base_lay+merge_layer_num} into {merge_base_lay}")

    # Merge weights from subsequent layers into base layer
    for diff_lay in range(merge_base_lay + 1, merge_base_lay + 1 + merge_layer_num):
        # MLP
        layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
            layers[diff_lay].mlp.gate_proj.weight.data -
            layers[merge_base_lay].mlp.gate_proj.weight.data
        )
        layers[merge_base_lay].mlp.up_proj.weight.data.add_(
            layers[diff_lay].mlp.up_proj.weight.data -
            layers[merge_base_lay].mlp.up_proj.weight.data
        )
        layers[merge_base_lay].mlp.down_proj.weight.data.add_(
            layers[diff_lay].mlp.down_proj.weight.data -
            layers[merge_base_lay].mlp.down_proj.weight.data
        )

        # Attention projections
        layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
            layers[diff_lay].self_attn.q_proj.weight.data -
            layers[merge_base_lay].self_attn.q_proj.weight.data
        )
        layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
            layers[diff_lay].self_attn.k_proj.weight.data -
            layers[merge_base_lay].self_attn.k_proj.weight.data
        )
        layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
            layers[diff_lay].self_attn.v_proj.weight.data -
            layers[merge_base_lay].self_attn.v_proj.weight.data
        )
        layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
            layers[diff_lay].self_attn.o_proj.weight.data -
            layers[merge_base_lay].self_attn.o_proj.weight.data
        )
        
        # Qwen3-specific: Q and K norms
        if hasattr(layers[merge_base_lay].self_attn, 'q_norm'):
            layers[merge_base_lay].self_attn.q_norm.weight.data.add_(
                layers[diff_lay].self_attn.q_norm.weight.data -
                layers[merge_base_lay].self_attn.q_norm.weight.data
            )
        if hasattr(layers[merge_base_lay].self_attn, 'k_norm'):
            layers[merge_base_lay].self_attn.k_norm.weight.data.add_(
                layers[diff_lay].self_attn.k_norm.weight.data -
                layers[merge_base_lay].self_attn.k_norm.weight.data
            )

    # Delete merged layers (in reverse order to maintain correct indices)
    for diff_lay in range(merge_base_lay + merge_layer_num, merge_base_lay, -1):
        del layers[diff_lay]
    
    # Update config
    model.config.num_hidden_layers = len(layers)
    
    return merge_layer_num


def sync_models(source_model, target_model):
    """
    Sync target_model to match source_model's architecture and weights.
    This handles the case where they have different numbers of layers.
    """
    source_layers = len(source_model.model.layers)
    target_layers = len(target_model.model.layers)
    
    if source_layers == target_layers:
        # Same size - simple state_dict copy
        target_model.load_state_dict(source_model.state_dict())
    elif source_layers < target_layers:
        # Source is smaller - need to delete layers from target first
        layers_to_remove = target_layers - source_layers
        
        # Delete extra layers from the end
        for _ in range(layers_to_remove):
            del target_model.model.layers[-1]
        
        # Now they're the same size, copy weights
        target_model.load_state_dict(source_model.state_dict())
        target_model.config.num_hidden_layers = source_layers
    else:
        # Source is larger - this shouldn't happen in our use case
        raise ValueError(f"Source ({source_layers}) has more layers than target ({target_layers})")


# =====================================================
# SIMILARITY CHECK (Original on CPU, candidate on GPU)
# =====================================================
def cal_last_hidden_sim(original_cpu_model, candidate_gpu_model, tokenizer, sents):
    sim_ls = []
    
    for s in sents:
        encoded_inputs = tokenizer(
            s,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Original on CPU
        orig_inputs = {k: v.to("cpu") for k, v in encoded_inputs.items()}
        with torch.no_grad():
            out1 = original_cpu_model(**orig_inputs, output_hidden_states=True)
        h1 = out1.hidden_states[-1].flatten().to(candidate_gpu_model.device)

        # Candidate on GPU
        cand_inputs = {k: v.to(candidate_gpu_model.device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            out2 = candidate_gpu_model(**cand_inputs, output_hidden_states=True)
        h2 = out2.hidden_states[-1].flatten()

        sim = torch.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0), dim=1)
        sim_ls.append(sim.item())
        
        # Cleanup
        del out1, out2, h1, h2

    avg_sim = float(np.mean(sim_ls))
    print(f"  Sims: {[f'{s:.3f}' for s in sim_ls]} → AVG: {avg_sim:.4f}")
    return avg_sim


# =====================================================
# CALIBRATION DATA
# =====================================================
calibration_sentences = [
    "Mouron is a commune in the Ardeche department in southern France.",
    "The 81st Mechanised Brigade is a mechanised brigade of the Romanian Land Force.",
    "There are 18 National Natural Landmarks in the U.S. state of Washington.",
    "Torreorgaz is a municipality in the province of Caceres, Extremadura, Spain.",
    "Copa Libertadores 1973 was won by defending champions Independiente of Argentina.",
    "The derivative of x^2 is 2x.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "Climate change refers to long-term shifts in temperatures and weather patterns.",
]

print(f"\n📝 Using {len(calibration_sentences)} calibration sentences")

# =====================================================
# MAIN PRUNING LOOP
# =====================================================
original_layers = len(original_model.model.layers)
lay = HIGHEST_LAY - MERGE_LAYERS
successful_merges = 0
rejected_merges = 0

print(f"\n🔄 Starting pruning loop...")
print(f"   Original layers: {original_layers}")
print(f"   Starting at layer: {lay}\n")

while lay >= LOWEST_LAY:
    current_layers = len(model_to_prune.model.layers)

    if lay + MERGE_LAYERS - 1 >= current_layers:
        print(f"⚠️  Not enough layers remaining. Stopping.")
        break

    print(f"📍 Layer {lay}/{current_layers} | Merges: {successful_merges} ✅ {rejected_merges} ❌")

    # 1) Sync candidate with current working model
    #    This handles mismatched layer counts properly
    sync_models(model_to_prune, candidate_model)
    print(f"    Synced candidate to {len(candidate_model.model.layers)} layers")

    # 2) Apply merge IN-PLACE on candidate_model (for testing)
    layers_merged = merge_layers_inplace(candidate_model, lay, MERGE_LAYERS - 1)
    
    if layers_merged == 0:
        print(f"  ⚠️ No layers merged, moving to next position")
        lay -= 1
        continue

    # 3) Compare ORIGINAL (CPU) vs CANDIDATE (GPU)
    sim_value = cal_last_hidden_sim(
        original_model,
        candidate_model,
        tokenizer,
        calibration_sentences,
    )

    if sim_value > THRESHOLD:
        print(f"  ✅ ACCEPT sim={sim_value:.4f} > {THRESHOLD}")
        print(f"     Removed {layers_merged} layers")
        
        # ACCEPT: Apply the same merge to model_to_prune
        merge_layers_inplace(model_to_prune, lay, MERGE_LAYERS - 1)
        
        successful_merges += 1
        lay -= INTERVAL

        # Adjust if we've shrunk past current position
        new_layer_count = len(model_to_prune.model.layers)
        if lay >= new_layer_count:
            lay = new_layer_count - 1 - MERGE_LAYERS
            
        print(f"     Working model now has {new_layer_count} layers\n")
    else:
        print(f"  ❌ REJECT sim={sim_value:.4f} < {THRESHOLD}\n")
        rejected_merges += 1
        lay -= 1

    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        print(f"  💾 VRAM: {vram_used:.2f} GB")

# =====================================================
# FINALIZE
# =====================================================
final_layers = len(model_to_prune.model.layers)
model_to_prune.config.num_hidden_layers = final_layers

print(f"\n{'='*60}")
print(f"🎉 PRUNING COMPLETE!")
print(f"{'='*60}")
print(f"📊 Original layers:    {original_layers}")
print(f"📊 Final layers:       {final_layers}")
print(f"📊 Layers removed:     {original_layers - final_layers}")
print(f"📊 Compression:        {100 * (1 - final_layers / original_layers):.1f}%")
print(f"📊 Successful merges:  {successful_merges}")
print(f"📊 Rejected merges:    {rejected_merges}")
print(f"{'='*60}\n")

# =====================================================
# QUALITY CHECK
# =====================================================
def quick_quality_check(orig_model, pruned_model, tokenizer):
    """Compare outputs on sample prompts"""
    test_prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in simple terms.",
        "Write a Python function to reverse a string."
    ]
    
    print("🔍 Quality Check (Sample Generations):\n")
    
    for prompt in test_prompts:
        # Original model (on CPU)
        inputs_cpu = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            orig_out = orig_model.generate(**inputs_cpu, max_new_tokens=50, do_sample=False)
        orig_text = tokenizer.decode(orig_out[0], skip_special_tokens=True)
        
        # Pruned model (on GPU)
        inputs_gpu = tokenizer(prompt, return_tensors="pt").to(pruned_model.device)
        with torch.no_grad():
            pruned_out = pruned_model.generate(**inputs_gpu, max_new_tokens=50, do_sample=False)
        pruned_text = tokenizer.decode(pruned_out[0], skip_special_tokens=True)
        
        print(f"Prompt: {prompt}")
        print(f"Original: {orig_text[:150]}...")
        print(f"Pruned:   {pruned_text[:150]}...")
        print()

quick_quality_check(original_model, model_to_prune, tokenizer)

# =====================================================
# SAVE MODEL
# =====================================================
output_path = "qwen3-8b-laco-pruned"
print(f"\n💾 Saving pruned model to {output_path}...")
model_to_prune.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)
print(f"✅ Model saved successfully!\n")