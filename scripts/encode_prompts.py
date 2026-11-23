"""Encode every benchmark prompt (+ the negative prompt) once with the Qwen3 text encoder and cache to disk.

Lets the experiment processes run the 4B DiT in fp16 on a 16 GB GPU without the text encoder resident.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffsynth.core import ModelConfig
from src.diffsynth_fov.pipeline import Flux2FoveatedImagePipeline
from src.semantic_foveation.ablation import MODEL_ID, prompt_key
from src.semantic_foveation.benchmark import PROMPT_SETS
from src.semantic_foveation.pipeline import DEFAULT_NEGATIVE_PROMPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else "cache/embeds"
os.makedirs(out_dir, exist_ok=True)
pipe = Flux2FoveatedImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cpu",
    model_configs=[ModelConfig(model_id=MODEL_ID, origin_file_pattern="text_encoder/*.safetensors")],
    tokenizer_config=ModelConfig(model_id=MODEL_ID, origin_file_pattern="tokenizer/"),
)
pipe.device = "cuda:0"
pipe.text_encoder_qwen3.to("cuda:0")
unit = next(u for u in pipe.units if u.__class__.__name__ == "Flux2FoveatedUnit_Qwen3PromptEmbedder")
texts = [DEFAULT_NEGATIVE_PROMPT] + [p["prompt"] for s in PROMPT_SETS.values() for p in s]
for t in texts:
    path = os.path.join(out_dir, prompt_key(t) + ".pt")
    if os.path.exists(path):
        continue
    with torch.no_grad():
        out = unit.process(pipe, prompt=t)
    emb = out["prompt_embeds"]
    print(f"{prompt_key(t)} max|emb|={emb.float().abs().max().item():.1f} {t[:60]}", flush=True)
    torch.save({"prompt_embeds": emb.cpu(), "text_ids": out["text_ids"].cpu(), "text": t}, path)
print("ENCODE_DONE", len(texts))
