"""Ablation pipeline: SemanticFoveatedPipeline with extra Stage-2 / Full-HR knobs.

Adds (all defaults reproduce the original method):
  * fp16 DiT and on-disk prompt-embedding cache (lets the 4B DiT run on a 16 GB GPU without the text encoder)
  * Stage 2: custom init images (e.g. Real-ESRGAN instead of bicubic), correlated LR/HR noise,
    schedule shift length, CFG truncation (no CFG on the last k steps), residual caching (skip the whole
    transformer stack on chosen steps and reuse the residual from the previous computed step),
    returns both resolution decodes for a seam metric
  * Full HR: the same CFG-truncation and caching knobs, so speed tricks are applied to the baseline too
"""

import hashlib
import json
import math
import os
import time
from typing import Dict, Optional, Sequence

from einops import rearrange
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from diffsynth.core import ModelConfig

from ..diffsynth_fov.pipeline import Flux2FoveatedImagePipeline, _gaussian_blur_mask_2d
from .localizer import CLIPSegLocalizer
from .pipeline import DEFAULT_NEGATIVE_PROMPT, SemanticFoveatedPipeline, set_flux2_subtrajectory_timesteps

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
FAST_DIT = "~/models/fov_dit_fp16.safetensors"


def infer_dit_config(sd):
    """Same inference as Flux2FoveatedImagePipeline._convert_to_foveated_dit."""
    inner = sd["x_embedder.weight"].shape[0]
    nl = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks.") and k.endswith(".attn.to_q.weight"))
    ns = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("single_transformer_blocks.") and k.endswith(".attn.to_qkv_mlp_proj.weight"))
    return dict(in_channels=sd["x_embedder.weight"].shape[1], num_layers=nl, num_single_layers=ns,
                attention_head_dim=128, num_attention_heads=inner // 128,
                joint_attention_dim=sd["context_embedder.weight"].shape[1],
                guidance_embeds="time_guidance_embed.guidance_embedder.linear_1.weight" in sd)


def prompt_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def tokens_to_spatial(tok: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[B, h*w, C] packed 2x2 tokens -> [B, C/4, 2h, 2w] latent pixels (same order as _downsample_latents)."""
    b, _, c = tok.shape
    x = tok.view(b, h, w, c // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c // 4, h * 2, w * 2)


def spatial_to_tokens(x: torch.Tensor) -> torch.Tensor:
    b, c, hh, ww = x.shape
    h, w = hh // 2, ww // 2
    x = x.view(b, c, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, h * w, c * 4)


class AblationPipeline(SemanticFoveatedPipeline):
    def __init__(self, device: str = "cuda:0", torch_dtype=torch.float16, embed_cache_dir: str = "cache/embeds"):
        self.device = device
        self.low_vram = False
        self.torch_dtype = torch_dtype
        self.model_id = MODEL_ID
        self.offload_text_encoder = True
        self.embed_cache_dir = embed_cache_dir
        fast = os.path.expanduser(FAST_DIT)
        vae_cfg = ModelConfig(model_id=MODEL_ID, origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
        if os.path.exists(fast):
            # Fast path: foveated fp16 DiT saved once by the slow path; built on meta, loaded from CPU in one copy
            print(f"[AblationPipeline] fast-loading foveated DiT from {fast}", flush=True)
            self.pipe = Flux2FoveatedImagePipeline.from_pretrained(
                torch_dtype=torch_dtype, device="cpu", model_configs=[vae_cfg], tokenizer_config=None)
            from safetensors.torch import load_file
            from ..diffsynth_fov.dit import Flux2DiTFoveated
            cfg = json.load(open(fast.replace(".safetensors", ".json")))
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch_dtype)
            with torch.device("meta"):
                dit = Flux2DiTFoveated(**cfg)
            torch.set_default_dtype(prev)
            dit = dit.to_empty(device=device)
            sd = load_file(fast, device="cpu")
            dit.load_state_dict(sd)
            del sd
            self.pipe.dit = dit.eval()
        else:
            print(f"[AblationPipeline] slow-loading DiT + VAE ({torch_dtype}); will cache to {fast}", flush=True)
            self.pipe = Flux2FoveatedImagePipeline.from_pretrained(
                torch_dtype=torch_dtype, device="cpu",
                model_configs=[ModelConfig(model_id=MODEL_ID, origin_file_pattern="transformer/*.safetensors"), vae_cfg],
                tokenizer_config=None)
            from safetensors.torch import save_file
            sd = {k: v.to(torch_dtype).contiguous() for k, v in self.pipe.dit.state_dict().items()}
            json.dump(infer_dit_config(sd), open(fast.replace(".safetensors", ".json"), "w"))
            save_file(sd, fast)
            del sd
        self.pipe.device = device
        self.pipe.dit.to(device)
        self.pipe.vae.to(device)
        self.localizer = CLIPSegLocalizer(device=device)
        self._lora_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        self._active_lora_mode = None
        self._dit_modules = dict(self.pipe.dit.named_modules())
        self._sr_model = None

    # ------------------------------------------------------------------ prompts
    def _load_embed(self, text: str) -> dict:
        path = os.path.join(self.embed_cache_dir, prompt_key(text) + ".pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No cached embedding for {text!r} ({path}); run scripts/encode_prompts.py")
        d = torch.load(path, map_location="cpu")
        emb = d["prompt_embeds"].float()
        if emb.abs().max() > 6.0e4:
            emb = emb.clamp(-6.0e4, 6.0e4)
        return {"prompt_embeds": emb.to(self.device, self.torch_dtype), "text_ids": d["text_ids"].to(self.device)}

    def encode_prompt_cached(self, prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT, cfg_scale=4.0):
        return self._load_embed(prompt), self._load_embed(negative_prompt)

    # ------------------------------------------------------------------ helpers
    def sr_upscale(self, img: Image.Image) -> Image.Image:
        """Real-ESRGAN x4plus (via spandrel)."""
        if self._sr_model is None:
            from spandrel import ModelLoader
            path = os.path.expanduser("~/models/esrgan/RealESRGAN_x4plus.pth")
            self._sr_model = ModelLoader().load_from_file(path).model.to(self.device).half().eval()
        x = torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1)[None].float().div(255)
        with torch.no_grad():
            y = self._sr_model(x.to(self.device).half()).float().clamp(0, 1)
        return Image.fromarray((y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))

    def _denoise(self, latents, inputs_shared, inputs_posi, inputs_nega, cfg_scale, cfg_off_last=0,
                 reuse_steps: Sequence[int] = ()):
        dit = self.pipe.dit
        steps = list(self.pipe.scheduler.timesteps)
        n = len(steps)
        use_cache = len(reuse_steps) > 0
        dit._resid_cache = {}
        nfe = 0
        try:
            for i, t in enumerate(steps):
                tv = t.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                inputs_shared["latents"] = latents
                mode = None
                if use_cache:
                    mode = "reuse" if i in reuse_steps else "store"
                cfg_i = 1.0 if i >= n - cfg_off_last else cfg_scale
                dit._resid_mode, dit._resid_key = mode, "posi"
                pos = self.pipe.model_fn(dit=dit, **inputs_posi, **inputs_shared, timestep=tv)
                nfe += mode != "reuse"
                if cfg_i != 1.0:
                    dit._resid_key = "nega"
                    neg = self.pipe.model_fn(dit=dit, **inputs_nega, **inputs_shared, timestep=tv)
                    nfe += mode != "reuse"
                    pred = neg + cfg_i * (pos - neg)
                else:
                    pred = pos
                pred = pred[:, : latents.size(1), :]
                latents = self.pipe.step(self.pipe.scheduler, progress_id=i, noise_pred=pred, latents=latents)
        finally:
            dit._resid_mode = None
            dit._resid_cache = {}
        return latents, nfe

    # ------------------------------------------------------------------ Stage 2
    @torch.no_grad()
    def stage2_ext(
        self,
        cached_prompts,
        draft_low: Image.Image,
        mask_tok: Optional[torch.Tensor],
        mask_full: Optional[torch.Tensor] = None,
        seed: int = 42,
        steps: int = 7,
        t_switch: float = 0.95,
        lora_mode: Optional[str] = "random",
        init_hr: Optional[Image.Image] = None,
        init_lr: Optional[Image.Image] = None,
        noise_mode: str = "independent",
        shift_len: Optional[int] = None,
        cfg_scale: float = 4.0,
        cfg_off_last: int = 0,
        reuse_steps: Sequence[int] = (),
        H: int = 1024,
        W: int = 1024,
        lr_factor: int = 2,
        return_parts: bool = False,
    ):
        """Returns (image, seconds, info). `mask_tok` None => every token full-res."""
        self.set_lora_mode(lora_mode)
        inputs_posi, inputs_nega = cached_prompts
        lh, lw = H // 16, W // 16
        if mask_tok is not None and float(mask_tok.mean()) >= 0.999:
            mask_tok = None  # all-HR: plain attention path
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        if init_hr is None:
            init_hr = draft_low.resize((W, H), Image.Resampling.BICUBIC)
        x = self.pipe.preprocess_image(init_hr).to(self.device, self.torch_dtype)
        z0_high = rearrange(self.pipe.vae.encode(x), "B C H W -> B (H W) C")
        z0 = z0_high
        if mask_tok is not None:
            if init_lr is None:
                init_lr = draft_low.resize((W // lr_factor, H // lr_factor), Image.Resampling.BICUBIC)
            xl = self.pipe.preprocess_image(init_lr).to(self.device, self.torch_dtype)
            z0_low = rearrange(self.pipe.vae.encode(xl), "B C H W -> B (H W) C")
            z0 = self.pipe._downsample_latents(z0_high, lh, lw, foveation_mask=mask_tok,
                                               latents_downsampled=z0_low, lr_factor=lr_factor)

        set_flux2_subtrajectory_timesteps(self.pipe.scheduler, num_inference_steps=steps, t_switch=t_switch,
                                          dynamic_shift_len=shift_len or lh * lw)
        noise_high = self.pipe.generate_noise((1, 128, lh, lw), seed=seed + 1000, rand_device=self.device,
                                              rand_torch_dtype=self.torch_dtype)
        noise_high = noise_high.reshape(1, 128, lh * lw).permute(0, 2, 1)
        z1 = noise_high
        if mask_tok is not None:
            if noise_mode == "correlated":
                sp = tokens_to_spatial(noise_high.float(), lh, lw)
                noise_low = spatial_to_tokens(F.avg_pool2d(sp, 2) * 2.0).to(self.torch_dtype)
            else:
                noise_low = self.pipe.generate_noise((1, 128, lh // lr_factor, lw // lr_factor), seed=seed + 1000,
                                                     rand_device=self.device, rand_torch_dtype=self.torch_dtype)
                noise_low = noise_low.reshape(1, 128, -1).permute(0, 2, 1)
            z1 = self.pipe._downsample_latents(noise_high, lh, lw, foveation_mask=mask_tok,
                                               latents_downsampled=noise_low, lr_factor=lr_factor)
        latents = self.pipe.scheduler.add_noise(z0, z1, timestep=self.pipe.scheduler.timesteps[0])

        ids, rmask, rmask_tl = self.pipe._prepare_latent_image_ids_foveation(
            batch_size=1, height=lh, width=lw, device=self.device, dtype=self.torch_dtype,
            foveation_mask=mask_tok, lr_factor=lr_factor)
        if ids.ndim == 2:
            ids = ids.unsqueeze(0)
        shared = {"cfg_scale": cfg_scale, "embedded_guidance": 4.0, "height": H, "width": W, "latents": latents,
                  "image_ids": ids, "resolution_mask": rmask, "resolution_mask_top_left": rmask_tl,
                  "lr_downsample_factor": lr_factor}
        latents, nfe = self._denoise(latents, shared, inputs_posi, inputs_nega, cfg_scale, cfg_off_last, reuse_steps)

        info = {"tokens": int(latents.shape[1]), "nfe": nfe}
        if mask_tok is not None:
            lr_lat = self.pipe._downsample_remaining_high_res(latents, lh, lw, foveation_mask=mask_tok, lr_factor=lr_factor)
            hd, wd = lh // lr_factor, lw // lr_factor
            lr_img = self.pipe.vae.decode(lr_lat.view(1, hd, wd, -1).permute(0, 3, 1, 2))
            hr_lat = self.pipe._upsample_latents(latents, lh, lw, foveation_mask=mask_tok, lr_factor=lr_factor)
            hr_img = self.pipe.vae.decode(hr_lat.view(1, lh, lw, -1).permute(0, 3, 1, 2))
            lr_img = F.interpolate(lr_img, size=(H, W), mode="bicubic", align_corners=False)
            if mask_full is not None:
                m = mask_full[None, None].float()
            else:
                m = F.interpolate(mask_tok[None, None].float(), size=(H, W), mode="nearest")
            m = m.to(self.device, latents.dtype)
            m = _gaussian_blur_mask_2d(m, sigma_pixels=H / lh, device=self.device, dtype=latents.dtype).clamp(0, 1)
            image = self.pipe.vae_output_to_image(lr_img * (1 - m) + hr_img * m)
            if return_parts:
                info["hr_decode"] = self.pipe.vae_output_to_image(hr_img)
                info["lr_decode"] = self.pipe.vae_output_to_image(lr_img)
        else:
            image = self.pipe.vae_output_to_image(self.pipe.vae.decode(
                rearrange(latents, "B (H W) C -> B C H W", H=lh, W=lw)))
        torch.cuda.synchronize(self.device)
        return image, time.perf_counter() - t0, info

    # ------------------------------------------------------------------ Full HR
    @torch.no_grad()
    def full_hr_ext(self, cached_prompts, seed, steps=28, cfg_scale=4.0, cfg_off_last=0, reuse_steps=(),
                    lora_mode="no_fov", H=1024, W=1024):
        self.set_lora_mode(lora_mode)
        inputs_posi, inputs_nega = cached_prompts
        lh, lw = H // 16, W // 16
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        set_flux2_subtrajectory_timesteps(self.pipe.scheduler, num_inference_steps=steps, t_switch=1.0,
                                          dynamic_shift_len=lh * lw)
        noise = self.pipe.generate_noise((1, 128, lh, lw), seed=seed, rand_device=self.device,
                                         rand_torch_dtype=self.torch_dtype)
        latents = noise.reshape(1, 128, lh * lw).permute(0, 2, 1)
        ids, _, _ = self.pipe._prepare_latent_image_ids_foveation(batch_size=1, height=lh, width=lw,
                                                                 device=self.device, dtype=self.torch_dtype)
        ids = ids.unsqueeze(0) if ids.ndim == 2 else ids
        shared = {"cfg_scale": cfg_scale, "embedded_guidance": 4.0, "height": H, "width": W, "latents": latents,
                  "image_ids": ids, "resolution_mask": None, "resolution_mask_top_left": None,
                  "lr_downsample_factor": 2}
        latents, nfe = self._denoise(latents, shared, inputs_posi, inputs_nega, cfg_scale, cfg_off_last, reuse_steps)
        image = self.pipe.vae_output_to_image(self.pipe.vae.decode(
            rearrange(latents, "B (H W) C -> B C H W", H=lh, W=lw)))
        torch.cuda.synchronize(self.device)
        return image, time.perf_counter() - t0, {"tokens": lh * lw, "nfe": nfe}
