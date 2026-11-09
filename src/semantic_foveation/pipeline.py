"""Two-Stage Semantic Foveated Diffusion Pipeline.

Implements the full method described in method.tex:
  - Stage 1: Coarse low-resolution draft generation on an (h/2 x w/2) token grid
             + open-vocabulary semantic mask inference via CLIPSeg.
  - Stage 2: Selective high-resolution re-encoding of masked blocks, merging with
             Stage-1 peripheral low-resolution latents, partial flow-matching
             re-noising at t_switch, mixed-resolution denoising with CRPA RoPE
             attention and Foveated LoRA, and Gaussian-smoothed pixel-space blending.
"""

from dataclasses import dataclass
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

from einops import rearrange
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline

from ..diffsynth_fov.pipeline import Flux2FoveatedImagePipeline, _gaussian_blur_mask_2d
from ..masks.shapes import create_foveation_mask, create_foveation_mask_full_res
from .localizer import CLIPSegLocalizer, SemanticMaskOutput


HF_LORA_REPO = "bchao1/foveated-diffusion"
IMAGE_LORA_FILES = {
    "random": "image/fov_random.safetensors",
    "bbox": "image/fov_bbox.safetensors",
    "saliency": "image/fov_saliency.safetensors",
    "no_fov": "image/no_fov.safetensors",
}


@dataclass
class Stage1Output:
    """Outputs from Stage 1 low-resolution draft generation."""
    draft_image_low: Image.Image       # Coarse RGB draft at (H/2, W/2)
    draft_image_upsampled: Image.Image # Bicubically upsampled RGB draft at (H, W)
    latents_low: torch.Tensor          # Clean LR latents z_0^low of shape [1, (h/2)*(w/2), C]
    stage1_time_s: float               # Wall-clock latency for Stage 1 denoising + decode


@dataclass
class SemanticFoveatedOutput:
    """Full output of the 2-stage Semantic Foveated Diffusion pipeline."""
    image: Image.Image                 # Final blended H x W RGB image
    stage1: Stage1Output               # Stage 1 draft outputs
    mask_output: SemanticMaskOutput    # CLIPSeg heatmap and block-aligned mask M
    stage2_time_s: float               # Wall-clock latency for Stage 2 re-encode + denoise + decode
    total_time_s: float                # End-to-end latency (Stage 1 + Mask Inference + Stage 2)


def set_flux2_subtrajectory_timesteps(
    scheduler: FlowMatchScheduler,
    num_inference_steps: int,
    t_switch: float = 1.0,
    dynamic_shift_len: int = 4096,
):
    """Configure FLUX.2 flow-matching schedule so sigmas[0] equals the exact noise level t_switch.

    Upstream `set_timesteps_flux2` applies the exponential time-shift exp(mu)/(exp(mu) + (1/u - 1))
    after linearly scaling `denoising_strength`, which warps `denoising_strength=0.5` into
    `sigmas[0] ~= 0.90`. Here we invert the FLUX.2 shift at `t_switch` so that:
      z_{t_switch} = (1 - t_switch) * z_0 + t_switch * z_1
    starts at the exact requested noise fraction `sigmas[0] = t_switch`.
    """
    sigma_min_shifted = 1.0 / max(20, num_inference_steps)
    mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, max(20, num_inference_steps))
    exp_mu = math.exp(mu)

    if t_switch >= 0.999:
        u_start = 1.0
    else:
        t_clamped = max(sigma_min_shifted + 1e-4, min(0.999, float(t_switch)))
        # Invert sigma = exp_mu / (exp_mu + (1/u - 1))
        u_start = 1.0 / (1.0 + exp_mu * (1.0 / t_clamped - 1.0))

    u_min = 1.0 / (1.0 + exp_mu * (1.0 / sigma_min_shifted - 1.0))
    u = torch.linspace(u_start, u_min, num_inference_steps, dtype=torch.float32)
    sigmas = exp_mu / (exp_mu + (1.0 / u - 1.0))
    if t_switch < 0.999:
        sigmas[0] = float(t_switch)
    timesteps = sigmas * scheduler.num_train_timesteps
    scheduler.sigmas = sigmas
    scheduler.timesteps = timesteps
    scheduler.training = False


# Without it, refining an upsampled draft keeps the draft's blur (the model treats it as a soft photo).
DEFAULT_NEGATIVE_PROMPT = "blurry, out of focus, low resolution, soft, smooth, jpeg artifacts, upscaled"


class SemanticFoveatedPipeline:
    """End-to-end Semantic Foveated Diffusion pipeline & baseline runner."""

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.2-klein-base-4B",
        clipseg_model_id: str = "CIDAS/clipseg-rd64-refined",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        offload_text_encoder: bool = True,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_id = model_id
        self.offload_text_encoder = offload_text_encoder

        print(f"[SemanticFoveatedPipeline] Loading FLUX.2 Foveated Pipeline on {device}...")
        self.pipe = Flux2FoveatedImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device="cpu",
            model_configs=[
                ModelConfig(model_id=model_id, origin_file_pattern="transformer/*.safetensors"),
                ModelConfig(model_id=model_id, origin_file_pattern="text_encoder/*.safetensors"),
                ModelConfig(model_id=model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=ModelConfig(model_id=model_id, origin_file_pattern="tokenizer/"),
        )
        self.pipe.device = device
        self.pipe.dit.to(device)
        self.pipe.vae.to(device)

        print(f"[SemanticFoveatedPipeline] Loading CLIPSeg localizer ({clipseg_model_id}) on {device}...")
        self.localizer = CLIPSegLocalizer(model_id=clipseg_model_id, device=device)

        # Cache of precomputed LoRA weight deltas {mode: {layer_name: delta_tensor_cpu}}
        self._lora_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        self._active_lora_mode: Optional[str] = None
        self._dit_modules: Dict[str, torch.nn.Module] = dict(self.pipe.dit.named_modules())

    def _preload_lora_deltas(self, lora_mode: str) -> Dict[str, torch.Tensor]:
        """Load and precompute (B @ A) weight deltas for fast (<5ms) LoRA switching."""
        if lora_mode in self._lora_deltas:
            return self._lora_deltas[lora_mode]

        if os.path.exists(lora_mode):
            lora_path = lora_mode
        else:
            assert lora_mode in IMAGE_LORA_FILES, f"Unknown lora_mode: {lora_mode}"
            lora_path = hf_hub_download(repo_id=HF_LORA_REPO, filename=IMAGE_LORA_FILES[lora_mode])

        raw_sd = load_state_dict(lora_path, torch_dtype=self.torch_dtype, device="cpu")
        lora_loader = self.pipe.lora_loader(torch_dtype=self.torch_dtype, device=self.device)
        converted = lora_loader.convert_state_dict(raw_sd)
        layer_names = sorted(
            set(k.replace(".lora_B.weight", "") for k in converted if k.endswith(".lora_B.weight"))
        )

        deltas: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name in layer_names:
                if name not in self._dit_modules:
                    continue
                w_up = converted[name + ".lora_B.weight"].to(device=self.device, dtype=torch.float32)
                w_down = converted[name + ".lora_A.weight"].to(device=self.device, dtype=torch.float32)
                if w_up.ndim == 4:
                    w_up = w_up.squeeze(3).squeeze(2)
                    w_down = w_down.squeeze(3).squeeze(2)
                    delta = torch.mm(w_up, w_down).unsqueeze(2).unsqueeze(3)
                else:
                    delta = torch.mm(w_up, w_down)
                deltas[name] = delta.to(dtype=self.torch_dtype).cpu()

        self._lora_deltas[lora_mode] = deltas
        return deltas

    @torch.no_grad()
    def set_lora_mode(self, lora_mode: Optional[str] = None):
        """Switch the DiT between base weights (`None`) and a Foveated LoRA (`"random"`, `"bbox"`, `"saliency"`)."""
        if lora_mode == self._active_lora_mode:
            return

        # 1. Subtract currently active LoRA if any
        if self._active_lora_mode is not None:
            old_deltas = self._lora_deltas[self._active_lora_mode]
            for name, delta_cpu in old_deltas.items():
                mod = self._dit_modules[name]
                mod.weight.data.sub_(delta_cpu.to(device=mod.weight.device, dtype=mod.weight.dtype))
            self._active_lora_mode = None

        # 2. Add new LoRA if requested
        if lora_mode is not None and lora_mode != "none":
            new_deltas = self._preload_lora_deltas(lora_mode)
            for name, delta_cpu in new_deltas.items():
                mod = self._dit_modules[name]
                mod.weight.data.add_(delta_cpu.to(device=mod.weight.device, dtype=mod.weight.dtype))
            self._active_lora_mode = lora_mode

    @torch.no_grad()
    def encode_prompt_cached(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        cfg_scale: float = 4.0,
    ) -> Tuple[dict, dict]:
        """Encode positive and negative prompts once and optionally offload the text encoder to CPU."""
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}

        for unit in self.pipe.units:
            if unit.__class__.__name__ in (
                "Flux2FoveatedUnit_PromptEmbedder",
                "Flux2FoveatedUnit_Qwen3PromptEmbedder",
            ):
                if self.pipe.text_encoder_qwen3 is not None:
                    enc_dev = next(self.pipe.text_encoder_qwen3.parameters()).device
                    if enc_dev != torch.device(self.device):
                        self.pipe.text_encoder_qwen3.to(self.device)
                inputs_posi = unit.process(self.pipe, prompt=prompt)
                if cfg_scale != 1.0:
                    inputs_nega = unit.process(self.pipe, prompt=negative_prompt)

        if self.offload_text_encoder and self.pipe.text_encoder_qwen3 is not None:
            self.pipe.text_encoder_qwen3.to("cpu")
            torch.cuda.empty_cache()

        return inputs_posi, inputs_nega

    @torch.no_grad()
    def run_stage1_draft(
        self,
        prompt: str,
        seed: int = 42,
        target_height: int = 1024,
        target_width: int = 1024,
        lr_factor: int = 2,
        stage1_steps: int = 14,
        cfg_scale: float = 4.0,
        embedded_guidance: float = 4.0,
        cached_prompts: Optional[Tuple[dict, dict]] = None,
        lora_mode: Optional[str] = "no_fov",
        draft_factor: Optional[int] = None,
    ) -> Stage1Output:
        """Stage 1: Generate a coarse low-resolution draft at (H/draft_factor, W/draft_factor)."""
        self.set_lora_mode(lora_mode)

        if cached_prompts is None:
            inputs_posi, inputs_nega = self.encode_prompt_cached(prompt, cfg_scale=cfg_scale)
        else:
            inputs_posi, inputs_nega = cached_prompts

        draft_factor = draft_factor or lr_factor
        draft_h = target_height // draft_factor
        draft_w = target_width // draft_factor
        latent_h = draft_h // 16
        latent_w = draft_w // 16

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        set_flux2_subtrajectory_timesteps(
            self.pipe.scheduler,
            num_inference_steps=stage1_steps,
            t_switch=1.0,
            dynamic_shift_len=latent_h * latent_w,
        )

        # Generate low-res noise matching the (h/2, w/2) token grid
        noise_low = self.pipe.generate_noise(
            (1, 128, latent_h, latent_w),
            seed=seed,
            rand_device=self.device,
            rand_torch_dtype=self.torch_dtype,
        )
        latents = noise_low.reshape(1, 128, latent_h * latent_w).permute(0, 2, 1)

        latent_ids, _, _ = self.pipe._prepare_latent_image_ids_foveation(
            batch_size=1,
            height=latent_h,
            width=latent_w,
            device=self.device,
            dtype=self.torch_dtype,
            foveation_mask=None,
            lr_factor=lr_factor,
        )
        if latent_ids.ndim == 2:
            latent_ids = latent_ids.unsqueeze(0)

        inputs_shared = {
            "cfg_scale": cfg_scale,
            "embedded_guidance": embedded_guidance,
            "height": draft_h,
            "width": draft_w,
            "latents": latents,
            "image_ids": latent_ids,
            "resolution_mask": None,
            "resolution_mask_top_left": None,
            "lr_downsample_factor": lr_factor,
        }

        models = {"dit": self.pipe.dit}
        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep_val = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            inputs_shared["latents"] = latents
            noise_pred = self.pipe.cfg_guided_model_fn(
                self.pipe.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                timestep=timestep_val,
                progress_id=progress_id,
            )
            noise_pred = noise_pred[:, :latents.size(1), :]
            latents = self.pipe.step(
                self.pipe.scheduler,
                progress_id=progress_id,
                noise_pred=noise_pred,
                latents=latents,
            )

        # Decode coarse RGB draft image x_hat^low = D(z_hat_0^low)
        latents_spatial = rearrange(latents, "B (H W) C -> B C H W", H=latent_h, W=latent_w)
        decoded_low = self.pipe.vae.decode(latents_spatial)
        draft_image_low = self.pipe.vae_output_to_image(decoded_low)

        torch.cuda.synchronize(self.device)
        stage1_time_s = time.perf_counter() - t0

        draft_image_upsampled = draft_image_low.resize(
            (target_width, target_height), resample=Image.Resampling.BICUBIC
        )

        return Stage1Output(
            draft_image_low=draft_image_low,
            draft_image_upsampled=draft_image_upsampled,
            latents_low=latents.clone(),
            stage1_time_s=stage1_time_s,
        )

    @torch.no_grad()
    def run_stage2_refinement(
        self,
        prompt: str,
        stage1: Stage1Output,
        foveation_mask: Optional[torch.Tensor],
        full_res_foveation_mask: Optional[torch.Tensor] = None,
        seed: int = 42,
        target_height: int = 1024,
        target_width: int = 1024,
        lr_factor: int = 2,
        stage2_steps: int = 14,
        t_switch: float = 0.50,
        cfg_scale: float = 4.0,
        embedded_guidance: float = 4.0,
        lora_mode: Optional[str] = "random",
        soft_foveation_blend: bool = True,
        cached_prompts: Optional[Tuple[dict, dict]] = None,
    ) -> Tuple[Image.Image, float]:
        """Stage 2: Selective HR re-encoding, partial re-noising at t_switch, and mixed-res denoising.

        If `foveation_mask` is None (or all ones), refines all 4096 HR tokens (Stage-2 Full-HR Oracle).
        """
        self.set_lora_mode(lora_mode)

        if cached_prompts is None:
            inputs_posi, inputs_nega = self.encode_prompt_cached(prompt, cfg_scale=cfg_scale)
        else:
            inputs_posi, inputs_nega = cached_prompts

        latent_h = target_height // 16
        latent_w = target_width // 16

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        # 1. Selective Re-Encoding into Mixed-Resolution Tokens:
        #    Upsample x_hat^low to (H, W) and encode via VAE encoder: z_hat_0^high = E(Up(x_hat^low))
        upsampled_tensor = self.pipe.preprocess_image(stage1.draft_image_upsampled).to(
            device=self.device, dtype=self.torch_dtype
        )
        z0_high_spatial = self.pipe.vae.encode(upsampled_tensor)
        z0_high = rearrange(z0_high_spatial, "B C H W -> B (H W) C")
        low_h, low_w = latent_h // lr_factor, latent_w // lr_factor
        if stage1.latents_low.shape[1] == low_h * low_w:
            z0_low = stage1.latents_low.to(device=self.device, dtype=self.torch_dtype)
        else:
            # Draft grid differs from the periphery grid: re-encode the draft at periphery resolution
            draft_periph = stage1.draft_image_low.resize(
                (target_width // lr_factor, target_height // lr_factor), resample=Image.Resampling.BICUBIC
            )
            low_tensor = self.pipe.preprocess_image(draft_periph).to(device=self.device, dtype=self.torch_dtype)
            z0_low = rearrange(self.pipe.vae.encode(low_tensor), "B C H W -> B (H W) C")

        # Merge z0_high (inside M) and z0_low (outside M) into mixed-res sequence z_hat_0^fov
        z0_fov = self.pipe._downsample_latents(
            z0_high,
            latent_h,
            latent_w,
            foveation_mask=foveation_mask,
            latents_downsampled=z0_low,
            lr_factor=lr_factor,
        )

        # 2. Partial Re-Noising at t_switch:
        #    z_{t_switch}^fov = (1 - t_switch) * z_hat_0^fov + t_switch * z_1^fov
        set_flux2_subtrajectory_timesteps(
            self.pipe.scheduler,
            num_inference_steps=stage2_steps,
            t_switch=t_switch,
            dynamic_shift_len=latent_h * latent_w,
        )

        # Sample spatially aligned Gaussian noise z_1^fov
        noise_high = self.pipe.generate_noise(
            (1, 128, latent_h, latent_w),
            seed=seed + 1000,
            rand_device=self.device,
            rand_torch_dtype=self.torch_dtype,
        ).reshape(1, 128, latent_h * latent_w).permute(0, 2, 1)
        noise_low = self.pipe.generate_noise(
            (1, 128, latent_h // lr_factor, latent_w // lr_factor),
            seed=seed + 1000,
            rand_device=self.device,
            rand_torch_dtype=self.torch_dtype,
        ).reshape(1, 128, (latent_h // lr_factor) * (latent_w // lr_factor)).permute(0, 2, 1)

        z1_fov = self.pipe._downsample_latents(
            noise_high,
            latent_h,
            latent_w,
            foveation_mask=foveation_mask,
            latents_downsampled=noise_low,
            lr_factor=lr_factor,
        )

        latents = self.pipe.scheduler.add_noise(
            z0_fov, z1_fov, timestep=self.pipe.scheduler.timesteps[0]
        )

        # Prepare CRPA RoPE coordinates and resolution masks once before the loop
        latent_ids, resolution_mask, resolution_mask_top_left = (
            self.pipe._prepare_latent_image_ids_foveation(
                batch_size=1,
                height=latent_h,
                width=latent_w,
                device=self.device,
                dtype=self.torch_dtype,
                foveation_mask=foveation_mask,
                lr_factor=lr_factor,
            )
        )
        if latent_ids.ndim == 2:
            latent_ids = latent_ids.unsqueeze(0)

        inputs_shared = {
            "cfg_scale": cfg_scale,
            "embedded_guidance": embedded_guidance,
            "height": target_height,
            "width": target_width,
            "latents": latents,
            "image_ids": latent_ids,
            "resolution_mask": resolution_mask,
            "resolution_mask_top_left": resolution_mask_top_left,
            "lr_downsample_factor": lr_factor,
        }

        models = {"dit": self.pipe.dit}
        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep_val = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            inputs_shared["latents"] = latents
            noise_pred = self.pipe.cfg_guided_model_fn(
                self.pipe.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                timestep=timestep_val,
                progress_id=progress_id,
            )
            if foveation_mask is None:
                noise_pred = noise_pred[:, :latents.size(1), :]
            latents = self.pipe.step(
                self.pipe.scheduler,
                progress_id=progress_id,
                noise_pred=noise_pred,
                latents=latents,
            )

        # 3. Decode & Blend:
        #    Split z_0^fov into HR and LR grids, decode both with VAE D, and blend using smoothed mask M'
        if foveation_mask is not None:
            low_res_latents = self.pipe._downsample_remaining_high_res(
                latents, latent_h, latent_w, foveation_mask=foveation_mask, lr_factor=lr_factor
            )
            h_d, w_d = latent_h // lr_factor, latent_w // lr_factor
            low_res_spatial = low_res_latents.view(1, h_d, w_d, -1).permute(0, 3, 1, 2)
            low_res_image = self.pipe.vae.decode(low_res_spatial)

            high_res_latents = self.pipe._upsample_latents(
                latents, latent_h, latent_w, foveation_mask=foveation_mask, lr_factor=lr_factor
            )
            high_res_spatial = high_res_latents.view(1, latent_h, latent_w, -1).permute(0, 3, 1, 2)
            high_res_image = self.pipe.vae.decode(high_res_spatial)

            low_res_image = F.interpolate(
                low_res_image, size=(target_height, target_width), mode="bicubic", align_corners=False
            )

            if full_res_foveation_mask is not None:
                hr_mask = full_res_foveation_mask[None, None, :, :].float()
            else:
                hr_mask = F.interpolate(
                    foveation_mask[None, None, :, :].float(),
                    size=(target_height, target_width),
                    mode="nearest",
                )
            hr_mask = hr_mask.to(device=self.device, dtype=latents.dtype)

            if soft_foveation_blend:
                upscale = target_height / latent_h
                hr_mask = _gaussian_blur_mask_2d(
                    hr_mask, sigma_pixels=upscale, device=self.device, dtype=latents.dtype
                ).clamp(0.0, 1.0)

            merged = low_res_image * (1.0 - hr_mask) + high_res_image * hr_mask
            image = self.pipe.vae_output_to_image(merged)
        else:
            latents_spatial = rearrange(latents, "B (H W) C -> B C H W", H=latent_h, W=latent_w)
            decoded = self.pipe.vae.decode(latents_spatial)
            image = self.pipe.vae_output_to_image(decoded)

        torch.cuda.synchronize(self.device)
        stage2_time_s = time.perf_counter() - t0
        return image, stage2_time_s

    @torch.no_grad()
    def run_single_stage_full_hr(
        self,
        prompt: str,
        seed: int = 42,
        target_height: int = 1024,
        target_width: int = 1024,
        num_steps: int = 28,
        cfg_scale: float = 4.0,
        embedded_guidance: float = 4.0,
        cached_prompts: Optional[Tuple[dict, dict]] = None,
        lora_mode: Optional[str] = "no_fov",
    ) -> Tuple[Image.Image, float]:
        """Baseline: Standard single-stage full-resolution (1024x1024, 4096 tokens) generation."""
        self.set_lora_mode(lora_mode)
        if cached_prompts is None:
            inputs_posi, inputs_nega = self.encode_prompt_cached(prompt, cfg_scale=cfg_scale)
        else:
            inputs_posi, inputs_nega = cached_prompts

        latent_h = target_height // 16
        latent_w = target_width // 16

        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        set_flux2_subtrajectory_timesteps(
            self.pipe.scheduler,
            num_inference_steps=num_steps,
            t_switch=1.0,
            dynamic_shift_len=latent_h * latent_w,
        )

        noise = self.pipe.generate_noise(
            (1, 128, latent_h, latent_w),
            seed=seed,
            rand_device=self.device,
            rand_torch_dtype=self.torch_dtype,
        )
        latents = noise.reshape(1, 128, latent_h * latent_w).permute(0, 2, 1)

        latent_ids, _, _ = self.pipe._prepare_latent_image_ids_foveation(
            batch_size=1,
            height=latent_h,
            width=latent_w,
            device=self.device,
            dtype=self.torch_dtype,
            foveation_mask=None,
            lr_factor=2,
        )
        if latent_ids.ndim == 2:
            latent_ids = latent_ids.unsqueeze(0)

        inputs_shared = {
            "cfg_scale": cfg_scale,
            "embedded_guidance": embedded_guidance,
            "height": target_height,
            "width": target_width,
            "latents": latents,
            "image_ids": latent_ids,
            "resolution_mask": None,
            "resolution_mask_top_left": None,
            "lr_downsample_factor": 2,
        }

        models = {"dit": self.pipe.dit}
        for progress_id, timestep in enumerate(self.pipe.scheduler.timesteps):
            timestep_val = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            inputs_shared["latents"] = latents
            noise_pred = self.pipe.cfg_guided_model_fn(
                self.pipe.model_fn,
                cfg_scale,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                **models,
                timestep=timestep_val,
                progress_id=progress_id,
            )
            noise_pred = noise_pred[:, :latents.size(1), :]
            latents = self.pipe.step(
                self.pipe.scheduler,
                progress_id=progress_id,
                noise_pred=noise_pred,
                latents=latents,
            )

        latents_spatial = rearrange(latents, "B (H W) C -> B C H W", H=latent_h, W=latent_w)
        decoded = self.pipe.vae.decode(latents_spatial)
        image = self.pipe.vae_output_to_image(decoded)

        torch.cuda.synchronize(self.device)
        elapsed_s = time.perf_counter() - t0
        return image, elapsed_s

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        focus_prompt: Union[str, List[str]],
        seed: int = 42,
        target_height: int = 1024,
        target_width: int = 1024,
        lr_factor: int = 2,
        draft_factor: int = 4,
        stage1_steps: int = 14,
        stage2_steps: int = 14,
        t_switch: float = 0.85,
        budget_ratio: float = 0.25,
        selection_mode: str = "threshold",
        threshold: float = 0.5,
        dilation_blocks: int = 2,
        lora_mode: str = "random",
        soft_foveation_blend: bool = True,
        cfg_scale: float = 4.0,
    ) -> SemanticFoveatedOutput:
        """Run full 2-stage Semantic Foveated Diffusion end-to-end."""
        cached_prompts = self.encode_prompt_cached(prompt, cfg_scale=cfg_scale)

        stage1 = self.run_stage1_draft(
            prompt=prompt,
            seed=seed,
            target_height=target_height,
            target_width=target_width,
            lr_factor=lr_factor,
            stage1_steps=stage1_steps,
            cfg_scale=cfg_scale,
            cached_prompts=cached_prompts,
            lora_mode="no_fov",
            draft_factor=draft_factor,
        )

        mask_output = self.localizer.infer_mask(
            draft_image=stage1.draft_image_low,
            focus_prompt=focus_prompt,
            target_height=target_height,
            target_width=target_width,
            lr_factor=lr_factor,
            budget_ratio=budget_ratio,
            selection_mode=selection_mode,
            threshold=threshold,
            dilation_blocks=dilation_blocks,
        )

        final_image, stage2_time_s = self.run_stage2_refinement(
            prompt=prompt,
            stage1=stage1,
            foveation_mask=mask_output.mask_token_grid,
            full_res_foveation_mask=mask_output.mask_full_res,
            seed=seed,
            target_height=target_height,
            target_width=target_width,
            lr_factor=lr_factor,
            stage2_steps=stage2_steps,
            t_switch=t_switch,
            cfg_scale=cfg_scale,
            lora_mode=lora_mode,
            soft_foveation_blend=soft_foveation_blend,
            cached_prompts=cached_prompts,
        )

        total_time_s = stage1.stage1_time_s + mask_output.inference_time_s + stage2_time_s
        return SemanticFoveatedOutput(
            image=final_image,
            stage1=stage1,
            mask_output=mask_output,
            stage2_time_s=stage2_time_s,
            total_time_s=total_time_s,
        )
