import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import json
import os
import time
import gc
from PIL import Image
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download

from mlx_z_image import ZImageTransformerMLX
from mlx_text_encoder import TextEncoderMLX

def create_coordinate_grid(size, start):
    d0, d1, d2 = size
    s0, s1, s2 = start
    i = mx.arange(s0, s0 + d0)
    j = mx.arange(s1, s1 + d1)
    k = mx.arange(s2, s2 + d2)
    grid_i = mx.broadcast_to(i[:, None, None], (d0, d1, d2))
    grid_j = mx.broadcast_to(j[None, :, None], (d0, d1, d2))
    grid_k = mx.broadcast_to(k[None, None, :], (d0, d1, d2))
    return mx.stack([grid_i, grid_j, grid_k], axis=-1).reshape(-1, 3)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def load_sharded_weights(model_path):
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    weights = {}
    if os.path.exists(index_path):
        print(f"   [Loader] Detected sharded weights index: {index_path}")
        with open(index_path, "r") as f:
            index_data = json.load(f)
        shard_files = sorted(list(set(index_data["weight_map"].values())))
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            print(f"   [Loader] Loading shard: {shard_file}...")
            weights.update(mx.load(shard_path))
            if hasattr(mx, "clear_cache"): mx.clear_cache()
    else:
        single_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(single_path):
            weights = mx.load(single_path)
        else:
            import glob
            files = glob.glob(os.path.join(model_path, "*.safetensors"))
            for f in files:
                weights.update(mx.load(f))
    return weights


# =========================================
# Main Pipeline Class
# =========================================
class ZImagePipeline:
    def __init__(self,
                 model_path="Z-Image-Turbo-MLX",
                 text_encoder_path="Z-Image-Turbo-MLX/text_encoder",
                 repo_id="uqer1244/MLX-z-image"):
        self.model_path = model_path
        self.text_encoder_path = text_encoder_path
        self.repo_id = repo_id

        if not os.path.exists(self.model_path):
            print(f"Downloading base model from {self.repo_id}...")
            snapshot_download(repo_id=self.repo_id, local_dir=self.model_path)

    def generate(self, prompt, width=720, height=1024, steps=9, seed=42):
        print(f"🚀 Pipeline Started | Size: {width}x{height} | Steps: {steps}")
        global_start = time.time()

        # ----------------------------------------------------------------
        # [Phase 1] Text Encoding (BF16)
        # ----------------------------------------------------------------
        t_start = time.time()
        print(f"[Phase 1] Text Encoding (BF16)...", end=" ", flush=True)

        tokenizer_path = os.path.join(self.model_path, "tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        with open(os.path.join(self.text_encoder_path, "config.json"), "r") as f:
            te_config = json.load(f)

        # Load Text Encoder
        text_encoder = TextEncoderMLX(te_config)
        te_weights = load_sharded_weights(self.text_encoder_path)
        text_encoder.load_weights(list(te_weights.items()))
        del te_weights

        # Encode
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            prompt_fmt = prompt

        inputs = tokenizer(prompt_fmt, padding="max_length", max_length=512, truncation=True, return_tensors="np")
        prompt_embeds = text_encoder(mx.array(inputs["input_ids"]))
        mx.eval(prompt_embeds)

        # Padding & Casting
        cap_feats_np = np.array(prompt_embeds)
        pad = (-cap_feats_np.shape[1]) % 32
        if pad > 0: cap_feats_np = np.concatenate([cap_feats_np, np.repeat(cap_feats_np[:, -1:, :], pad, axis=1)],
                                                  axis=1)
        cap_feats_mx = mx.array(cap_feats_np).astype(mx.bfloat16)

        # Cleanup Phase 1
        del text_encoder, tokenizer
        mx.clear_cache()
        gc.collect()
        print(f"Done ({time.time() - t_start:.2f}s)")

        # ----------------------------------------------------------------
        # [Phase 2] Transformer Loading (4-bit)
        # ----------------------------------------------------------------
        t_start = time.time()
        trans_path = os.path.join(self.model_path, "transformer")
        print(f"[Phase 2] Loading Transformer (4-bit)...", end=" ", flush=True)

        with open(os.path.join(trans_path, "config.json"), "r") as f:
            config = json.load(f)

        model = ZImageTransformerMLX(config)
        nn.quantize(model, bits=4, group_size=32)  # 메모리 절약 핵심
        model.load_weights(os.path.join(trans_path, "model.safetensors"))
        model.eval()
        print(f"Done ({time.time() - t_start:.2f}s)")

        # ----------------------------------------------------------------
        # [Phase 3] Denoising
        # ----------------------------------------------------------------
        print(f"[Phase 3] Denoising...", end="\n")

        # Scheduler
        sched_path = os.path.join(self.model_path, "scheduler")
        try:
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sched_path)
        except:
            scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True)

        # Latents
        latents = torch.randn((1, 16, height // 8, width // 8), generator=torch.Generator().manual_seed(seed),
                              dtype=torch.float32)
        mu = calculate_shift((latents.shape[2] // 2) * (latents.shape[3] // 2))
        scheduler.set_timesteps(steps, mu=mu)

        # Positional Embeddings
        total_len = cap_feats_mx.shape[1]
        H_tok, W_tok = (height // 8) // 2, (width // 8) // 2
        img_pos = mx.array(
            create_coordinate_grid((1, H_tok, W_tok), (total_len + 1, 0, 0)).reshape(-1, 3)[None]).astype(mx.bfloat16)
        cap_pos = mx.array(create_coordinate_grid((total_len, 1, 1), (1, 0, 0)).reshape(-1, 3)[None]).astype(
            mx.bfloat16)

        # Compiled Step Function
        @mx.compile
        def step_fn(x, t, feats, i_pos, c_pos):
            B, C, H, W = x.shape
            x = x.reshape(C, 1, 1, H_tok, 2, W_tok, 2).transpose(1, 2, 3, 5, 4, 6, 0).reshape(1, -1, C * 4)
            out = model(x, t, feats, i_pos, c_pos, cap_mask=None)
            return -out.reshape(1, 1, H_tok, W_tok, 2, 2, C).transpose(6, 0, 1, 2, 4, 3, 5).reshape(1, C, H, W)

        # Loop
        denoise_start = time.time()
        for i, t in enumerate(scheduler.timesteps):
            step_start = time.time()

            latents_mx = mx.array(latents.numpy()).astype(mx.bfloat16)
            t_mx = mx.array([(1000.0 - t.item()) / 1000.0], dtype=mx.bfloat16)

            noise_mx = step_fn(latents_mx, t_mx, cap_feats_mx, img_pos, cap_pos)
            mx.eval(noise_mx)

            noise_pt = torch.from_numpy(np.array(noise_mx.astype(mx.float32)))
            latents = scheduler.step(noise_pt, t, latents, return_dict=False)[0]

            mx.clear_cache()
            gc.collect()

            print(f"   Step {i + 1}/{steps}: {time.time() - step_start:.2f}s")

        del model
        print(f"   Avg Speed: {(time.time() - denoise_start) / steps:.2f} s/it")

        # ----------------------------------------------------------------
        # [Phase 4] Decoding (VAE)
        # ----------------------------------------------------------------
        print("[Phase 4] Decoding...", end=" ", flush=True)
        t_dec = time.time()

        vae_path = os.path.join(self.model_path, "vae")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        vae = AutoencoderKL.from_pretrained(vae_path).to(device)

        latents = latents.to(device)
        latents = (latents / vae.config.scaling_factor) + getattr(vae.config, "shift_factor", 0.0)

        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        pil_image = Image.fromarray((image[0] * 255).round().astype("uint8"))

        print(f"Done ({time.time() - t_dec:.2f}s)")
        print(f"✨ Pipeline Finished in {time.time() - global_start:.2f}s")

        return pil_image