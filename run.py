import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import json
import os
import argparse
import time
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


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo MLX Generator")
    parser.add_argument("--model_path", type=str, default="Z-Image-Turbo-MLX")
    parser.add_argument("--repo_id", type=str, default="uqer1244/MLX-z-image")
    parser.add_argument("--prompt", type=str,
                        default="8k, super detailed semi-realistic anime style female warrior, detailed armor, backlighting, dynamic pose, illustration, highly detailed, dramatic lighting")
    parser.add_argument("--output", type=str, default="res.png")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()

    # Total Benchmark Start
    global_start = time.time()

    # 1. Model Download Check
    if not os.path.exists(args.model_path):
        if args.repo_id:
            print(f"Downloading model from {args.repo_id}...")
            snapshot_download(repo_id=args.repo_id, local_dir=args.model_path)
        else:
            print(f"Error: Path '{args.model_path}' not found.")
            return

    print(f"Generating Image...")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Size: {args.width}x{args.height} | Steps: {args.steps} | Seed: {args.seed}")

    # 2. Text Encoding
    t_start = time.time()
    print("[Phase 1] Text Encoding...", end=" ", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "tokenizer"), trust_remote_code=True)
    te_path = os.path.join(args.model_path, "text_encoder")
    with open(os.path.join(te_path, "config.json"), "r") as f:
        te_config = json.load(f)

    text_encoder = TextEncoderMLX(te_config)
    nn.quantize(text_encoder, bits=4, group_size=32)
    text_encoder.load_weights(os.path.join(te_path, "model.safetensors"))

    messages = [{"role": "user", "content": args.prompt}]
    try:
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        prompt_formatted = args.prompt

    inputs = tokenizer(prompt_formatted, padding="max_length", max_length=512, truncation=True, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])
    prompt_embeds = text_encoder(input_ids)
    mx.eval(prompt_embeds)

    # Padding & BF16 Casting
    cap_feats_np = np.array(prompt_embeds)
    cap_len = cap_feats_np.shape[1]
    pad_len = (-cap_len) % 32
    if pad_len > 0:
        cap_feats_np = np.concatenate([cap_feats_np, np.repeat(cap_feats_np[:, -1:, :], pad_len, axis=1)], axis=1)

    cap_feats_mx = mx.array(cap_feats_np).astype(mx.bfloat16)

    del text_encoder, tokenizer
    if hasattr(mx, "clear_cache"): mx.clear_cache()
    print(f"Done ({time.time() - t_start:.2f}s)")

    # 3. Load Transformer
    t_start = time.time()
    print("[Phase 2] Loading Transformer...", end=" ", flush=True)
    trans_path = os.path.join(args.model_path, "transformer")
    with open(os.path.join(trans_path, "config.json"), "r") as f:
        config = json.load(f)

    model = ZImageTransformerMLX(config)
    nn.quantize(model, bits=4, group_size=32)
    model.load_weights(os.path.join(trans_path, "model.safetensors"))
    model.eval()
    print(f"Done ({time.time() - t_start:.2f}s)")

    # 4. Prepare Scheduler & Latents
    sched_path = os.path.join(args.model_path, "scheduler")
    try:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(sched_path)
    except:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True)

    latents_shape = (1, 16, args.height // 8, args.width // 8)
    latents_pt = torch.randn(latents_shape, generator=torch.Generator().manual_seed(args.seed), dtype=torch.float32)
    mu = calculate_shift((latents_pt.shape[2] // 2) * (latents_pt.shape[3] // 2))
    scheduler.set_timesteps(args.steps, mu=mu)

    # 5. Positional Embeddings (BF16)
    total_len = cap_feats_mx.shape[1]
    p = 2
    H_tok, W_tok = (args.height // 8) // p, (args.width // 8) // p

    img_pos_mx = mx.array(create_coordinate_grid((1, H_tok, W_tok), (total_len + 1, 0, 0)).reshape(-1, 3)[None]).astype(
        mx.bfloat16)
    cap_pos_mx = mx.array(create_coordinate_grid((total_len, 1, 1), (1, 0, 0)).reshape(-1, 3)[None]).astype(mx.bfloat16)

    # 6. JIT Compilation Setup
    @mx.compile
    def compiled_step(x, t, feats, i_pos, c_pos):
        B, C, H, W = x.shape
        x_reshaped = x.reshape(C, 1, 1, H_tok, p, W_tok, p)
        x_in = x_reshaped.transpose(1, 2, 3, 5, 4, 6, 0).reshape(1, -1, C * p * p)
        noise_pred = model(x_in, t, feats, i_pos, c_pos, cap_mask=None)
        out = noise_pred.reshape(1, 1, H_tok, W_tok, p, p, C)
        out = out.transpose(6, 0, 1, 2, 4, 3, 5).reshape(1, C, H, W)
        return -out

    # 7. Denoising Loop
    print(f"[Phase 3] Denoising ({args.steps} Steps)...")
    denoise_start = time.time()

    for i, t in enumerate(scheduler.timesteps):
        step_start = time.time()
        t_val = t.item()

        # Data Prep
        latents_mx = mx.array(latents_pt.numpy()).astype(mx.bfloat16)
        t_mx = mx.array([(1000.0 - t_val) / 1000.0], dtype=mx.bfloat16)

        # Inference (First step will implicitly compile)
        noise_pred_mx = compiled_step(latents_mx, t_mx, cap_feats_mx, img_pos_mx, cap_pos_mx)
        mx.eval(noise_pred_mx)

        # Data Post
        noise_pred_pt = torch.from_numpy(np.array(noise_pred_mx.astype(mx.float32)))
        latents_pt = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=False)[0]

        step_time = time.time() - step_start
        note = " (includes compile)" if i == 0 else ""
        print(f"   Step {i + 1}/{args.steps}: {step_time:.2f}s (t={t_val:.1f}){note}")

    total_denoise_time = time.time() - denoise_start
    print(f"   Avg Speed: {total_denoise_time / args.steps:.2f} s/it")
    del model

    # 8. VAE Decoding
    print("[Phase 4] Decoding...", end=" ", flush=True)
    t_start = time.time()
    vae_path = os.path.join(args.model_path, "vae")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)

    latents_pt = latents_pt.to(device)
    latents_pt = (latents_pt / vae.config.scaling_factor) + getattr(vae.config, "shift_factor", 0.0)

    with torch.no_grad():
        image = vae.decode(latents_pt).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")

    Image.fromarray(image[0]).save(args.output)
    print(f"Done ({time.time() - t_start:.2f}s)")

    print(f"Total Time: {time.time() - global_start:.2f}s | Saved to: {args.output}")

if __name__ == "__main__":
    main()