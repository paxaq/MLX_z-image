import argparse
import os
import json
import torch  # torch import 필수
import mlx.core as mx
from safetensors.torch import load_file as load_pt_file
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert Sharded Text Encoder to MLX (BF16)")
    # 기본값에 언더바(_) 적용
    parser.add_argument("--src_path", type=str, default="Z-Image-Turbo/text_encoder",
                        help="Path to PyTorch model folder")
    parser.add_argument("--dest_path", type=str, default="Z-Image-Turbo-MLX-TextEncoder-BF16", help="Output path")
    args = parser.parse_args()

    print(f"🚀 Starting Low-Memory Conversion: {args.src_path} -> {args.dest_path}")
    os.makedirs(args.dest_path, exist_ok=True)

    # 1. Index 파일 로드
    index_path = os.path.join(args.src_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"❌ Error: '{index_path}' not found.")
        return

    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    files_to_process = sorted(list(set(weight_map.values())))

    print(f"📦 Found {len(files_to_process)} shards. Processing one by one...")

    # 2. 순차 변환
    for i, filename in enumerate(files_to_process):
        print(f"\n[{i + 1}/{len(files_to_process)}] Processing {filename}...")

        file_path = os.path.join(args.src_path, filename)
        pt_weights = load_pt_file(file_path)

        mlx_shard = {}

        for k, v in pt_weights.items():
            # 🔥 [수정] BF16 텐서 -> Float32 변환 -> Numpy -> MLX BF16
            # PyTorch BF16은 바로 numpy()가 안되므로 .float() (즉 float32)로 바꾼 뒤 넘겨야 함
            if isinstance(v, torch.Tensor):
                val_np = v.float().numpy()
            else:
                val_np = v

            # MLX에서 다시 BF16으로 저장 (용량 절약)
            val_mx = mx.array(val_np).astype(mx.bfloat16)

            mlx_shard[k] = val_mx

        save_path = os.path.join(args.dest_path, filename)
        mx.save_safetensors(save_path, mlx_shard)
        print(f"   ✅ Saved to {save_path}")

        del pt_weights
        del mlx_shard
        if hasattr(mx, "clear_cache"): mx.clear_cache()

    # 3. Config 복사
    print("\n📑 Copying Config and Index files...")

    config_src = os.path.join(args.src_path, "config.json")
    if os.path.exists(config_src):
        with open(config_src, "r") as f: config = json.load(f)
        with open(os.path.join(args.dest_path, "config.json"), "w") as f: json.dump(config, f, indent=4)

    # Index 복사
    with open(os.path.join(args.dest_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=4)

    print("\n🎉 Conversion Complete! (Sharded)")


if __name__ == "__main__":
    main()