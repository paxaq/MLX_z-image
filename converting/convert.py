import os
import argparse
import json
import torch
import numpy as np
import mlx.core as mx
from safetensors.torch import load_file as load_pt_file


def map_key_and_convert(key, tensor):
    # PyTorch Tensor -> Numpy (Float32)
    # BF16 변환은 나중에 MLX array 생성 시 수행
    if isinstance(tensor, torch.Tensor):
        val = tensor.detach().cpu().float().numpy()
    else:
        val = tensor

    new_key = key

    # 키 매핑 로직 (기존과 동일)
    if "t_embedder.mlp.0" in key:
        new_key = key.replace("t_embedder.mlp.0", "t_embedder.linear1")
    elif "t_embedder.mlp.2" in key:
        new_key = key.replace("t_embedder.mlp.2", "t_embedder.linear2")
    elif "all_x_embedder.2-1" in key:
        new_key = key.replace("all_x_embedder.2-1", "x_embedder")
    elif "cap_embedder.0" in key:
        new_key = key.replace("cap_embedder.0", "cap_embedder.layers.0")
    elif "cap_embedder.1" in key:
        new_key = key.replace("cap_embedder.1", "cap_embedder.layers.1")
    elif "all_final_layer.2-1" in key:
        new_key = key.replace("all_final_layer.2-1", "final_layer")

    if "adaLN_modulation.1" in new_key:
        new_key = new_key.replace("adaLN_modulation.1", "adaLN_modulation.layers.1")
    elif "attention.to_out.0" in key:
        new_key = key.replace("attention.to_out.0", "attention.to_out")
    elif "adaLN_modulation.0" in key and "final" not in key:
        new_key = key.replace("adaLN_modulation.0", "adaLN_modulation")
    elif "adaLN_modulation.1" in key and "final" not in key:
        new_key = key.replace("adaLN_modulation.1", "adaLN_modulation")

    # 🔥 MLX Array로 변환 (BF16)
    return new_key, mx.array(val).astype(mx.bfloat16)


def main():
    parser = argparse.ArgumentParser(description="Convert Local Sharded Transformer to MLX (BF16)")
    # 로컬 경로를 입력받습니다.
    parser.add_argument("--src_path", type=str, default="Z-Image-Turbo/transformer",
                        help="Path to local folder containing .safetensors and index.json")
    parser.add_argument("--dest_path", type=str, default="Z-Image-Turbo-Transformer-BF16",
                        help="Output directory")
    args = parser.parse_args()

    print(f"🚀 Starting Sharded Conversion: {args.src_path} -> {args.dest_path}")
    os.makedirs(args.dest_path, exist_ok=True)

    # 1. Config 복사
    config_src = os.path.join(args.src_path, "config.json")
    if os.path.exists(config_src):
        with open(config_src, "r") as f:
            config = json.load(f)

        # MLX 호환성 수정
        if "n_heads" in config and "nheads" not in config:
            config["nheads"] = config["n_heads"]
        config["t_scale"] = config.get("t_scale", 1000.0)

        with open(os.path.join(args.dest_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        print("✅ Config copied and updated.")
    else:
        print("⚠️ Warning: config.json not found in source path.")

    # 2. Index 파일 로드 (분할 정보 확인)
    index_path = os.path.join(args.src_path, "diffusion_pytorch_model.safetensors.index.json")
    if not os.path.exists(index_path):
        # 인덱스 파일이 없는 경우 (단일 파일일 수도 있지만, 사용자가 3개라고 했으므로 에러 처리)
        print(f"❌ Error: Index file not found at {index_path}")
        print("   Make sure you are pointing to the folder containing the .index.json file.")
        return

    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    # 처리할 파일 목록 추출 (중복 제거 및 정렬)
    files_to_process = sorted(list(set(weight_map.values())))

    print(f"📦 Found {len(files_to_process)} shards to convert.")

    new_weight_map = {}

    # 3. 파일별 순차 변환
    for i, filename in enumerate(files_to_process):
        src_file_path = os.path.join(args.src_path, filename)

        # 파일명 변환 (diffusion_pytorch_model... -> model...)
        # MLX/HF 표준인 model-xxxxx-of-xxxxx.safetensors 형식으로 변경하거나 그대로 유지
        # 여기서는 구분을 위해 'model-' 접두사로 통일합니다.
        dest_filename = filename.replace("diffusion_pytorch_model", "model")
        dest_file_path = os.path.join(args.dest_path, dest_filename)

        print(f"\n[{i + 1}/{len(files_to_process)}] Processing {filename} -> {dest_filename}...")

        # PyTorch 가중치 로드
        pt_weights = load_pt_file(src_file_path)
        mlx_shard = {}

        # 키 변환 및 BF16 캐스팅
        for k, v in pt_weights.items():
            new_k, new_v = map_key_and_convert(k, v)
            mlx_shard[new_k] = new_v

            # 새로운 weight_map 생성을 위해 기록
            new_weight_map[new_k] = dest_filename

        # 저장
        mx.save_safetensors(dest_file_path, mlx_shard)
        print(f"   ✅ Saved shard to {dest_file_path}")

        # 메모리 정리
        del pt_weights
        del mlx_shard
        if hasattr(mx, "clear_cache"): mx.clear_cache()

    # 4. 새로운 Index 파일 생성
    new_index_data = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": new_weight_map
    }

    # 총 사이즈 계산 (선택 사항, 기존 메타데이터 유지)
    with open(os.path.join(args.dest_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index_data, f, indent=4)

    print("\n🎉 All shards converted successfully!")
    print(f"   Check output at: {args.dest_path}")


if __name__ == "__main__":
    main()