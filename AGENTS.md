# Repository Guidelines

## Project Structure & Module Organization
- `mlx_z_image.py`, `mlx_text_encoder.py`, and `mlx_pipeline.py` house the MLX transformer, text encoder, and orchestration logic; extend components next to these files.
- `run.py` is the entry point, pulling prompts from `prompt.txt`, downloading weights into `Z-Image-Turbo-MLX/`, and saving renders under `img/`.
- `custom_nodes/` delivers the ComfyUI bridge; `TB4_Bridge/` and `converting/` contain Thunderbolt clustering and conversion experiments—mark prototypes clearly.
- Support scripts (`check_lora.py`, `lora_utils.py`) live at the repo root; keep new utilities snake_case and colocated.
- Docs (`readme.md`, `LICENCE`, `AGENTS.md`) stay at the top level for quick discovery.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate   # optional isolation
pip install -r requirements.txt                     # sync MLX, diffusers, HF deps
python run.py --width 1024 --height 1024            # generate an image, auto-download weights
python check_lora.py path/to/model.safetensors      # validate LoRA compatibility
```
See `custom_nodes/readme.md` for ComfyUI wiring; run `python run.py --help` for flags and pin `--seed 42` when sharing repros.

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation, snake_case modules/functions, PascalCase classes, and ALL_CAPS constants.
- Surface configuration near the top of each script and minimize implicit globals.
- Format touched Python files with `black` (88 columns) or equivalent and group imports stdlib/third-party/local.
- Name new assets using lowercase words plus hyphens (`img/sample-grid.png`) to stay aligned with current artifacts.

## Testing Guidelines
- No automated suite exists; rely on deterministic inference runs and capture the exact CLI plus output path in the pull request.
- For LoRA or pipeline changes, run `python run.py --lora <file> --steps 9 --output img/feature_test.png` and attach before/after renders.
- For Thunderbolt or conversion edits, log the device mix, throughput, and manual checks used.

## Commit & Pull Request Guidelines
- Follow the existing log style: concise, present-tense subjects that capture scope (`Add hf_transfer via huggingface downloader`, `Comfy Update`), optionally adding a short qualifier after a dash.
- Link issues (`Fixes #12`), call out user impact, and summarize regression risk in the PR body.
- Provide repro commands, hardware specs, and representative outputs or screenshots whenever behavior changes.
- Keep PRs scoped (pipeline vs. UI vs. conversion) and note any downloaded weights or environment variables reviewers must configure.

## Security & Configuration Tips
- Do not commit downloaded weights or Hugging Face tokens; leverage runtime downloads via `huggingface_hub`.
- Guard Apple Silicon–only optimizations behind flags so Intel machines can still import modules safely.
