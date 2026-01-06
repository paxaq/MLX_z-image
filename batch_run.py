import argparse
import datetime as dt
import random
import re
import subprocess
import sys
from pathlib import Path


def build_filename(output_dir: Path, seed_value: int) -> Path:
    """Return img/YYYYMMDD_HHMMSS_mmmmmm_seed.png style path."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"{timestamp}_{seed_value}.png"


def parse_prompts(prompt_file: Path, parser: argparse.ArgumentParser) -> tuple[str, list[str]]:
    original_text = prompt_file.read_text(encoding="utf-8")
    normalized = original_text.strip()
    if not normalized:
        parser.error(f"{prompt_file} is empty")
    paragraphs = [p.strip() for p in re.split(r"(?:\r?\n){2,}", original_text) if p.strip()]
    if not paragraphs:
        paragraphs = [normalized]
    return original_text, paragraphs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run run.py multiple times with timestamped outputs"
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("img"),
        help="Folder to store generated images (default: img/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without invoking run.py",
    )
    args, passthrough_args = parser.parse_known_args()
    batch_prompt_file = Path("batch_prompt.txt")
    run_prompt_file = Path("prompt.txt")

    if args.count < 1:
        parser.error("--count must be at least 1")

    if not batch_prompt_file.exists():
        parser.error(f"{batch_prompt_file} not found")

    run_prompt_existed = run_prompt_file.exists()
    original_run_prompt_text = (
        run_prompt_file.read_text(encoding="utf-8") if run_prompt_existed else ""
    )

    _, prompts = parse_prompts(batch_prompt_file, parser)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    user_defined_seed = False
    user_seed_value = None
    i = 0
    while i < len(passthrough_args):
        arg = passthrough_args[i]
        if arg == "--seed":
            if i + 1 >= len(passthrough_args):
                parser.error("--seed requires a value")
            user_defined_seed = True
            user_seed_value = int(passthrough_args[i + 1])
            break
        if arg.startswith("--seed="):
            user_defined_seed = True
            user_seed_value = int(arg.split("=", 1)[1])
            break
        i += 1

    try:
        for prompt_idx, prompt in enumerate(prompts, start=1):
            if not args.dry_run:
                run_prompt_file.write_text(f"{prompt}\n", encoding="utf-8")
            print(f"[Prompt {prompt_idx}/{len(prompts)}] {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

            for run_idx in range(1, args.count + 1):
                seed_value = (
                    user_seed_value
                    if user_defined_seed and user_seed_value is not None
                    else random.randint(0, 2**31 - 1)
                )
                output_path = build_filename(output_dir, seed_value)
                cmd = [sys.executable, "run.py", "--output", str(output_path)]

                if not user_defined_seed:
                    cmd.extend(["--seed", str(seed_value)])

                cmd.extend(passthrough_args)
                print(
                    f"  [Prompt {prompt_idx} | {run_idx}/{args.count}] Running: {' '.join(cmd)}"
                )
                if args.dry_run:
                    print("    (dry-run) Command skipped")
                    continue

                subprocess.run(cmd, check=True)
    finally:
        if not args.dry_run:
            if run_prompt_existed:
                run_prompt_file.write_text(original_run_prompt_text, encoding="utf-8")
            elif run_prompt_file.exists():
                run_prompt_file.unlink()


if __name__ == "__main__":
    main()
