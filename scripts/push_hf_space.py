"""Create/update a Hugging Face Space from hf_space/.

Usage:
  set HF_TOKEN=hf_xxx
  python scripts/push_hf_space.py --repo YOUR_USERNAME/plant-leaf-disease
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="",
        help="Space id like username/plant-leaf-disease",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private Space",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN first (https://huggingface.co/settings/tokens)."
        )

    api = HfApi(token=token)
    info = whoami(token=token)
    username = info["name"]
    repo_id = args.repo or f"{username}/plant-leaf-disease"

    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    root = Path(__file__).resolve().parents[1] / "hf_space"
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="space",
        token=token,
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
    )
    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"Uploaded from {root}")
    print(f"Space URL: {url}")


if __name__ == "__main__":
    main()
