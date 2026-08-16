#!/usr/bin/env python
"""Push outputs/best_model.keras to a Hugging Face repo so Render can download it.

The 127 MB model cannot live in the GitHub repo (100 MB/file limit), so we host
it on Hugging Face and point Render's MODEL_URL at it.

Usage:
  pip install huggingface_hub
  export HF_TOKEN=hf_...                      # https://huggingface.co/settings/tokens
  python backend/push_model_to_hf.py <your-hf-username>/dr-grade-model

Prints the download URL to set as MODEL_URL on Render.
A PUBLIC repo is recommended so the backend can download it without credentials.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi  # noqa: E402

MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs",
    "best_model.keras",
)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python backend/push_model_to_hf.py <your-hf-username>/dr-grade-model")
    repo = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN first: https://huggingface.co/settings/tokens")
    if not os.path.exists(MODEL):
        sys.exit(f"Model not found: {MODEL} — run train.py/evaluate first.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=MODEL,
        path_in_repo="best_model.keras",
        repo_id=repo,
        repo_type="model",
    )
    url = f"https://huggingface.co/{repo}/resolve/main/best_model.keras"
    print(f"\nDone. Set this as MODEL_URL on Render:\n\n  {url}\n")


if __name__ == "__main__":
    main()
