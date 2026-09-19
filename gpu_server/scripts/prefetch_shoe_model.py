"""Download the pinned, Apache-2.0 shoe-editing model without loading GPU weights."""
import argparse
from huggingface_hub import snapshot_download

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.destination,
          allow_patterns=["model_index.json", "LICENSE.md", "README.md", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"], max_workers=2), flush=True)
