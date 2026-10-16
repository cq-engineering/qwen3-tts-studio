"""Pre-download Qwen3-TTS weights so first synthesis is instant."""
import argparse
from huggingface_hub import snapshot_download

DEFAULT = ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]
ALL = DEFAULT + [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="download every model + tokenizer")
    ap.add_argument("--model", default=None, help="single repo id to fetch")
    a = ap.parse_args()
    targets = [a.model] if a.model else (ALL if a.all else DEFAULT)
    for repo in targets:
        print(f"==> downloading {repo} ...", flush=True)
        path = snapshot_download(repo)
        print(f"    cached at {path}")
    print("done ✓")
