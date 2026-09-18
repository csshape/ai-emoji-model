"""Export the trained model to ONNX, optionally int8-quantised.

Not needed for the local demo -- this is the path to running on a phone via
ONNX Runtime (iOS and Android). Install the extras first:

    uv add --optional export onnx onnxruntime
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .model import EmojiEncoder, ModelConfig


def export(checkpoint: Path, data_dir: Path, out_dir: Path, quantize: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = torch.load(checkpoint, map_location="cpu")
    cfg = ModelConfig(**blob["config"])
    model = EmojiEncoder(cfg)
    model.load_state_dict(blob["model"])
    model.eval()

    dummy = torch.ones((1, cfg.max_len), dtype=torch.long)
    fp32 = out_dir / "emoji_model.onnx"
    torch.onnx.export(
        model, (dummy,), str(fp32),
        input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {fp32}  ({fp32.stat().st_size/1e6:.2f} MB)")

    # Ship the tokenizer and label list alongside the graph -- the app needs both.
    for name in ("spm.model", "emoji_vocab.json"):
        shutil.copy(data_dir / name, out_dir / name)
    (out_dir / "model_config.json").write_text(json.dumps(cfg.to_dict(), indent=1))

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        int8 = out_dir / "emoji_model.int8.onnx"
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
        print(f"quantised {int8}  ({int8.stat().st_size/1e6:.2f} MB)")

    _verify(fp32, model, dummy)


def _verify(onnx_path: Path, model: torch.nn.Module, dummy: torch.Tensor) -> None:
    """Confirm the exported graph matches PyTorch before we trust it."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("(skipping verification: onnxruntime not installed)")
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input_ids": dummy.numpy()})[0]
    with torch.no_grad():
        want = model(dummy).numpy()
    diff = float(np.abs(got - want).max())
    print(f"max |onnx - torch| = {diff:.2e}  {'OK' if diff < 1e-4 else 'MISMATCH'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="export")
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()
    export(Path(args.checkpoint), Path(args.data), Path(args.out), args.quantize)


if __name__ == "__main__":
    main()
