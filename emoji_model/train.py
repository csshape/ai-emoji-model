"""Train the emoji encoder with a multi-label (BCE) objective."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import EmojiEncoder, ModelConfig
from .vocab import EmojiVocab


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(path: Path) -> TensorDataset:
    blob = torch.load(path)
    return TensorDataset(blob["input_ids"].long(), blob["targets"].float())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    n = top1 = 0
    recall5 = 0.0
    loss_sum = 0.0
    lossf = nn.BCEWithLogitsLoss()
    for ids, tgt in loader:
        ids, tgt = ids.to(device), tgt.to(device)
        logits = model(ids)
        loss_sum += lossf(logits, tgt).item() * ids.shape[0]
        top5 = logits.topk(5, dim=-1).indices
        hits = tgt.gather(1, top5)                      # (B,5) 1.0 where correct
        top1 += hits[:, 0].sum().item()
        recall5 += (hits.sum(1) / tgt.sum(1).clamp(min=1)).sum().item()
        n += ids.shape[0]
    model.train()
    return {"loss": loss_sum / n, "top1": top1 / n, "recall@5": recall5 / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--label-head", action="store_true",
                    help="tie output weights to emoji descriptions")
    ap.add_argument("--class-balance", type=float, default=0.0,
                    help="weight rare emoji up in the loss (0 = off, 1 = full "
                         "inverse frequency; 0.5 is a good starting point)")
    args = ap.parse_args()

    data, outdir = Path(args.data), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    train_ds, val_ds = load_split(data / "train.pt"), load_split(data / "val.pt")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=512)

    cfg = ModelConfig(**json.loads((data / "model_config.json").read_text()))
    cfg.label_head = args.label_head
    model = EmojiEncoder(cfg)
    if args.label_head:
        model.set_label_tokens(torch.load(data / "label_tokens.pt"))
    model = model.to(device)
    vocab = EmojiVocab.load(data / "emoji_vocab.json")
    print(f"device={device}  params={model.n_params():,}  "
          f"train={len(train_ds):,}  val={len(val_ds):,}  labels={len(vocab)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.class_balance > 0:
        # 😂 appears 56,642 times, 📝 676 -- an 84x gap, so the model learns to
        # guess 😂 and little else. Scaling each emoji's positive term by
        # (median / count)^beta makes a miss on a rare emoji cost as much as a
        # miss on a common one, without discarding data or patching at
        # inference time.
        counts = torch.load(data / "train.pt")["targets"].sum(0).float()
        weights = (counts.median() / counts.clamp(min=1)).pow(args.class_balance)
        # Normalise so the median emoji keeps weight 1.0: this redistributes
        # emphasis between emoji without quietly scaling the whole loss (and
        # thus the effective learning rate) up or down.
        weights = (weights / weights.median()).clamp(0.05, 50.0).to(device)
        lossf = nn.BCEWithLogitsLoss(pos_weight=weights)
        print(f"class balance beta={args.class_balance}: "
              f"weights {weights.min():.2f}-{weights.max():.2f}, "
              f"median {weights.median():.2f}")
    else:
        lossf = nn.BCEWithLogitsLoss()
    total_steps = args.epochs * len(train_dl)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    step, best = 0, 0.0
    for epoch in range(1, args.epochs + 1):
        t0, running = time.time(), 0.0
        for i, (ids, tgt) in enumerate(train_dl):
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            ids, tgt = ids.to(device), tgt.to(device)
            loss = lossf(model(ids), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            step += 1
            if (i + 1) % 200 == 0:
                print(f"  epoch {epoch} step {i+1}/{len(train_dl)} "
                      f"loss {running/(i+1):.4f}", flush=True)

        m = evaluate(model, val_dl, device)
        print(f"epoch {epoch}: train_loss {running/len(train_dl):.4f} | "
              f"val_loss {m['loss']:.4f} | top1 {m['top1']:.3f} | "
              f"recall@5 {m['recall@5']:.3f} | {time.time()-t0:.0f}s", flush=True)

        if m["recall@5"] > best:
            best = m["recall@5"]
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
                       outdir / "best.pt")
            print(f"  saved checkpoint (recall@5 {best:.3f})")

    print(f"\nbest recall@5: {best:.3f}  ->  {outdir/'best.pt'}")


if __name__ == "__main__":
    main()
