"""Write report/status.json so the status page has something live to poll.

Everything here is derived from files on disk -- training logs, the relabel
output -- so it keeps working across restarts and does not need the training
scripts to know about it.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPOCH = re.compile(r"^epoch (\d+): .*val_loss ([\d.]+).*recall@5 ([\d.]+).*?(\d+)s", re.M)
RUNS = [("v6", "/tmp/v6.log"), ("v8", "/tmp/v8.log"), ("v9", "/tmp/v9.log")]


def job_running(pattern: str) -> bool:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return bool(out.stdout.strip())


def read_run(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="ignore")
    epochs = [{"epoch": int(e), "val_loss": float(v), "recall": float(r), "secs": int(s)}
              for e, v, r, s in EPOCH.findall(text)]
    best = re.search(r"^best recall@5: ([\d.]+)", text, re.M)
    step = re.findall(r"epoch (\d+) step (\d+)/(\d+)", text)
    return {
        "epochs": epochs,
        "done": bool(best),
        "best": float(best.group(1)) if best else None,
        "step": {"epoch": int(step[-1][0]), "at": int(step[-1][1]),
                 "of": int(step[-1][2])} if step and not best else None,
    }


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def snapshot() -> dict:
    relabeled = count_lines(ROOT / "data" / "relabeled.jsonl")
    rate = None
    log = Path("/tmp/lm_full.log")
    if log.exists():
        import re as _re
        m = _re.findall(r"(\d+)/(\d+) \[[\d:]+<([\d:]+),\s*([\d.]+)(s/msg|msg/s)",
                        log.read_text(encoding="utf-8", errors="ignore")[-4000:])
        if m:
            at, of, eta, speed, unit = m[-1]
            per_min = 60 / float(speed) if unit == "s/msg" else 60 * float(speed)
            rate = {"eta": eta, "per_min": round(per_min)}
    return {
        "relabel_rate": rate,
        "updated": time.strftime("%H:%M:%S"),
        "jobs": {
            "training": job_running("emoji_model.train"),
            "relabelling": job_running("emoji_model.relabel"),
        },
        "relabel": {"done": relabeled, "total": 102726},
        "runs": {name: read_run(path) for name, path in RUNS},
    }


def main() -> None:
    out = ROOT / "report" / "status.json"
    while True:
        out.write_text(json.dumps(snapshot(), indent=1), encoding="utf-8")
        time.sleep(10)


if __name__ == "__main__":
    main()
