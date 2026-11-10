import hashlib
import json
from pathlib import Path


def load_rows(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Dataset must be nonempty with unique sample IDs")
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def register_device(device):
    if device == "npu":
        import torch_npu  # noqa: F401


def disable_dropout(model):
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def reset_peak_memory(device):
    import torch

    backend = getattr(torch, device, None)
    if device in ("cuda", "npu") and backend is not None:
        backend.synchronize()
        backend.reset_peak_memory_stats()


def peak_memory(device):
    import torch

    backend = getattr(torch, device, None)
    if device in ("cuda", "npu") and backend is not None:
        backend.synchronize()
        return {
            "peak_allocated_bytes": backend.max_memory_allocated(),
            "peak_reserved_bytes": backend.max_memory_reserved(),
        }
    return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
