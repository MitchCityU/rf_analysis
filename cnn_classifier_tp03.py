#!/usr/bin/env python3
import os
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# paths
SAMPLES_DIR  = Path("/Users/mitchfade/Downloads/dwingeloo/samples")
ANNOT_PATH   = Path("/Users/mitchfade/Downloads/dwingeloo/annot.json")
CLS_MAP_PATH = Path("/Users/mitchfade/Downloads/dwingeloo/cls_map.json")
RAW_DIR      = Path("/Users/mitchfade/Downloads/ai_classifier")

RAW_FILES = [
    ("METEOR", "meteor_processed.iq"),
    ("NOAA15", "noaa15_processed.iq"),
]

BATCH_SIZE     = 16
EPOCHS         = 60
LEARNING_RATE  = 1e-3
TARGET_LEN     = 240_000  # per-channel samples


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def _atomic_write_json(path: Path, data: dict) -> None:
    """Safely write JSON to disk with a temp file and move."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".tmp")
    try:
        with open(tmp.name, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp.name, path)
    finally:
        try:
            if Path(tmp.name).exists():
                os.unlink(tmp.name)
        except Exception:
            pass

def _backup_file(path: Path) -> Path:
    if not path.exists():
        return path
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, backup)
    return backup


def _ensure_2x(iq: np.ndarray) -> np.ndarray:
    """
    Coerce various stored formats to shape (2, N):
    - (2, N): as-is
    - (N, 2): transpose
    - (2N,): interpret as interleaved I/Q -> reshape to (2, N)
    """
    if iq.ndim == 2:
        if iq.shape[0] == 2:
            return iq
        if iq.shape[1] == 2:
            return iq.T
        raise ValueError(f"Unexpected 2D IQ shape {iq.shape}; expected (2, N) or (N, 2).")
    if iq.ndim == 1:
        if iq.size % 2 != 0:
            raise ValueError(f"Odd-length 1D IQ array of size {iq.size}; cannot form I/Q pairs.")
        return iq.reshape(-1, 2).T
    raise ValueError(f"Unsupported IQ array ndim={iq.ndim} with shape {iq.shape}.")

def _center_crop_or_pad_2x(iq_2x: np.ndarray, target_len: int) -> np.ndarray:
    """Ensure (2, target_len). If longer, center-crop; if shorter, symmetric zero-pad."""
    assert iq_2x.shape[0] == 2, f"Expected (2, N), got {iq_2x.shape}"
    n = iq_2x.shape[1]
    if n == target_len:
        return iq_2x
    if n > target_len:
        start = (n - target_len) // 2
        return iq_2x[:, start:start + target_len]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(iq_2x, ((0, 0), (left, right)), mode="constant")

def _standardize_per_sample(iq_2x: np.ndarray) -> np.ndarray:
    """Per-sample standardization across both channels."""
    m = iq_2x.mean()
    s = iq_2x.std() + 1e-6
    return (iq_2x - m) / s

def _try_read_interleaved_iq(path: Path, dtypes=(np.float32, np.int16, np.int8)) -> np.ndarray:
    """
    Try reading interleaved I/Q as (2, N) float32 in [-1, 1] (scaled if integer types).
    Assumes little-endian; tries float32, then int16, then int8 by file-size compatibility.
    """
    size_bytes = path.stat().st_size
    for dt in dtypes:
        itemsize = np.dtype(dt).itemsize
        if size_bytes % (2 * itemsize) != 0:
            continue
        try:
            arr = np.fromfile(path, dtype=dt)
            if arr.size == 0:
                continue
            iq2 = arr.reshape(-1, 2).T.astype(np.float32)  # (2, N)
            if np.issubdtype(dt, np.integer):
                maxv = float(np.iinfo(dt).max)
                if maxv > 0:
                    iq2 = iq2 / maxv
            return iq2
        except Exception:
            pass
    raise ValueError(f"Could not read {path} as interleaved IQ of dtypes {dtypes}.")

# class map and annotation updates
def ensure_classes(cls_map: Dict[str, int], required: List[str]) -> Dict[str, int]:
    """Add missing class names with new indices at the end."""
    changed = False
    next_idx = 0 if not cls_map else (max(cls_map.values()) + 1)
    for name in required:
        if name not in cls_map:
            cls_map[name] = next_idx
            next_idx += 1
            changed = True
            print(f"[INFO] Added class '{name}' to class map.")
    return cls_map

def append_sample_annotation(annotations: Dict[str, dict], sample_id: str, satellite_name: str) -> bool:
    """
    Add {sample_id: {"Satellite": satellite_name}} if not already present.
    :return: True if added; False if existed.
    """
    if sample_id in annotations:
        return False
    annotations[sample_id] = {"Satellite": satellite_name}
    return True

def ingest_raw_iq_as_npy_and_annotate(
    samples_dir: Path,
    raw_dir: Path,
    annotations: Dict[str, dict],
    cls_map: Dict[str, int],
    target_len: int,
    raw_files: List[Tuple[str, str]],
) -> List[str]:
    """
    For each (class_name, filename) in raw_files:
      - If the raw file exists, convert to (2, TARGET_LEN) float32 numpy array and
        save under SAMPLES_DIR as extra_{CLASS}_{ts}.npy
      - Add an annotation entry for this sample_id with "Satellite": class_name
    Returns list of sample_ids added (for logging).
    """
    added_ids: List[str] = []
    for class_name, filename in raw_files:
        raw_path = raw_dir / filename
        if not raw_path.exists():
            print(f"[WARN] Skipping: raw file not found: {raw_path}")
            continue
        print(f"[INFO] Ingesting raw IQ for class '{class_name}': {raw_path.name}")
        iq = _try_read_interleaved_iq(raw_path)
        iq = _center_crop_or_pad_2x(iq, target_len)
        ts = int(time.time())
        sample_id = f"extra_{class_name}_{ts}"
        out_path = samples_dir / f"{sample_id}.npy"
        samples_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_path, iq.astype(np.float32))
        if append_sample_annotation(annotations, sample_id, class_name):
            added_ids.append(sample_id)
            print(f"[INFO] Saved {out_path.name} and annotated as Satellite='{class_name}'")
        else:
            print(f"[INFO] Annotation for {sample_id} already exists; not duplicated.")
    return added_ids

class IQDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        iq = np.load(path)
        iq = _ensure_2x(iq).astype(np.float32)
        iq = _center_crop_or_pad_2x(iq, TARGET_LEN)
        iq = _standardize_per_sample(iq).astype(np.float32)
        return torch.from_numpy(iq), torch.tensor(label, dtype=torch.long)

def safe_train_test_split(data, test_size=0.2, random_state=42):
    """
    data: list[(path, label_idx)]
    - Classes with <2 samples go entirely into TRAIN.
    - If remaining eligible set cannot be stratified, fall back to simple split.
    """
    lbl_counts = Counter(lbl for _, lbl in data)
    eligible = [(p, l) for (p, l) in data if lbl_counts[l] >= 2]
    rare     = [(p, l) for (p, l) in data if lbl_counts[l] <  2]

    elig_labels = [l for _, l in eligible]
    can_stratify = len(eligible) > 0 and len(set(elig_labels)) > 1 and all(
        Counter(elig_labels)[c] >= 2 for c in set(elig_labels)
    )

    if can_stratify:
        min_test_frac = len(set(elig_labels)) / max(1, len(eligible))
        ts = max(test_size, min_test_frac)
        ts = min(ts, 0.5)
        tr_e, te_e = train_test_split(
            eligible,
            test_size=ts,
            random_state=random_state,
            stratify=elig_labels
        )
    else:
        tr_e, te_e = train_test_split(
            eligible,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

    train = tr_e + rare
    test  = te_e

    if rare:
        rare_classes = sorted({l for _, l in rare})
        print(f"[INFO] {len(rare)} rare samples moved to TRAIN only. Classes: {rare_classes}")
    print(f"[INFO] Split sizes -> train: {len(train)}  test: {len(test)}")
    return train, test

# model
class DeepIQCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.net(x))

def main():
    if not ANNOT_PATH.exists():
        raise FileNotFoundError(f"Missing annot file: {ANNOT_PATH}")
    if not CLS_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing class map: {CLS_MAP_PATH}")

    annotations = _load_json(ANNOT_PATH)
    cls_map = _load_json(CLS_MAP_PATH)

    before = dict(cls_map)
    cls_map = ensure_classes(cls_map, ["METEOR", "NOAA15"])
    if cls_map != before:
        _backup_file(CLS_MAP_PATH)
        _atomic_write_json(CLS_MAP_PATH, cls_map)
        print(f"[INFO] Updated class map written to: {CLS_MAP_PATH}")

    label_map = cls_map
    inv_label_map = {v: k for k, v in label_map.items()}

    added_ids = ingest_raw_iq_as_npy_and_annotate(
        samples_dir=SAMPLES_DIR,
        raw_dir=RAW_DIR,
        annotations=annotations,
        cls_map=cls_map,
        target_len=TARGET_LEN,
        raw_files=RAW_FILES,
    )

    if added_ids:
        _backup_file(ANNOT_PATH)
        _atomic_write_json(ANNOT_PATH, annotations)
        print(f"[INFO] Updated annotations written to: {ANNOT_PATH}")
        print(f"[INFO] Added samples: {', '.join(added_ids)}")

    print("First 5 annotations:")
    for i, (sample_id, meta) in enumerate(annotations.items()):
        print(f"{i}: {sample_id} -> {meta}")
        if i >= 4:
            break

    data: List[Tuple[str, int]] = []
    for sample_id, meta in annotations.items():
        npy_name = f"{sample_id}.npy"
        full_path = SAMPLES_DIR / npy_name
        label_name = meta.get("Satellite")
        if full_path.exists() and label_name in label_map:
            data.append((str(full_path), label_map[label_name]))

    print(f"Loaded {len(data)} samples")
    if not data:
        raise RuntimeError("No matching samples found. Check your paths and annotations.")

    train_data, test_data = safe_train_test_split(data, test_size=0.2, random_state=42)

    train_loader = DataLoader(
        IQDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )
    test_loader  = DataLoader(
        IQDataset(test_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_map)
    model = DeepIQCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 8) Train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {total_loss / max(1, len(train_loader)):.4f}")

    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().tolist()
            y_true_all.extend(y.tolist())
            y_pred_all.extend(preds)

    overall_acc = 100.0 * sum(int(t == p) for t, p in zip(y_true_all, y_pred_all)) / max(1, len(y_true_all))
    print(f"Accuracy: {overall_acc:.2f}%")

    num_classes = len(inv_label_map)
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_all, y_pred_all):
        conf[t, p] += 1

    row_sums = conf.sum(axis=1, keepdims=True).clip(min=1)
    conf_pct = conf / row_sums * 100.0

    idx_to_name = [inv_label_map.get(i, f"<cls_{i}>") for i in range(num_classes)]

    def print_percentage_report(conf_counts: np.ndarray, conf_percent: np.ndarray):
        print("\n=== Percentage Prediction Report (row = TRUE, columns = PREDICTED) ===")
        col_header = "TRUE \\ PRED".ljust(22) + " | " + " | ".join(name[:14].ljust(14) for name in idx_to_name)
        print(col_header)
        print("-" * len(col_header))
        for i, true_name in enumerate(idx_to_name):
            row_pct = " | ".join(f"{conf_percent[i, j]:6.1f}%" for j in range(num_classes))
            print(f"{true_name[:22].ljust(22)} | {row_pct}")
        print("\n(Note) Rows sum to ~100% (per true class).")

        print("\n=== Per-Satellite Recall (Top-1 accuracy per true class) ===")
        for i, true_name in enumerate(idx_to_name):
            support = conf_counts[i, :].sum()
            recall = (conf_counts[i, i] / support * 100.0) if support > 0 else 0.0
            print(f"{true_name[:22].ljust(22)} : recall={recall:6.2f}%   support={support}")

    print_percentage_report(conf, conf_pct)

    from collections import Counter as _Counter
    pred_counter = _Counter(y_pred_all)
    true_counter = _Counter(y_true_all)

    def print_breakdown(counter: _Counter, title: str):
        print(f"\n=== {title} ===")
        total_items = sum(counter.values())
        for idx, cnt in counter.most_common():
            name = inv_label_map.get(idx, f"<cls_{idx}>")
            pct = 100.0 * cnt / max(1, total_items)
            print(f"{name:<22} : {cnt:5d}  ({pct:5.1f}%)")

    print_breakdown(true_counter, "Test set TRUE label distribution")
    print_breakdown(pred_counter, "Test set PREDICTED label distribution")

    def load_iq_for_inference(iq_path: Path, target_len: int = TARGET_LEN) -> torch.Tensor:
        iq = _try_read_interleaved_iq(iq_path)
        iq = _center_crop_or_pad_2x(iq, target_len)
        iq = _standardize_per_sample(iq).astype(np.float32)
        tens = torch.from_numpy(iq)  # (2, L)
        return tens.unsqueeze(0)      # (1, 2, L)

    def topk_readout(logits: torch.Tensor, k: int = 5):
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (C,)
        vals, idxs = torch.topk(probs, k=min(k, probs.numel()))
        return [(int(i), float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]

    def print_topk(name: str, logits: torch.Tensor, k: int = 5):
        pairs = topk_readout(logits, k=k)
        print(f"\nTop-{len(pairs)} predictions for {name}:")
        for cls_idx, p in pairs:
            label = inv_label_map.get(cls_idx, f"<cls_{cls_idx}>")
            print(f"  {label:<20}  p={p:.4f}")

    for sat_name, fname in RAW_FILES:
        raw_path = RAW_DIR / fname
        if raw_path.exists():
            with torch.no_grad():
                x = load_iq_for_inference(raw_path, TARGET_LEN).to(device)
                logits = model(x)
                print_topk(raw_path.name, logits, k=5)
                pred_idx = int(logits.argmax(1).item())
                pred_label = inv_label_map.get(pred_idx, str(pred_idx))
                print(f"Predicted label for {raw_path.name}: {pred_label}")

if __name__ == "__main__":
    main()   


