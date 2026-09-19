"""Recorded training samples: store, export, import.

Mirrors js/dataset.js. Each sample is
{"label": str, "category": str, "hands": int, "vector": [float, ...], "ts": int}
persisted as JSON on disk instead of localStorage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_FILE = DATA_DIR / "dataset.json"


class Dataset:
    def __init__(self, path: Path | None = None):
        # Resolved at call time rather than as a default argument, so a
        # test (or a future caller) can point at a different DATASET_FILE
        # without needing to construct the path itself.
        self.path = path if path is not None else DATASET_FILE
        self.samples: List[dict] = []
        self.load()

    def load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding='utf-8'))
                samples = data.get('samples') if isinstance(data, dict) else None
                self.samples = samples if isinstance(samples, list) else []
            else:
                self.samples = []
        except (json.JSONDecodeError, OSError):
            self.samples = []

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({'samples': self.samples}, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    def count_for(self, label: str) -> int:
        return sum(1 for s in self.samples if s['label'] == label)

    def add_samples(self, label: str, category: str, hands: int, vectors: List[List[float]]) -> int:
        added = 0
        now = int(time.time() * 1000)
        for vec in vectors:
            self.samples.append({'label': label, 'category': category, 'hands': hands, 'vector': vec, 'ts': now})
            added += 1
        if added:
            self.save()
        return added

    def remove_label(self, label: str) -> None:
        self.samples = [s for s in self.samples if s['label'] != label]
        self.save()

    def reset_all(self) -> None:
        self.samples = []
        self.save()

    def export_json(self) -> str:
        return json.dumps({'samples': self.samples}, ensure_ascii=False, indent=2)

    def import_json(self, text: str, merge: bool) -> None:
        incoming = json.loads(text)
        samples = incoming.get('samples') if isinstance(incoming, dict) else None
        if not isinstance(samples, list):
            raise ValueError('File does not look like a valid dataset export.')
        self.samples = (self.samples + samples) if merge else samples
        self.save()
