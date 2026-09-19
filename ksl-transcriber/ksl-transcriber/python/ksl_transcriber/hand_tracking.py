"""Geometry: MediaPipe landmarks -> normalized feature vector -> k-NN classification.

Mirrors js/hand-tracking.js exactly (same normalization, same constants) so
a dataset recorded in one implementation is comparable to the other, and so
Phase 2 can reuse whichever side's data.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

K_NEIGHBORS = 5
MAX_DIST = 1.35

# 21-point MediaPipe hand skeleton connections, for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def extract_features(landmarks: Sequence) -> List[float]:
    """21 (x, y) landmarks -> 42 numbers, wrist-relative and scale-normalized.

    landmarks[i] must expose .x and .y (MediaPipe's NormalizedLandmark does).
    """
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    scale = math.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y) or 1e-6
    feat: List[float] = []
    for p in landmarks:
        feat.append((p.x - wrist.x) / scale)
        feat.append((p.y - wrist.y) / scale)
    return feat


def build_vector(hand_landmark_lists: Sequence[Sequence], handedness_labels: Sequence[str]) -> tuple:
    """Combine up to two hands into a fixed 84-number vector.

    Deliberately takes plain data rather than a MediaPipe result object:
    ``hand_landmark_lists`` is a list of hands, each a list of 21 objects
    exposing ``.x``/``.y`` (works whether the caller's landmark type comes
    from an old or new MediaPipe API, or elsewhere entirely), and
    ``handedness_labels`` is a parallel list of ``"Left"``/``"Right"``
    strings used only to fix a deterministic hand order. Hands are sorted
    by that label; a missing second hand is zero-padded. Returns
    (vector_or_None, hand_count).
    """
    if not hand_landmark_lists:
        return None, 0

    labels = list(handedness_labels) if handedness_labels else []
    labels += ['Left'] * (len(hand_landmark_lists) - len(labels))
    hands = sorted(zip(labels, hand_landmark_lists), key=lambda h: h[0])

    vec = [0.0] * 84
    for idx, (_, lms) in enumerate(hands[:2]):
        feat = extract_features(lms)
        vec[idx * 42: idx * 42 + 42] = feat
    return vec, len(hands)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def classify(vector: Optional[List[float]], hand_count: int, candidate_labels: Sequence[str], samples: Sequence[dict]) -> Optional[dict]:
    """Weighted k-nearest-neighbors against recorded samples.

    Only compares against samples with the same hand_count and a label in
    candidate_labels. Returns None (no confident match) rather than
    guessing when even the closest sample is farther than MAX_DIST.
    """
    if not vector or hand_count == 0:
        return None
    pool = [s for s in samples if s['hands'] == hand_count and s['label'] in candidate_labels]
    if not pool:
        return None
    scored = sorted(({'label': s['label'], 'd': _dist(s['vector'], vector)} for s in pool), key=lambda x: x['d'])
    top = scored[:min(K_NEIGHBORS, len(scored))]
    if top[0]['d'] > MAX_DIST:
        return None
    weight: dict = {}
    total = 0.0
    for t in top:
        w = 1.0 / (t['d'] + 0.05)
        weight[t['label']] = weight.get(t['label'], 0.0) + w
        total += w
    best_label = max(weight, key=weight.get)
    confidence = (weight[best_label] / total) if total else 0.0
    return {'label': best_label, 'confidence': confidence, 'distance': top[0]['d']}
