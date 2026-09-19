/* hand-tracking.js
 * Pure geometry: turning MediaPipe hand landmarks into a scale/position
 * invariant feature vector, nearest-neighbor classification against the
 * recorded dataset, and skeleton drawing. No DOM/camera concerns here —
 * see camera.js for that.
 */
window.KSL = window.KSL || {};
(function (KSL) {
  "use strict";

  const K_NEIGHBORS = 5;
  const MAX_DIST = 1.35;

  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17]
  ];

  // 21 landmarks -> 42 numbers, translated so the wrist is the origin and
  // scaled so the wrist-to-middle-knuckle distance is 1. Makes the vector
  // independent of where the hand is in frame and how close it is.
  function extractFeatures(lm) {
    const wrist = lm[0], midMcp = lm[9];
    const scale = Math.hypot(midMcp.x - wrist.x, midMcp.y - wrist.y) || 1e-6;
    const feat = [];
    for (const p of lm) feat.push((p.x - wrist.x) / scale, (p.y - wrist.y) / scale);
    return feat;
  }

  // Combines up to two hands (sorted by MediaPipe's Left/Right label for a
  // deterministic order) into one fixed-length 84-number vector, zero-padded
  // when only one hand is present.
  function buildVector(results) {
    if (!results || !results.multiHandLandmarks || !results.multiHandLandmarks.length) {
      return { vector: null, handCount: 0 };
    }
    const hands = results.multiHandLandmarks.map((lm, i) => ({
      lm,
      handedness: (results.multiHandedness && results.multiHandedness[i]) ? results.multiHandedness[i].label : 'Left'
    }));
    hands.sort((a, b) => a.handedness.localeCompare(b.handedness));
    const vec = new Array(84).fill(0);
    hands.slice(0, 2).forEach((h, idx) => {
      const f = extractFeatures(h.lm);
      for (let i = 0; i < 42; i++) vec[idx * 42 + i] = f[i];
    });
    return { vector: vec, handCount: hands.length };
  }

  function dist(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
    return Math.sqrt(s);
  }

  // Weighted k-nearest-neighbors against the recorded dataset, restricted to
  // samples with the same hand count and to the given candidate labels
  // (normally "whatever category is currently selected").
  function classify(vector, handCount, candidateLabels) {
    if (!vector || handCount === 0) return null;
    const pool = KSL.Dataset.samples.filter(s => s.hands === handCount && candidateLabels.includes(s.label));
    if (!pool.length) return null;
    const scored = pool.map(s => ({ label: s.label, d: dist(s.vector, vector) })).sort((a, b) => a.d - b.d);
    const top = scored.slice(0, Math.min(K_NEIGHBORS, scored.length));
    if (top[0].d > MAX_DIST) return null;
    const weight = {};
    let total = 0;
    top.forEach(t => { const w = 1 / (t.d + 0.05); weight[t.label] = (weight[t.label] || 0) + w; total += w; });
    let best = null, bestScore = -1;
    for (const [label, score] of Object.entries(weight)) {
      if (score > bestScore) { bestScore = score; best = label; }
    }
    return { label: best, confidence: total ? bestScore / total : 0, distance: top[0].d };
  }

  function drawSkeleton(ctx, w, h, results) {
    ctx.clearRect(0, 0, w, h);
    if (!results || !results.multiHandLandmarks) return;
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(63,188,232,0.9)';
    ctx.fillStyle = 'rgba(217,154,43,0.95)';
    results.multiHandLandmarks.forEach(lm => {
      ctx.beginPath();
      HAND_CONNECTIONS.forEach(([a, b]) => {
        ctx.moveTo(lm[a].x * w, lm[a].y * h);
        ctx.lineTo(lm[b].x * w, lm[b].y * h);
      });
      ctx.stroke();
      lm.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, 3.2, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  }

  KSL.Hand = { extractFeatures, buildVector, dist, classify, drawSkeleton, K_NEIGHBORS, MAX_DIST };

})(window.KSL);
