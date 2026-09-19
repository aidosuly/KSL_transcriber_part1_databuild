# Architecture (Phase 1, browser version)

This covers the browser implementation (`index.html`, `css/`, `js/`). The
Python desktop version mirrors the same modules and math one-for-one —
see [`../python/README.md`](../python/README.md) for its layout.

Plain static site, no build step, no framework. Files are loaded as classic
`<script>` tags (not ES modules) specifically so `index.html` can be opened
directly with a double-click — ES modules require a real HTTP origin and
fail under `file://`.

## Files

- `index.html` — markup only; loads MediaPipe Hands from jsDelivr, then the
  app scripts in dependency order.
- `css/styles.css` — all styling, theme-aware (light/dark via
  `prefers-color-scheme`).
- `js/labels.js` — the built-in alphabet/digit/word lists, plus custom-word
  management (add/remove), persisted under the `kslCustomWords_v1`
  localStorage key. Exposes `window.KSL.CATEGORIES`, `KSL.addWord()`,
  `KSL.removeCustomWord()`.
- `js/dataset.js` — the recorded training samples store, persisted under
  `kslTranscriberDB_v1`. Exposes `window.KSL.Dataset` with `addSamples()`,
  `countFor()`, `removeLabel()`, `resetAll()`, `exportJSON()`,
  `importJSON()`.
- `js/hand-tracking.js` — pure geometry and matching: turns MediaPipe
  landmarks into a normalized feature vector, and does k-nearest-neighbor
  classification against `KSL.Dataset`. No DOM code. Exposes
  `window.KSL.Hand`.
- `js/camera.js` — owns the webcam stream and the MediaPipe `Hands`
  instance; runs the detection loop and calls back into `app.js` with each
  frame's results. Exposes `window.KSL.Camera`.
- `js/app.js` — everything UI: tabs, the label grid, recording bursts,
  the recognize loop's stability/auto-commit logic, the transcript buffer,
  and the data tab. Depends on all of the above being loaded first.

All modules hang off a single `window.KSL` namespace object instead of using
ES `import`/`export`, so load order in `index.html` matters:
`labels.js` → `dataset.js` → `hand-tracking.js` → `camera.js` → `app.js`.

## The feature vector

MediaPipe gives 21 (x, y, z) landmarks per detected hand, in image-relative
coordinates. Two hands raw would depend on where your hands are in frame and
how close to the camera — useless for matching. `extractFeatures()` in
`hand-tracking.js` fixes that:

1. Subtract the wrist landmark (index 0) from every point — translation
   invariant.
2. Divide by the distance from the wrist to the middle-finger knuckle
   (index 9) — scale invariant.

That's 21 × 2 = 42 numbers per hand (z is dropped — it's noisier and mostly
redundant with x/y for this purpose). Two hands are sorted by MediaPipe's
`Left`/`Right` handedness label for a deterministic order and concatenated
into an 84-number vector, zero-padding the second half when only one hand is
present. Samples and live frames are only ever compared when they have the
same hand count, so the zero-padding never distorts a distance calculation.

## Classification

`KSL.Hand.classify(vector, handCount, candidateLabels)` computes the
Euclidean distance from the live vector to every stored sample with a
matching hand count and a label in the current category, takes the 5
closest, and lets them vote weighted by inverse distance. If even the
closest sample is farther than a fixed threshold, it returns `null` (no
confident match) rather than guessing.

This is deliberately simple — no training step, no model file, just
distance to examples you recorded. It's also exactly why Phase 2 exists:
it doesn't generalize beyond what one person has recorded, and it has no
notion of motion, only a held pose.

## Extending it

- **Add a word without touching code:** use the "+ Add word" box in the
  Train tab (Words category). It's exactly the feature described in the
  main README.
- **Add a new label programmatically:** call `KSL.addWord(kk, gloss)` from
  the console, or extend `BUILTIN_WORDS` / `ALPHABET` / `DIGITS` in
  `js/labels.js` directly for anything meant to ship with the app rather
  than be user-added.
- **Swap the classifier:** everything downstream only calls
  `KSL.Hand.classify()`; replacing k-NN with something else (e.g. a small
  trained model once Phase 2's dataset exists) only means rewriting that
  one function's internals, not the UI.
