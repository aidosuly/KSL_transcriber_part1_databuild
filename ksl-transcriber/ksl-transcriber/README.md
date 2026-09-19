# KSL Live Transcriber

A trainable live transcriber for Kazakh Sign Language (KSL). It tracks your
hand with MediaPipe, learns signs from a handful of examples you record
yourself, and recognizes them live to build up a Kazakh-language transcript
— no translation step needed, since the signs already are Kazakh words.

This is **Phase 1** of a three-phase plan. See [`docs/roadmap.md`](docs/roadmap.md)
for Phases 2 (building a real dataset from KSL-interpreted news footage) and
3 (two-way voiceover, so a hearing person and a KSL signer can talk
directly). [`docs/architecture.md`](docs/architecture.md) covers how the
code is organized and how the recognition actually works.

Two implementations of Phase 1 live in this repo, built to the same feature
set and the same dataset format (a `.json` exported from one imports
straight into the other):

- **Browser version** (this folder, `index.html`) — zero install, opens in
  Chrome/Edge, easiest to try or hand to someone else.
- **Python desktop version** ([`python/`](python/)) — OpenCV + MediaPipe +
  Tkinter. More setup (a venv, a few pip installs), but it's the natural
  base for Phase 2's video/audio processing pipeline, which will be Python
  regardless. See [`python/README.md`](python/README.md).

The rest of this file covers the browser version.

## Quick start

No build step, no dependencies. Just open `index.html` in Chrome or Edge.

If your browser blocks the camera when opening the file directly (some
browsers restrict `getUserMedia` on `file://` origins), serve the folder
instead:

```bash
# option A — Node
npm start

# option B — Python, no install needed
python3 -m http.server
# then open http://localhost:8000
```

Click **Start camera** and allow access when prompted.

## Using it

1. **Train.** Pick a category (Alphabet / Numbers / Words), pick a label,
   hold that hand shape steady in frame, and click **Record (1.5s)**. Do
   this 5–10 times per label, from slightly different angles, for more
   reliable matching. The badge on each label shows how many samples
   you've recorded for it.
2. **Add your own words.** Switch to the Words category in the Train tab —
   a small form appears above the label grid. Type the Kazakh word and an
   optional gloss (its meaning, for your own reference) and click **+ Add
   word**. It's immediately available to train and recognize, right next to
   the built-in words. Custom words can be removed with the small **×** on
   their tile; this also deletes any samples recorded for that word.
3. **Recognize.** Switch to the Recognize tab, turn on **Recognition
   running**, and hold up a sign. The predicted label and a confidence bar
   appear on the camera panel. Press **Space** (or click **+ Add**) to add
   it to the transcript, or turn on **Auto-commit** to have it added
   automatically once held steady for about a third of a second.
4. **Transcript.** Alphabet/number signs build up in the buffer line; press
   **Enter** (or **Finish word**) to commit the spelled word into the
   transcript. Whole-word signs go straight into the transcript. Use
   **Backspace**, **Clear**, or **Copy** as needed.
5. **Data.** The Data tab shows sample counts per label and overall
   coverage. **Export dataset** downloads everything as JSON — this is the
   real, growing dataset that Phase 2's model training will eventually
   build on, so export it regularly. **Import dataset** merges or replaces
   from a previous export (handy for moving your data to another browser
   or machine). **Reset all data** clears recorded samples only — your
   custom word list is untouched.

## How recognition works, briefly

Every video frame, MediaPipe returns 21 landmark points per detected hand.
Those get normalized (translation- and scale-invariant) into a fixed-length
vector, and a live vector is matched against your recorded samples with
k-nearest-neighbors. There's no pretrained model — it only knows what you've
taught it. Full details in [`docs/architecture.md`](docs/architecture.md).

## Project layout

```
index.html          Markup + script/CDN loading
css/styles.css       All styling (light + dark)
js/labels.js         Built-in label sets + custom word add/remove
js/dataset.js        Recorded training samples: store, export, import
js/hand-tracking.js  Feature extraction + k-NN classifier (no DOM code)
js/camera.js         Webcam + MediaPipe Hands wiring
js/app.js            UI wiring: tabs, training, recognition loop, transcript
docs/roadmap.md      The three-phase plan in detail
docs/architecture.md How the code fits together
data/                Suggested drop point for dataset exports you want to version
```

## Known limits

- Recognizes held hand shapes, not motion — signs that rely on movement
  rather than a static pose won't be captured well yet (Phase 2 territory).
- Accuracy depends entirely on how much and how consistently you train it.
- Dataset lives in one browser's local storage; use Export/Import to move
  it or back it up.
