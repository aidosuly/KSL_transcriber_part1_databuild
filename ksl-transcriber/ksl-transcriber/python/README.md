# KSL Live Transcriber — Python desktop version

Same Phase 1 idea as the browser version in the repo root — train it on a
handful of examples per sign, then recognize live and build a Kazakh
transcript — but as a Python/Tkinter desktop app using OpenCV + MediaPipe
instead of a browser. See the [top-level README](../README.md) and
[`docs/roadmap.md`](../docs/roadmap.md) for the overall project context.

Why a Python version at all, alongside the browser one: Phase 2 (building a
real dataset from KSL-interpreted news footage — speech-to-text, alignment,
a much bigger corpus) is naturally a Python project. Having Phase 1 already
in Python too means the hand-tracking and classifier code
(`ksl_transcriber/hand_tracking.py`) can be reused directly when that
pipeline gets built, rather than ported from JavaScript at that point.

## Setup

Requires Python 3.10+.

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tkinter itself ships with most Python installs, but on some Linux
distributions it's a separate system package:

```bash
# Debian/Ubuntu
sudo apt-get install python3-tk
```

## Run

```bash
python -m ksl_transcriber
```

Click **Start camera** and allow access if your OS prompts for it.

## Using it

Same workflow as the browser version:

1. **Train tab** — pick a category (Alphabet / Numbers / Words), select a
   label, hold the sign steady, click **Record (1.5s)**. Repeat 5-10x per
   label from slightly different angles.
2. **Add your own words** — switch to the Words category; a small form
   appears above the label grid. Type the word and its meaning, click
   **+ Add word**. Remove a custom word (and its samples) with the small
   **×** on its tile. Built-in words can't be removed this way.
3. **Recognize tab** — turn on **Recognition running**, hold up a sign. The
   predicted label and confidence show on the camera panel. Press **Space**
   (or **+ Add**) to commit it, or turn on **Auto-commit** to add stable
   signs automatically.
4. **Transcript** — letters/digits build up in the buffer; **Enter** (or
   **Finish word**) commits the spelled word. Whole-word signs go straight
   into the transcript. **Backspace**, **Clear**, **Copy** as needed.
5. **Data tab** — sample counts per label, overall coverage, and
   **Export/Import dataset** (JSON, same format as the browser version) and
   **Reset all data** (clears recordings only, keeps your custom words).

## Where data lives

- `data/dataset.json` — your recorded training samples.
- `data/custom_words.json` — words you've added beyond the built-in list.

Both are plain JSON and gitignored by default (they're personal recordings,
not code). Use the Data tab's Export button to save a dated copy into the
top-level `../data/` folder if you want to version a snapshot.

**The dataset format is identical to the browser version's export**, so a
`.json` exported from one can be imported into the other.

## Project layout

```
python/
├── ksl_transcriber/
│   ├── labels.py         Built-in label sets + custom word add/remove
│   ├── dataset.py         Recorded samples: store, export, import (JSON on disk)
│   ├── hand_tracking.py   Feature extraction + k-NN classifier (no GUI code, unit-tested)
│   ├── app.py             Tkinter GUI + OpenCV/MediaPipe camera loop
│   ├── __main__.py        Entry point for `python -m ksl_transcriber`
│   └── __init__.py
├── tests/
│   └── test_hand_tracking.py   Unit tests for the geometry/classifier core
├── data/                  Working storage (gitignored, see above)
└── requirements.txt
```

## Tests

The geometry and classification logic (`hand_tracking.py`) is unit-tested
without needing a camera or a display:

```bash
python -m unittest discover -s tests -v
```

## Notes on how it works

Identical approach to the browser version — same normalization, same
k-nearest-neighbor matching, same constants — so a recorded dataset behaves
the same in either implementation. See
[`../docs/architecture.md`](../docs/architecture.md) for the full
explanation; the only real difference here is OpenCV/Tkinter standing in
for the browser's camera APIs and DOM.
