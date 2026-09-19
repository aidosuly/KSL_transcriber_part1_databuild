# Roadmap

This project is being built in three phases. Phase 1 is what's in this repo today.

## Phase 1 — Trainable live transcriber (this repo)

Tracks your hand with MediaPipe Hands, lets you record a few examples of
each sign, and then recognizes signs live against those recordings with a
nearest-neighbor matcher. Two implementations, same dataset format: a
browser version (`index.html`) and a Python/Tkinter desktop version
(`python/`). Covers:

- The 42-letter Kazakh Cyrillic alphabet (dactyl fingerspelling)
- Digits 0–9
- ~20 essential words, plus any word you add yourself through the UI

There is no pretrained KSL model behind this — none exists that could simply
be embedded — so the app learns from whoever trains it. Every recorded
sample is a labeled data point (`{label, category, hands, landmark vector}`),
exportable as JSON from the Data tab.

**Known limits:** accuracy depends entirely on how much and how consistently
you train it; it recognizes static hand shapes, not motion, so signs that
depend on movement rather than a held pose won't be captured well yet; and
it's a single-person tool — the dataset lives in one browser's local storage.

## Phase 2 — Corpus from real footage

Goal: turn hours of KSL-interpreted news broadcasts into a large, real
labeled dataset instead of one person's recordings.

Rough pipeline:

1. Pull audio from KSL-interpreted news video and run speech-to-text to get
   a timestamped transcript of what's being said.
2. Run hand/pose tracking (this repo's `js/hand-tracking.js` approach, or a
   heavier model like MediaPipe Holistic, extended to also track motion
   over time rather than a single static pose) over the interpreter's video
   track.
3. Align spoken segments to the interpreter's signing with human review —
   the speech-to-text gives rough timing, a person confirms which sign(s)
   correspond to which word(s) or phrase(s).
4. Store the result as labeled sign clips (not just static poses): a
   sequence of landmark frames plus the aligned word or phrase.

This produces the dataset Phase 1's manual recording can't: broad vocabulary
coverage, multiple signers, and real motion data — the input a proper
sequence model (e.g. an LSTM/transformer over landmark sequences) needs.

**Not yet built.** This needs a video/audio processing pipeline (likely
Python, off-browser) and storage for a much larger dataset than
localStorage — out of scope for this static-site repo as it stands.

## Phase 3 — Two-way voiceover

Goal: let a hearing person and a KSL signer have a live conversation.

- **Sign → speech:** recognize signs live (as in Phase 1, but with the
  Phase 2 model) and read the resulting text aloud with text-to-speech.
- **Speech → sign:** take spoken Kazakh, and either play back matching
  recorded sign clips or drive a signing avatar, so the hearing person's
  side of the conversation is visible to the KSL signer too.

This is the actual product goal — Phases 1 and 2 exist to build the model
and dataset this needs.
