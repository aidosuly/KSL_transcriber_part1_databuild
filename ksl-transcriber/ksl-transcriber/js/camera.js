/* camera.js
 * Owns the webcam + MediaPipe Hands wiring: starting/stopping the stream,
 * running the detection loop, and drawing the skeleton overlay. Emits every
 * frame's results to whoever calls init() with a callback.
 */
window.KSL = window.KSL || {};
(function (KSL) {
  "use strict";

  const MEDIAPIPE_VERSION = '0.4.1675469240';

  let handsInstance = null;
  let stream = null;
  let running = false;
  let busy = false;
  let videoEl = null, overlayEl = null, octx = null;
  let onFrame = null;

  function init(video, overlay, onFrameCallback) {
    videoEl = video;
    overlayEl = overlay;
    octx = overlay.getContext('2d');
    onFrame = onFrameCallback;
  }

  function ensureHands() {
    if (handsInstance) return true;
    if (typeof Hands === 'undefined') return false;
    handsInstance = new Hands({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${MEDIAPIPE_VERSION}/${f}`
    });
    handsInstance.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.5 });
    handsInstance.onResults((results) => {
      KSL.Hand.drawSkeleton(octx, overlayEl.width, overlayEl.height, results);
      if (onFrame) onFrame(results);
    });
    return true;
  }

  async function frameLoop() {
    if (!running) return;
    if (!busy) {
      busy = true;
      try { await handsInstance.send({ image: videoEl }); } catch (e) { /* transient, ignore */ }
      busy = false;
    }
    requestAnimationFrame(frameLoop);
  }

  async function start() {
    if (!ensureHands()) {
      const err = new Error('MediaPipe Hands failed to load from the CDN.');
      err.code = 'MEDIAPIPE_UNAVAILABLE';
      throw err;
    }
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    videoEl.srcObject = stream;
    await videoEl.play();
    overlayEl.width = videoEl.videoWidth || 640;
    overlayEl.height = videoEl.videoHeight || 480;
    running = true;
    frameLoop();
  }

  function stop() {
    running = false;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (videoEl) videoEl.srcObject = null;
    if (octx && overlayEl) octx.clearRect(0, 0, overlayEl.width, overlayEl.height);
  }

  function isRunning() { return running; }

  KSL.Camera = { init, start, stop, isRunning };

})(window.KSL);
