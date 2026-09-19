/* app.js
 * UI wiring: tabs, label grid, training bursts, recognition loop, transcript
 * building, the data tab, and the "add a new word" flow. Depends on labels.js,
 * dataset.js, hand-tracking.js and camera.js being loaded first.
 */
(function (KSL) {
  "use strict";

  /* ---------------- state ---------------- */
  let trainCategory = 'alphabet';
  let recogCategory = 'alphabet';
  let trainSelectedLabel = null;
  let latestResults = null;
  let recognitionOn = false;
  let autoCommit = false;
  let buffer = '';
  let history = [];
  let lastCommittedLabel = null;
  let handClearedSinceCommit = true;
  let currentPrediction = null;

  const STABLE_FRAMES = 10;

  /* ---------------- flash toast ---------------- */
  let flashTimer;
  function flash(msg) {
    const el = document.getElementById('flash');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => el.classList.remove('show'), 1800);
  }

  /* ---------------- DOM refs ---------------- */
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const camhint = document.getElementById('camhint');
  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');
  const handBadge = document.getElementById('handBadge');
  const predSymbol = document.getElementById('predSymbol');
  const predLabelText = document.getElementById('predLabelText');
  const predConf = document.getElementById('predConf');
  const modeLabel = document.getElementById('modeLabel');
  const recogLabel = document.getElementById('recogLabel');
  const catRowTrain = document.getElementById('catRowTrain');
  const catRowRecog = document.getElementById('catRowRecog');
  const labelGrid = document.getElementById('labelGrid');
  const trainCur = document.getElementById('trainCur');
  const trainGloss = document.getElementById('trainGloss');
  const btnRecord = document.getElementById('btnRecord');
  const addWordPanel = document.getElementById('addWordPanel');
  const newWordKk = document.getElementById('newWordKk');
  const newWordGloss = document.getElementById('newWordGloss');
  const bufferText = document.getElementById('bufferText');
  const transcriptEl = document.getElementById('transcript');

  /* ---------------- camera wiring ---------------- */
  KSL.Camera.init(video, overlay, onFrame);

  function onFrame(results) {
    latestResults = results;
    const n = results.multiHandLandmarks ? results.multiHandLandmarks.length : 0;
    handBadge.textContent = 'Hands: ' + n;

    if (recognitionOn) {
      const { vector, handCount } = KSL.Hand.buildVector(results);
      const candidates = KSL.CATEGORIES[recogCategory].items.map(i => i.label);
      const result = KSL.Hand.classify(vector, handCount, candidates);
      currentPrediction = result;
      updatePredictionUI(result);
      pushHistory(result ? result.label : null);
      if (autoCommit) maybeAutoCommit();
    } else {
      currentPrediction = null;
      updatePredictionUI(null);
    }
  }

  async function startCamera() {
    try {
      await KSL.Camera.start();
    } catch (e) {
      camhint.classList.remove('hidden');
      if (e && e.code === 'MEDIAPIPE_UNAVAILABLE') {
        camhint.innerHTML = 'Could not load the hand-tracking library from the CDN.<br>Check your internet connection, then reload this page.';
      } else {
        camhint.innerHTML = 'Camera access was blocked or is unavailable.<br>' +
          '<span style="font-size:0.78rem;opacity:.8;">' + (e && e.message ? e.message : e) + '</span><br>' +
          '<span style="font-size:0.78rem;opacity:.8;">If nothing happens, try serving this folder with a local server ' +
          '(<code>npm start</code>, or <code>python3 -m http.server</code>) instead of opening index.html directly.</span>';
      }
      return;
    }
    camhint.classList.add('hidden');
    btnStart.disabled = true;
    btnStop.disabled = false;
  }
  function stopCamera() {
    KSL.Camera.stop();
    camhint.classList.remove('hidden');
    camhint.innerHTML = 'Camera stopped. Click "Start camera" to resume.';
    btnStart.disabled = false;
    btnStop.disabled = true;
    handBadge.textContent = 'Hands: 0';
  }
  btnStart.addEventListener('click', startCamera);
  btnStop.addEventListener('click', stopCamera);

  /* ---------------- prediction UI ---------------- */
  function updatePredictionUI(result) {
    if (!result) {
      predSymbol.textContent = '—';
      predSymbol.classList.add('empty');
      predLabelText.textContent = recognitionOn ? 'No confident match' : 'Recognition is off';
      predConf.style.width = '0%';
      return;
    }
    predSymbol.textContent = result.label;
    predSymbol.classList.remove('empty');
    const glossItem = KSL.CATEGORIES[recogCategory].items.find(i => i.label === result.label);
    predLabelText.textContent = (glossItem ? glossItem.gloss + ' · ' : '') + Math.round(result.confidence * 100) + '% confidence';
    predConf.style.width = Math.round(result.confidence * 100) + '%';
  }

  function pushHistory(label) {
    history.push(label);
    if (history.length > 20) history.shift();
    if (label === null) handClearedSinceCommit = true;
  }
  function maybeAutoCommit() {
    if (history.length < STABLE_FRAMES) return;
    const recent = history.slice(-STABLE_FRAMES);
    const first = recent[0];
    if (first === null) return;
    if (!recent.every(l => l === first)) return;
    if (first === lastCommittedLabel && !handClearedSinceCommit) return;
    commitLabel(first);
    lastCommittedLabel = first;
    handClearedSinceCommit = false;
  }

  /* ---------------- transcript / buffer ---------------- */
  function refreshTranscriptUI() { bufferText.textContent = buffer; }
  function commitLabel(label) {
    const meta = KSL.CATEGORIES[recogCategory].items.find(i => i.label === label);
    const buffered = meta ? meta.buffered : true;
    if (buffered) {
      buffer += label;
    } else {
      transcriptEl.value += (transcriptEl.value && !transcriptEl.value.endsWith(' ') ? ' ' : '') + label + ' ';
    }
    refreshTranscriptUI();
    flash('Added: ' + label);
  }
  function finishWord() {
    if (!buffer) return;
    transcriptEl.value += (transcriptEl.value && !transcriptEl.value.endsWith(' ') ? ' ' : '') + buffer + ' ';
    buffer = '';
    refreshTranscriptUI();
  }
  function backspace() {
    if (buffer) { buffer = buffer.slice(0, -1); refreshTranscriptUI(); }
    else { transcriptEl.value = transcriptEl.value.slice(0, -1); }
  }
  function clearAll() { buffer = ''; transcriptEl.value = ''; refreshTranscriptUI(); }

  document.getElementById('btnAddSymbol').addEventListener('click', () => { if (currentPrediction) commitLabel(currentPrediction.label); });
  document.getElementById('btnSpace').addEventListener('click', () => { transcriptEl.value += ' '; });
  document.getElementById('btnFinishWord').addEventListener('click', finishWord);
  document.getElementById('btnBackspace').addEventListener('click', backspace);
  document.getElementById('btnClear').addEventListener('click', clearAll);
  document.getElementById('btnCopy').addEventListener('click', () => {
    navigator.clipboard.writeText(transcriptEl.value).then(() => flash('Copied to clipboard'));
  });

  document.addEventListener('keydown', (e) => {
    if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')) return;
    if (e.code === 'Space') { e.preventDefault(); if (currentPrediction) commitLabel(currentPrediction.label); }
    else if (e.code === 'Enter') { e.preventDefault(); finishWord(); }
    else if (e.code === 'Backspace') { e.preventDefault(); backspace(); }
  });

  /* ---------------- tabs ---------------- */
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
  });

  /* ---------------- category selectors ---------------- */
  function buildCatButtons(container, getCat, onSelect) {
    container.innerHTML = '';
    KSL.CAT_ORDER.forEach(key => {
      const b = document.createElement('button');
      b.className = 'cat-btn' + (getCat() === key ? ' active' : '');
      b.textContent = KSL.CATEGORIES[key].name;
      b.addEventListener('click', () => onSelect(key));
      container.appendChild(b);
    });
  }
  function selectTrainCategory(key) {
    trainCategory = key;
    trainSelectedLabel = null;
    btnRecord.disabled = true;
    trainCur.textContent = '—';
    trainGloss.textContent = 'Pick a label above to train it.';
    addWordPanel.hidden = key !== 'words';
    buildCatButtons(catRowTrain, () => trainCategory, selectTrainCategory);
    renderLabelGrid();
    refreshDataTab();
  }
  function selectRecogCategory(key) {
    recogCategory = key;
    modeLabel.textContent = KSL.CATEGORIES[key].name;
    history = [];
    lastCommittedLabel = null;
    buildCatButtons(catRowRecog, () => recogCategory, selectRecogCategory);
  }
  buildCatButtons(catRowTrain, () => trainCategory, selectTrainCategory);
  buildCatButtons(catRowRecog, () => recogCategory, selectRecogCategory);

  /* ---------------- label grid (train tab) ---------------- */
  function renderLabelGrid() {
    labelGrid.innerHTML = '';
    KSL.CATEGORIES[trainCategory].items.forEach(item => {
      const b = document.createElement('button');
      b.className = 'label-btn' + (trainSelectedLabel === item.label ? ' selected' : '');
      const c = KSL.Dataset.countFor(item.label);
      b.innerHTML = item.label + '<small>' + item.gloss + '</small>' +
        '<span class="count-pip' + (c > 0 ? ' has' : '') + '">' + c + '</span>' +
        (item.custom ? '<span class="del-pip" title="Remove this word">×</span>' : '');
      b.addEventListener('click', (evt) => {
        if (evt.target.classList.contains('del-pip')) {
          evt.stopPropagation();
          if (confirm('Remove "' + item.label + '" and its recorded samples?')) {
            KSL.removeCustomWord(item.label);
            KSL.Dataset.removeLabel(item.label);
            if (trainSelectedLabel === item.label) {
              trainSelectedLabel = null;
              btnRecord.disabled = true;
              trainCur.textContent = '—';
              trainGloss.textContent = 'Pick a label above to train it.';
            }
            renderLabelGrid();
            refreshDataTab();
          }
          return;
        }
        trainSelectedLabel = item.label;
        trainCur.textContent = item.label;
        trainGloss.textContent = item.gloss + ' — samples recorded: ' + KSL.Dataset.countFor(item.label);
        btnRecord.disabled = false;
        renderLabelGrid();
      });
      labelGrid.appendChild(b);
    });
  }
  renderLabelGrid();

  /* ---------------- add-word flow ---------------- */
  function submitNewWord() {
    const res = KSL.addWord(newWordKk.value, newWordGloss.value);
    if (res.ok) {
      newWordKk.value = '';
      newWordGloss.value = '';
      renderLabelGrid();
      refreshDataTab();
      flash('Added new word');
    } else {
      flash(res.error);
    }
  }
  document.getElementById('btnAddWord').addEventListener('click', submitNewWord);
  [newWordKk, newWordGloss].forEach(inp => {
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); submitNewWord(); } });
  });

  /* ---------------- recording bursts ---------------- */
  btnRecord.addEventListener('click', () => {
    if (!trainSelectedLabel) return;
    if (!KSL.Camera.isRunning()) { flash('Start the camera first'); return; }
    const label = trainSelectedLabel;
    const durationMs = 1500, intervalMs = 90;
    let elapsed = 0;
    const captured = [];
    const progress = document.getElementById('recProgress');
    const progressBar = document.getElementById('recProgressBar');
    progress.classList.add('active');
    btnRecord.disabled = true;
    btnRecord.textContent = '● Recording…';
    const timer = setInterval(() => {
      elapsed += intervalMs;
      progressBar.style.width = Math.min(100, (elapsed / durationMs) * 100) + '%';
      const { vector, handCount } = KSL.Hand.buildVector(latestResults);
      if (vector && handCount > 0) {
        captured.push({ label, category: trainCategory, hands: handCount, vector, ts: Date.now() });
      }
      if (elapsed >= durationMs) {
        clearInterval(timer);
        KSL.Dataset.addSamples(captured);
        progress.classList.remove('active');
        progressBar.style.width = '0%';
        btnRecord.disabled = false;
        btnRecord.textContent = '● Record (1.5s)';
        const meta = KSL.CATEGORIES[trainCategory].items.find(i => i.label === label);
        trainGloss.textContent = (meta ? meta.gloss : '') + ' — samples recorded: ' + KSL.Dataset.countFor(label);
        renderLabelGrid();
        refreshDataTab();
        flash(captured.length ? ('Recorded ' + captured.length + ' samples for ' + label) : 'No hand detected during recording — try again');
      }
    }, intervalMs);
  });

  /* ---------------- recognize tab ---------------- */
  document.getElementById('chkRecognize').addEventListener('change', (e) => {
    recognitionOn = e.target.checked;
    recogLabel.textContent = recognitionOn ? 'on' : 'off';
    history = []; lastCommittedLabel = null;
    if (!recognitionOn) updatePredictionUI(null);
  });
  document.getElementById('chkAuto').addEventListener('change', (e) => { autoCommit = e.target.checked; });

  /* ---------------- data tab ---------------- */
  function refreshDataTab() {
    const tbody = document.querySelector('#dataTable tbody');
    tbody.innerHTML = '';
    let totalSamples = 0, labelsWithData = 0, coveredCore = 0;
    const coreLabels = [...KSL.ALPHABET, ...KSL.DIGITS];
    KSL.CAT_ORDER.forEach(catKey => {
      KSL.CATEGORIES[catKey].items.forEach(item => {
        const c = KSL.Dataset.countFor(item.label);
        totalSamples += c;
        if (c > 0) {
          labelsWithData++;
          if (coreLabels.includes(item.label)) coveredCore++;
        }
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + item.label + ' <span style="color:var(--muted);font-family:var(--sans);font-size:0.72rem;">' + item.gloss + '</span></td>' +
          '<td>' + KSL.CATEGORIES[catKey].name + '</td><td>' + c + '</td>';
        tbody.appendChild(tr);
      });
    });
    document.getElementById('statTotal').textContent = totalSamples;
    document.getElementById('statLabels').textContent = labelsWithData;
    document.getElementById('statCoverage').textContent = Math.round((coveredCore / coreLabels.length) * 100) + '%';
  }

  document.getElementById('btnExport').addEventListener('click', () => {
    const blob = new Blob([KSL.Dataset.exportJSON()], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ksl_dataset_' + new Date().toISOString().slice(0, 10) + '.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });
  document.getElementById('btnImport').addEventListener('click', () => document.getElementById('fileImport').click());
  document.getElementById('fileImport').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const merge = confirm('Merge with your current data? Cancel = replace entirely.');
        KSL.Dataset.importJSON(reader.result, merge);
        renderLabelGrid();
        refreshDataTab();
        flash('Dataset imported');
      } catch (err) {
        flash(err.message || 'Could not read that file');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  });
  document.getElementById('btnReset').addEventListener('click', () => {
    if (confirm('Delete ALL recorded samples? Custom words stay, but every recording is gone (export first if unsure).')) {
      KSL.Dataset.resetAll();
      renderLabelGrid();
      refreshDataTab();
      flash('All recorded samples cleared');
    }
  });

  /* ---------------- boot ---------------- */
  refreshDataTab();
  refreshTranscriptUI();

})(window.KSL);
