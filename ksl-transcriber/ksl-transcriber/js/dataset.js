/* dataset.js
 * Stores and manages the recorded training samples: {label, category, hands, vector, ts}.
 * Persisted to localStorage; also exportable/importable as JSON so a user's
 * recordings can be backed up, merged, or eventually feed the Phase 2 model.
 */
window.KSL = window.KSL || {};
(function (KSL) {
  "use strict";

  const STORE_KEY = 'kslTranscriberDB_v1';
  let DB = { samples: [] };

  function load() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) DB = JSON.parse(raw);
      if (!DB || !Array.isArray(DB.samples)) DB = { samples: [] };
    } catch (e) {
      console.warn('Could not load saved dataset', e);
      DB = { samples: [] };
    }
  }
  function save() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(DB)); }
    catch (e) { console.warn('Could not save dataset', e); }
  }

  function countFor(label) {
    return DB.samples.filter(s => s.label === label).length;
  }
  function addSamples(list) {
    if (!list || !list.length) return;
    DB.samples.push(...list);
    save();
  }
  function removeLabel(label) {
    DB.samples = DB.samples.filter(s => s.label !== label);
    save();
  }
  function resetAll() {
    DB = { samples: [] };
    save();
  }
  function exportJSON() {
    return JSON.stringify(DB, null, 2);
  }
  function importJSON(text, merge) {
    const incoming = JSON.parse(text);
    if (!incoming || !Array.isArray(incoming.samples)) {
      throw new Error('File does not look like a valid dataset export.');
    }
    DB = merge ? { samples: DB.samples.concat(incoming.samples) } : incoming;
    save();
  }

  load();

  KSL.Dataset = {
    get samples() { return DB.samples; },
    countFor,
    addSamples,
    removeLabel,
    resetAll,
    exportJSON,
    importJSON
  };

})(window.KSL);
