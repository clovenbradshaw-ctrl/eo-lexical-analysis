/**
 * Export module — download results as JSON, CSV, or ZIP.
 */

import Storage from './storage.js';

function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  downloadBlob(blob, filename);
}

function downloadCSV(rows, headers, filename) {
  const headerLine = headers.join(',');
  const lines = rows.map(row =>
    headers.map(h => {
      const val = row[h] ?? '';
      const str = String(val);
      return str.includes(',') || str.includes('"') || str.includes('\n')
        ? `"${str.replace(/"/g, '""')}"`
        : str;
    }).join(',')
  );
  const csv = [headerLine, ...lines].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  downloadBlob(blob, filename);
}

function downloadText(text, filename) {
  const blob = new Blob([text], { type: 'text/plain' });
  downloadBlob(blob, filename);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function downloadAllResults() {
  const data = await Storage.exportAll();
  downloadJSON(data, `eo-notebook-export-${new Date().toISOString().slice(0, 10)}.json`);
}

function downloadPlotlyAsPNG(plotDiv, filename) {
  if (typeof Plotly !== 'undefined' && plotDiv) {
    Plotly.downloadImage(plotDiv, {
      format: 'png',
      width: 1200,
      height: 800,
      filename: filename.replace('.png', ''),
    });
  }
}

// ── Classification export helpers ───────────────────────────

function exportClassifications(classifications, filename) {
  downloadJSON(classifications, filename || 'classifications.json');
}

function exportClassificationsCSV(classifications, filename) {
  const headers = ['verb', 'operator', 'confidence', 'gloss', 'alternative'];
  downloadCSV(classifications, headers, filename || 'classifications.csv');
}

function exportCorpus(corpus, filename) {
  downloadJSON(corpus, filename || 'corpus.json');
}

function exportCorpusCSV(corpus, filename) {
  if (Array.isArray(corpus)) {
    const headers = ['verb', 'definition', 'source'];
    downloadCSV(corpus, headers, filename || 'corpus.csv');
  } else {
    // Dict format
    const rows = Object.entries(corpus).map(([verb, info]) => ({
      verb,
      definition: typeof info === 'string' ? info : (info.definition || ''),
      source: typeof info === 'object' ? (info.source || '') : '',
    }));
    const headers = ['verb', 'definition', 'source'];
    downloadCSV(rows, headers, filename || 'corpus.csv');
  }
}

// ── Import ──────────────────────────────────────────────────

async function importFromFile(file) {
  const text = await file.text();
  const data = JSON.parse(text);
  await Storage.importAll(data);
  return data;
}

const Export = {
  downloadJSON,
  downloadCSV,
  downloadText,
  downloadBlob,
  downloadAllResults,
  downloadPlotlyAsPNG,
  exportClassifications,
  exportClassificationsCSV,
  exportCorpus,
  exportCorpusCSV,
  importFromFile,
};

export default Export;
