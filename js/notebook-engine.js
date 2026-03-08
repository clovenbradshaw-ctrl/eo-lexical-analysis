/**
 * Notebook engine — Pyodide integration, cell execution, output routing.
 * Loads Python-in-WebAssembly and runs analysis code in the browser.
 */

let pyodide = null;
let pyodideReady = false;
let pyodideLoading = false;

// ── Pyodide initialization ──────────────────────────────────

async function initPyodide(onStatus) {
  if (pyodideReady) return pyodide;
  if (pyodideLoading) {
    // Wait for existing load
    while (!pyodideReady) {
      await new Promise(r => setTimeout(r, 200));
    }
    return pyodide;
  }

  pyodideLoading = true;

  try {
    if (onStatus) onStatus('Loading Pyodide runtime...');
    pyodide = await loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.4/full/',
    });

    if (onStatus) onStatus('Installing numpy...');
    await pyodide.loadPackage('numpy');

    if (onStatus) onStatus('Installing scipy...');
    await pyodide.loadPackage('scipy');

    if (onStatus) onStatus('Installing scikit-learn...');
    await pyodide.loadPackage('scikit-learn');

    // Load our Python modules into the virtual filesystem
    if (onStatus) onStatus('Loading analysis modules...');
    await loadPythonModules();

    pyodideReady = true;
    if (onStatus) onStatus('Ready');
    return pyodide;
  } catch (e) {
    pyodideLoading = false;
    throw e;
  }
}


async function loadPythonModules() {
  // Fetch our Python files and write them to Pyodide's virtual FS
  const modules = [
    { path: '/home/pyodide/analysis.py', url: 'py/analysis.py' },
    { path: '/home/pyodide/conllu_parser.py', url: 'py/conllu_parser.py' },
    { path: '/home/pyodide/embeddings.py', url: 'py/embeddings.py' },
    { path: '/home/pyodide/operator_definitions.py', url: 'py/operator_definitions.py' },
  ];

  for (const mod of modules) {
    try {
      const resp = await fetch(mod.url);
      if (resp.ok) {
        const text = await resp.text();
        pyodide.FS.writeFile(mod.path, text);
      }
    } catch (e) {
      console.warn(`Failed to load ${mod.url}:`, e);
    }
  }

  // Add module directory to Python path
  await pyodide.runPythonAsync(`
import sys
if '/home/pyodide' not in sys.path:
    sys.path.insert(0, '/home/pyodide')
  `);
}


// ── Cell execution ──────────────────────────────────────────

async function runPython(code, globals) {
  if (!pyodideReady) throw new Error('Pyodide not initialized. Run setup first.');

  // Set globals if provided
  if (globals) {
    for (const [key, value] of Object.entries(globals)) {
      pyodide.globals.set(key, pyodide.toPy(value));
    }
  }

  // Capture stdout
  const output = [];
  await pyodide.runPythonAsync(`
import sys, io
_capture = io.StringIO()
sys.stdout = _capture
  `);

  try {
    const result = await pyodide.runPythonAsync(code);
    const stdout = pyodide.runPython('_capture.getvalue()');
    await pyodide.runPythonAsync('sys.stdout = sys.__stdout__');

    return {
      result: result ? result.toJs({ dict_converter: Object.fromEntries }) : null,
      stdout: stdout || '',
      error: null,
    };
  } catch (e) {
    await pyodide.runPythonAsync('sys.stdout = sys.__stdout__').catch(() => {});
    const stdout = pyodide.runPython('_capture.getvalue()').catch(() => '');
    return {
      result: null,
      stdout: typeof stdout === 'string' ? stdout : '',
      error: e.message,
    };
  }
}


// ── Pre-built analysis runners ──────────────────────────────

async function runCompleteness(verbEmbeddings, opEmbeddings, verbNames) {
  const py = await initPyodide();

  // Pass data to Python
  py.globals.set('verb_embs_flat', py.toPy(verbEmbeddings.flat()));
  py.globals.set('op_embs_flat', py.toPy(opEmbeddings.flat()));
  py.globals.set('verb_names', py.toPy(verbNames));
  py.globals.set('n_verbs', verbEmbeddings.length);
  py.globals.set('n_ops', opEmbeddings.length);
  py.globals.set('dim', verbEmbeddings[0].length);

  const result = await py.runPythonAsync(`
import numpy as np
from analysis import test_completeness

verb_embs = np.array(list(verb_embs_flat), dtype=np.float32).reshape(int(n_verbs), int(dim))
op_embs = np.array(list(op_embs_flat), dtype=np.float32).reshape(int(n_ops), int(dim))
names = list(verb_names)

result = test_completeness(verb_embs, op_embs, names)
# Remove numpy arrays from result (not JSON serializable)
del result['sim_matrix']
del result['nearest_idx']
del result['nearest_sim']
result
  `);

  return result.toJs({ dict_converter: Object.fromEntries });
}


async function runClustering(verbEmbeddings, opEmbeddings, verbNames) {
  const py = await initPyodide();

  py.globals.set('verb_embs_flat', py.toPy(verbEmbeddings.flat()));
  py.globals.set('op_embs_flat', py.toPy(opEmbeddings.flat()));
  py.globals.set('verb_names', py.toPy(verbNames));
  py.globals.set('n_verbs', verbEmbeddings.length);
  py.globals.set('n_ops', opEmbeddings.length);
  py.globals.set('dim', verbEmbeddings[0].length);

  const result = await py.runPythonAsync(`
import numpy as np
from analysis import test_clustering

verb_embs = np.array(list(verb_embs_flat), dtype=np.float32).reshape(int(n_verbs), int(dim))
op_embs = np.array(list(op_embs_flat), dtype=np.float32).reshape(int(n_ops), int(dim))
names = list(verb_names)

result = test_clustering(verb_embs, op_embs, names)
result
  `);

  return result.toJs({ dict_converter: Object.fromEntries });
}


async function runOrthogonality(opEmbeddings) {
  const py = await initPyodide();

  py.globals.set('op_embs_flat', py.toPy(opEmbeddings.flat()));
  py.globals.set('n_ops', opEmbeddings.length);
  py.globals.set('dim', opEmbeddings[0].length);

  const result = await py.runPythonAsync(`
import numpy as np
from analysis import test_orthogonality

op_embs = np.array(list(op_embs_flat), dtype=np.float32).reshape(int(n_ops), int(dim))
result = test_orthogonality(op_embs)
result
  `);

  return result.toJs({ dict_converter: Object.fromEntries });
}


async function runPCA(verbEmbeddings, opEmbeddings, nearestIdx) {
  const py = await initPyodide();

  py.globals.set('verb_embs_flat', py.toPy(verbEmbeddings.flat()));
  py.globals.set('op_embs_flat', py.toPy(opEmbeddings.flat()));
  py.globals.set('nearest_idx_list', py.toPy(nearestIdx));
  py.globals.set('n_verbs', verbEmbeddings.length);
  py.globals.set('n_ops', opEmbeddings.length);
  py.globals.set('dim', verbEmbeddings[0].length);

  const result = await py.runPythonAsync(`
import numpy as np
from analysis import compute_pca

verb_embs = np.array(list(verb_embs_flat), dtype=np.float32).reshape(int(n_verbs), int(dim))
op_embs = np.array(list(op_embs_flat), dtype=np.float32).reshape(int(n_ops), int(dim))
nearest_idx = np.array(list(nearest_idx_list), dtype=int)

result = compute_pca(verb_embs, op_embs, nearest_idx)
result
  `);

  return result.toJs({ dict_converter: Object.fromEntries });
}


async function runConlluParse(conlluText) {
  const py = await initPyodide();

  py.globals.set('conllu_text', conlluText);

  const result = await py.runPythonAsync(`
from conllu_parser import extract_verbs, extract_clauses, summarize_clauses

verbs = extract_verbs(conllu_text)
clauses = extract_clauses(conllu_text)
summary = summarize_clauses(clauses)

{
  'verbs': verbs,
  'clause_summary': summary,
  'n_verbs': len(verbs),
  'n_clauses': len(clauses),
}
  `);

  return result.toJs({ dict_converter: Object.fromEntries });
}


async function runFalsification(embeddings, labels) {
  const py = await initPyodide();

  py.globals.set('embs_flat', py.toPy(embeddings.flat()));
  py.globals.set('labels_list', py.toPy(labels));
  py.globals.set('n_items', embeddings.length);
  py.globals.set('dim', embeddings[0].length);

  const result = await py.runPythonAsync(`
import numpy as np
from analysis import test_falsification

embs = np.array(list(embs_flat), dtype=np.float32).reshape(int(n_items), int(dim))
labs = np.array(list(labels_list), dtype=int)

result = test_falsification(embs, labs, n_random_taxonomies=10)
result
  `);

  return result.toJs({ dict_converter: Object.fromEntries });
}


// ── Cell UI management ──────────────────────────────────────

function setCellStatus(cellId, status) {
  const cell = document.getElementById(cellId);
  if (!cell) return;

  const indicator = cell.querySelector('.cell-status');
  if (indicator) {
    indicator.className = `cell-status status-${status}`;
    const labels = { idle: 'Idle', running: 'Running...', complete: 'Complete', error: 'Error' };
    indicator.textContent = labels[status] || status;
  }
}


function appendCellOutput(cellId, html) {
  const cell = document.getElementById(cellId);
  if (!cell) return;
  const output = cell.querySelector('.cell-output');
  if (output) {
    output.innerHTML += html;
    output.scrollTop = output.scrollHeight;
  }
}


function setCellOutput(cellId, html) {
  const cell = document.getElementById(cellId);
  if (!cell) return;
  const output = cell.querySelector('.cell-output');
  if (output) {
    output.innerHTML = html;
  }
}


function logToCell(cellId, message) {
  appendCellOutput(cellId, `<div class="log-line">${escHtml(message)}</div>`);
}


function escHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}


const NotebookEngine = {
  initPyodide,
  runPython,
  runCompleteness,
  runClustering,
  runOrthogonality,
  runPCA,
  runConlluParse,
  runFalsification,
  setCellStatus,
  setCellOutput,
  appendCellOutput,
  logToCell,
  isPyodideReady: () => pyodideReady,
};

export default NotebookEngine;
