/**
 * IndexedDB persistence layer for the EO Notebook Experiment App.
 * Stores corpus data, embeddings, classifications, and results across sessions.
 */

const DB_NAME = 'eo-notebook';
const DB_VERSION = 1;

const STORES = {
  corpus: 'corpus',
  embeddings: 'embeddings',
  classifications: 'classifications',
  results: 'results',
  crossling: 'crossling',
};

let _db = null;

function openDB() {
  if (_db) return Promise.resolve(_db);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      for (const name of Object.values(STORES)) {
        if (!db.objectStoreNames.contains(name)) {
          db.createObjectStore(name);
        }
      }
    };
    req.onsuccess = (e) => {
      _db = e.target.result;
      resolve(_db);
    };
    req.onerror = (e) => reject(e.target.error);
  });
}

async function dbGet(store, key) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const req = tx.objectStore(store).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function dbPut(store, key, value) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    const req = tx.objectStore(store).put(value, key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

async function dbDelete(store, key) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    const req = tx.objectStore(store).delete(key);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

async function dbGetAll(store) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const req = tx.objectStore(store).getAll();
    const reqKeys = tx.objectStore(store).getAllKeys();
    const results = {};
    req.onsuccess = () => {
      reqKeys.onsuccess = () => {
        const keys = reqKeys.result;
        const values = req.result;
        for (let i = 0; i < keys.length; i++) {
          results[keys[i]] = values[i];
        }
        resolve(results);
      };
    };
    req.onerror = () => reject(req.error);
  });
}

async function dbClear(store) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    const req = tx.objectStore(store).clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// ── High-level API ──────────────────────────────────────────

const Storage = {
  // Corpus
  async saveCorpus(data) { return dbPut(STORES.corpus, 'main', data); },
  async loadCorpus() { return dbGet(STORES.corpus, 'main'); },

  // Embeddings — chunked storage to avoid IndexedDB memory limits.
  // Large embedding arrays are split into chunks of CHUNK_SIZE vectors each,
  // stored under keys like "main_chunk_0", "main_chunk_1", etc.
  // Metadata (model, dim, count, verb_names) is stored under the base key.
  async saveEmbeddings(key, data) {
    const CHUNK_SIZE = 1000;
    const vectors = data.embeddings || [];

    // Clear any previous chunks for this key
    const db = await openDB();
    const existingKeys = await new Promise((resolve, reject) => {
      const tx = db.transaction(STORES.embeddings, 'readonly');
      const req = tx.objectStore(STORES.embeddings).getAllKeys();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    for (const k of existingKeys) {
      if (k === key || (typeof k === 'string' && k.startsWith(key + '_chunk_'))) {
        await dbDelete(STORES.embeddings, k);
      }
    }

    // Store metadata (everything except the raw embeddings array)
    // Avoid spread to prevent temporarily duplicating the large embeddings array in memory
    const meta = {};
    for (const k of Object.keys(data)) {
      if (k !== 'embeddings') meta[k] = data[k];
    }
    meta.embeddings = null;
    meta._chunked = true;
    meta._chunkCount = Math.ceil(vectors.length / CHUNK_SIZE);
    await dbPut(STORES.embeddings, key, meta);

    // Store embedding vectors in chunks
    for (let i = 0; i < vectors.length; i += CHUNK_SIZE) {
      const chunkIndex = Math.floor(i / CHUNK_SIZE);
      const chunk = vectors.slice(i, i + CHUNK_SIZE);
      await dbPut(STORES.embeddings, `${key}_chunk_${chunkIndex}`, chunk);
    }
  },

  async loadEmbeddings(key) {
    const meta = await dbGet(STORES.embeddings, key);
    if (!meta) return undefined;

    // Handle legacy non-chunked data
    if (!meta._chunked) return meta;

    // Reassemble chunks
    const embeddings = [];
    for (let i = 0; i < meta._chunkCount; i++) {
      const chunk = await dbGet(STORES.embeddings, `${key}_chunk_${i}`);
      if (chunk) embeddings.push(...chunk);
    }

    const { _chunked, _chunkCount, ...rest } = meta;
    return { ...rest, embeddings };
  },

  async listEmbeddings() { return dbGetAll(STORES.embeddings); },

  // Classifications
  async saveClassifications(data) { return dbPut(STORES.classifications, 'main', data); },
  async loadClassifications() { return dbGet(STORES.classifications, 'main'); },

  // Results
  async saveResult(key, data) { return dbPut(STORES.results, key, data); },
  async loadResult(key) { return dbGet(STORES.results, key); },
  async listResults() { return dbGetAll(STORES.results); },

  // Cross-linguistic
  async saveCrossling(lang, data) { return dbPut(STORES.crossling, lang, data); },
  async loadCrossling(lang) { return dbGet(STORES.crossling, lang); },
  async listCrossling() { return dbGetAll(STORES.crossling); },

  // Bulk export
  async exportAll() {
    const data = {};
    for (const [name, store] of Object.entries(STORES)) {
      data[name] = await dbGetAll(store);
    }
    return data;
  },

  // Bulk import
  async importAll(data) {
    for (const [name, store] of Object.entries(STORES)) {
      if (data[name]) {
        await dbClear(store);
        for (const [key, value] of Object.entries(data[name])) {
          await dbPut(store, key, value);
        }
      }
    }
  },

  // Clear everything
  async clearAll() {
    for (const store of Object.values(STORES)) {
      await dbClear(store);
    }
  },

  // API Keys (localStorage, not IndexedDB)
  getApiKey(provider) {
    return localStorage.getItem(`eo-apikey-${provider}`) || '';
  },
  setApiKey(provider, key) {
    if (key) {
      localStorage.setItem(`eo-apikey-${provider}`, key);
    } else {
      localStorage.removeItem(`eo-apikey-${provider}`);
    }
  },

  // Settings
  getSetting(key, defaultValue) {
    const v = localStorage.getItem(`eo-setting-${key}`);
    return v !== null ? JSON.parse(v) : defaultValue;
  },
  setSetting(key, value) {
    localStorage.setItem(`eo-setting-${key}`, JSON.stringify(value));
  },
};

export default Storage;
