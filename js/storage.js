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

  // Embeddings
  async saveEmbeddings(key, data) { return dbPut(STORES.embeddings, key, data); },
  async loadEmbeddings(key) { return dbGet(STORES.embeddings, key); },
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
