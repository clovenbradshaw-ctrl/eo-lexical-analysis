/**
 * Corpus manager — downloads and manages linguistic corpora.
 * Handles UD treebanks (fetch from GitHub), pre-computed data, and custom uploads.
 */

import Storage from './storage.js';

// ── UD Treebank language definitions ────────────────────────

const UD_LANGUAGES = [
  // Ancient
  { name: 'Ancient_Greek', treebank: 'UD_Ancient_Greek-Perseus', code: 'grc', tbName: 'perseus', family: 'IE-Hellenic', era: 'ancient', region: 'Mediterranean', morph: 'fusional' },
  { name: 'Latin', treebank: 'UD_Latin-Perseus', code: 'la', tbName: 'perseus', family: 'IE-Italic', era: 'ancient', region: 'Mediterranean', morph: 'fusional' },
  { name: 'Classical_Chinese', treebank: 'UD_Classical_Chinese-Kyoto', code: 'lzh', tbName: 'kyoto', family: 'Sino-Tibetan', era: 'ancient', region: 'East Asia', morph: 'isolating' },
  { name: 'Sanskrit', treebank: 'UD_Sanskrit-Vedic', code: 'sa', tbName: 'vedic', family: 'IE-Indo-Aryan', era: 'ancient', region: 'South Asia', morph: 'fusional' },
  { name: 'Old_Church_Slavonic', treebank: 'UD_Old_Church_Slavonic-PROIEL', code: 'cu', tbName: 'proiel', family: 'IE-Slavic', era: 'ancient', region: 'Eastern Europe', morph: 'fusional' },
  { name: 'Gothic', treebank: 'UD_Gothic-PROIEL', code: 'got', tbName: 'proiel', family: 'IE-Germanic', era: 'ancient', region: 'Northern Europe', morph: 'fusional' },
  { name: 'Old_French', treebank: 'UD_Old_French-SRCMF', code: 'fro', tbName: 'srcmf', family: 'IE-Romance', era: 'medieval', region: 'Western Europe', morph: 'fusional' },
  { name: 'Old_East_Slavic', treebank: 'UD_Old_East_Slavic-TOROT', code: 'orv', tbName: 'torot', family: 'IE-Slavic', era: 'medieval', region: 'Eastern Europe', morph: 'fusional' },
  { name: 'Coptic', treebank: 'UD_Coptic-Scriptorium', code: 'cop', tbName: 'scriptorium', family: 'Afro-Asiatic', era: 'ancient', region: 'North Africa', morph: 'agglutinative' },

  // Modern Global North
  { name: 'English', treebank: 'UD_English-EWT', code: 'en', tbName: 'ewt', family: 'IE-Germanic', era: 'modern', region: 'Global North', morph: 'fusional' },
  { name: 'German', treebank: 'UD_German-GSD', code: 'de', tbName: 'gsd', family: 'IE-Germanic', era: 'modern', region: 'Global North', morph: 'fusional' },
  { name: 'Russian', treebank: 'UD_Russian-SynTagRus', code: 'ru', tbName: 'syntagrus', family: 'IE-Slavic', era: 'modern', region: 'Global North', morph: 'fusional' },
  { name: 'French', treebank: 'UD_French-GSD', code: 'fr', tbName: 'gsd', family: 'IE-Romance', era: 'modern', region: 'Global North', morph: 'fusional' },
  { name: 'Finnish', treebank: 'UD_Finnish-TDT', code: 'fi', tbName: 'tdt', family: 'Uralic', era: 'modern', region: 'Global North', morph: 'agglutinative' },

  // Modern Global South / Non-Western
  { name: 'Japanese', treebank: 'UD_Japanese-GSD', code: 'ja', tbName: 'gsd', family: 'Japonic', era: 'modern', region: 'East Asia', morph: 'agglutinative' },
  { name: 'Mandarin', treebank: 'UD_Chinese-GSDSimp', code: 'zh', tbName: 'gsdsimp', family: 'Sino-Tibetan', era: 'modern', region: 'East Asia', morph: 'isolating' },
  { name: 'Korean', treebank: 'UD_Korean-Kaist', code: 'ko', tbName: 'kaist', family: 'Koreanic', era: 'modern', region: 'East Asia', morph: 'agglutinative' },
  { name: 'Arabic', treebank: 'UD_Arabic-PADT', code: 'ar', tbName: 'padt', family: 'Afro-Asiatic', era: 'modern', region: 'Middle East', morph: 'fusional' },
  { name: 'Hindi', treebank: 'UD_Hindi-HDTB', code: 'hi', tbName: 'hdtb', family: 'IE-Indo-Aryan', era: 'modern', region: 'South Asia', morph: 'fusional' },
  { name: 'Turkish', treebank: 'UD_Turkish-BOUN', code: 'tr', tbName: 'boun', family: 'Turkic', era: 'modern', region: 'West Asia', morph: 'agglutinative' },
  { name: 'Indonesian', treebank: 'UD_Indonesian-GSD', code: 'id', tbName: 'gsd', family: 'Austronesian', era: 'modern', region: 'Southeast Asia', morph: 'agglutinative' },
  { name: 'Vietnamese', treebank: 'UD_Vietnamese-VTB', code: 'vi', tbName: 'vtb', family: 'Austroasiatic', era: 'modern', region: 'Southeast Asia', morph: 'isolating' },
  { name: 'Tamil', treebank: 'UD_Tamil-TTB', code: 'ta', tbName: 'ttb', family: 'Dravidian', era: 'modern', region: 'South Asia', morph: 'agglutinative' },
  { name: 'Yoruba', treebank: 'UD_Yoruba-YTB', code: 'yo', tbName: 'ytb', family: 'Niger-Congo', era: 'modern', region: 'West Africa', morph: 'isolating' },
  { name: 'Wolof', treebank: 'UD_Wolof-WTB', code: 'wo', tbName: 'wtb', family: 'Niger-Congo', era: 'modern', region: 'West Africa', morph: 'agglutinative' },
  { name: 'Naija', treebank: 'UD_Naija-NSC', code: 'pcm', tbName: 'nsc', family: 'Creole-English', era: 'modern', region: 'West Africa', morph: 'isolating' },
  { name: 'Tagalog', treebank: 'UD_Tagalog-TRG', code: 'tl', tbName: 'trg', family: 'Austronesian', era: 'modern', region: 'Southeast Asia', morph: 'agglutinative' },
  { name: 'Persian', treebank: 'UD_Persian-PerDT', code: 'fa', tbName: 'perdt', family: 'IE-Iranian', era: 'modern', region: 'West Asia', morph: 'fusional' },
  { name: 'Uyghur', treebank: 'UD_Uyghur-UDT', code: 'ug', tbName: 'udt', family: 'Turkic', era: 'modern', region: 'Central Asia', morph: 'agglutinative' },
  { name: 'Basque', treebank: 'UD_Basque-BDT', code: 'eu', tbName: 'bdt', family: 'Isolate', era: 'modern', region: 'Western Europe', morph: 'agglutinative' },
];

const UD_BASE = 'https://raw.githubusercontent.com/UniversalDependencies';


// ── Download UD treebank ────────────────────────────────────

async function downloadTreebank(langName, onProgress) {
  const lang = UD_LANGUAGES.find(l => l.name === langName);
  if (!lang) throw new Error(`Unknown language: ${langName}`);

  const splits = ['train', 'dev', 'test'];
  const fname = `${lang.code}_${lang.tbName}`;
  let combined = '';
  let downloaded = 0;

  for (const split of splits) {
    const url = `${UD_BASE}/${lang.treebank}/master/${fname}-ud-${split}.conllu`;
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const text = await resp.text();
        combined += text + '\n';
        downloaded++;
      }
    } catch (e) {
      // Split may not exist for this treebank
    }

    if (onProgress) {
      onProgress({ split, downloaded, total: splits.length });
    }
  }

  if (!combined.trim()) {
    throw new Error(`No CoNLL-U data found for ${langName}. The treebank may have different file naming.`);
  }

  // Cache the raw CoNLL-U text
  await Storage.saveCrossling(`${langName}_conllu`, combined);

  return {
    language: langName,
    family: lang.family,
    era: lang.era,
    region: lang.region,
    morph_type: lang.morph,
    treebank: lang.treebank,
    conllu_size: combined.length,
    splits_downloaded: downloaded,
    raw: combined,
  };
}


// ── Load pre-computed data from repo ────────────────────────

async function loadPrecomputedData(basePath) {
  const files = {
    verbs: 'data/verbs.json',
    operators: 'data/operators.json',
    metrics: 'data/metrics.json',
    boundaries: 'data/boundaries.json',
    crossling: 'data/crossling.json',
    confusion: 'data/confusion.json',
    influence: 'data/influence.json',
  };

  const data = {};
  for (const [key, path] of Object.entries(files)) {
    try {
      const resp = await fetch(basePath ? `${basePath}/${path}` : path);
      if (resp.ok) {
        data[key] = await resp.json();
      }
    } catch (e) {
      // File may not exist
    }
  }

  return data;
}


// ── Load cross-linguistic classified data ───────────────────

async function loadCrosslingClassified(langName, basePath) {
  // Try from IndexedDB cache first
  const cached = await Storage.loadCrossling(langName);
  if (cached && cached.classifications) return cached;

  // Try from GCS / repo
  const urls = [
    basePath ? `${basePath}/data/crossling/${langName}/classified.json` : null,
    `data/crossling/${langName}/classified.json`,
  ].filter(Boolean);

  for (const url of urls) {
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const data = await resp.json();
        await Storage.saveCrossling(langName, data);
        return data;
      }
    } catch (e) {
      // Try next URL
    }
  }

  return null;
}


// ── Custom corpus upload ────────────────────────────────────

function parseUploadedCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const verbIdx = headers.indexOf('verb');
  const defIdx = headers.indexOf('definition');

  if (verbIdx < 0) throw new Error('CSV must have a "verb" column');

  const verbs = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim());
    if (cols[verbIdx]) {
      verbs.push({
        verb: cols[verbIdx],
        definition: defIdx >= 0 ? cols[defIdx] : '',
        source: 'custom',
      });
    }
  }
  return verbs;
}

function parseUploadedJSON(text) {
  const data = JSON.parse(text);
  if (Array.isArray(data)) {
    return data.map(item => ({
      verb: item.verb || item.word || item.lemma || '',
      definition: item.definition || item.def || item.gloss || '',
      source: 'custom',
    })).filter(v => v.verb);
  }
  // Handle dict format {verb: {definition: ...}}
  return Object.entries(data).map(([verb, info]) => ({
    verb,
    definition: typeof info === 'string' ? info : (info.definition || info.def || ''),
    source: 'custom',
  }));
}


const CorpusManager = {
  UD_LANGUAGES,
  downloadTreebank,
  loadPrecomputedData,
  loadCrosslingClassified,
  parseUploadedCSV,
  parseUploadedJSON,

  getLanguageInfo(name) {
    return UD_LANGUAGES.find(l => l.name === name) || null;
  },

  getLanguagesByEra(era) {
    return UD_LANGUAGES.filter(l => l.era === era);
  },

  getLanguagesByFamily(family) {
    return UD_LANGUAGES.filter(l => l.family === family);
  },
};

export default CorpusManager;
