/**
 * API client for OpenAI and Anthropic — makes calls directly from the browser.
 * API keys are provided by the user and stored in localStorage.
 */

import Storage from './storage.js';

const OPERATOR_DEFINITIONS = `You are classifying verb meanings into nine transformation types.
For each verb, determine which single operator best describes the
transformation the verb enacts. Consider what exists BEFORE the verb
acts and what exists AFTER.

THE NINE OPERATORS:

NUL — MARK ABSENCE: Transform a state by making the absence of something the salient fact.
DES — DRAW DISTINCTION: Register something as different from its ground.
INS — SOMETHING APPEARS: Create a new event, entity, or state in the world.
SEG — ONE BECOMES MANY: Transform a unity into parts by introducing a boundary.
CON — CREATE PERSISTENT LINK: Establish a relationship between separate identities that persists.
SYN — MANY BECOME ONE NEW THING: Combine separate elements into a unified whole.
ALT — CHANGE STATE: Same entity, different state. Toggle, switch, convert.
SUP — HOLD INCOMPATIBLE WITHOUT RESOLUTION: Maintain multiple mutually exclusive values simultaneously.
REC — REBUILD AROUND NEW CENTER: Take an existing structure and reorganize it around a different principle.

CLASSIFICATION RULES:
- Choose the SINGLE best operator for the verb's primary meaning
- If uncertain, note your confidence (high/medium/low) and best alternative
- Consider the verb's most common usage, not rare or metaphorical senses`;


// ── OpenAI Embeddings ───────────────────────────────────────

async function generateEmbeddings(texts, onProgress) {
  const apiKey = Storage.getApiKey('openai');
  if (!apiKey) throw new Error('OpenAI API key not set. Configure in sidebar.');

  const model = Storage.getSetting('embeddingModel', 'text-embedding-3-large');
  const batchSize = 2048;
  const allEmbeddings = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);

    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model, input: batch }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(`OpenAI API error ${response.status}: ${err.error?.message || response.statusText}`);
    }

    const data = await response.json();
    const sorted = data.data.sort((a, b) => a.index - b.index);
    for (const item of sorted) {
      allEmbeddings.push(item.embedding);
    }

    if (onProgress) {
      onProgress({
        done: Math.min(i + batchSize, texts.length),
        total: texts.length,
        pct: Math.min(100, Math.round((i + batchSize) / texts.length * 100)),
      });
    }
  }

  return {
    embeddings: allEmbeddings,
    model,
    dim: allEmbeddings[0]?.length || 0,
    count: allEmbeddings.length,
  };
}


// ── LLM Classification ──────────────────────────────────────

async function classifyVerbsOpenAI(verbs, onProgress) {
  const apiKey = Storage.getApiKey('openai');
  if (!apiKey) throw new Error('OpenAI API key not set.');

  const batchSize = Storage.getSetting('batchSize', 50);
  const model = Storage.getSetting('llmModel', 'gpt-4o-mini');
  const classifications = [];

  for (let i = 0; i < verbs.length; i += batchSize) {
    const batch = verbs.slice(i, i + batchSize);
    const verbList = batch.map(v =>
      typeof v === 'string' ? v : `${v.verb} (${v.definition || ''})`
    ).join('\n');

    const prompt = `Classify these verbs into EO operators.
For each verb, provide: verb, operator (NUL/DES/INS/SEG/CON/SYN/ALT/SUP/REC), confidence (high/medium/low), brief gloss, alternative operator if not high confidence.

VERBS:
${verbList}

Respond in JSON array format:
[{"verb": "...", "operator": "...", "confidence": "...", "gloss": "...", "alternative": ""}]`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: OPERATOR_DEFINITIONS },
          { role: 'user', content: prompt },
        ],
        temperature: 0.3,
        max_tokens: 4096,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(`OpenAI error ${response.status}: ${err.error?.message || ''}`);
    }

    const data = await response.json();
    let text = data.choices[0].message.content.trim();

    // Extract JSON from markdown code blocks
    const match = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (match) text = match[1].trim();

    try {
      const batchResults = JSON.parse(text);
      classifications.push(...batchResults);
    } catch (e) {
      console.warn('Failed to parse batch', i, e);
    }

    if (onProgress) {
      onProgress({
        done: Math.min(i + batchSize, verbs.length),
        total: verbs.length,
        pct: Math.min(100, Math.round((i + batchSize) / verbs.length * 100)),
      });
    }

    // Rate limiting
    await new Promise(r => setTimeout(r, 500));
  }

  return classifications;
}


async function classifyVerbsAnthropic(verbs, onProgress) {
  const apiKey = Storage.getApiKey('anthropic');
  if (!apiKey) throw new Error('Anthropic API key not set.');

  const batchSize = Storage.getSetting('batchSize', 50);
  const model = 'claude-sonnet-4-20250514';
  const classifications = [];

  for (let i = 0; i < verbs.length; i += batchSize) {
    const batch = verbs.slice(i, i + batchSize);
    const verbList = batch.map(v =>
      typeof v === 'string' ? v : `${v.verb} (${v.definition || ''})`
    ).join('\n');

    const prompt = `Classify these verbs into EO operators.
For each verb, provide: verb, operator (NUL/DES/INS/SEG/CON/SYN/ALT/SUP/REC), confidence (high/medium/low), brief gloss, alternative operator if not high confidence.

VERBS:
${verbList}

Respond in JSON array format:
[{"verb": "...", "operator": "...", "confidence": "...", "gloss": "...", "alternative": ""}]`;

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model,
        max_tokens: 4096,
        system: OPERATOR_DEFINITIONS,
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(`Anthropic error ${response.status}: ${err.error?.message || ''}`);
    }

    const data = await response.json();
    let text = data.content[0].text.trim();

    const match = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (match) text = match[1].trim();

    try {
      const batchResults = JSON.parse(text);
      classifications.push(...batchResults);
    } catch (e) {
      console.warn('Failed to parse batch', i, e);
    }

    if (onProgress) {
      onProgress({
        done: Math.min(i + batchSize, verbs.length),
        total: verbs.length,
        pct: Math.min(100, Math.round((i + batchSize) / verbs.length * 100)),
      });
    }

    await new Promise(r => setTimeout(r, 1000));
  }

  return classifications;
}


async function classifyVerbs(verbs, onProgress) {
  const backend = Storage.getSetting('llmBackend', 'anthropic');
  if (backend === 'openai') {
    return classifyVerbsOpenAI(verbs, onProgress);
  }
  return classifyVerbsAnthropic(verbs, onProgress);
}


// ── Single verb classification ──────────────────────────────

async function classifySingleVerb(verb) {
  const results = await classifyVerbs([verb]);
  return results[0] || null;
}


const ApiClient = {
  generateEmbeddings,
  classifyVerbs,
  classifyVerbsOpenAI,
  classifyVerbsAnthropic,
  classifySingleVerb,
  OPERATOR_DEFINITIONS,
};

export default ApiClient;
