/* ═══════════════════════════════════════════════════
   Prompt Enrichment Studio – client-side JavaScript
   ═══════════════════════════════════════════════════ */

'use strict';

// ── DOM refs ──────────────────────────────────────────
const promptInput      = document.getElementById('promptInput');
const charCount        = document.getElementById('charCount');
const enrichBtn        = document.getElementById('enrichBtn');
const generateBtn      = document.getElementById('generateBtn');
const loadingSection   = document.getElementById('loadingSection');
const resultsSection   = document.getElementById('resultsSection');

// loading step nodes
const LS = [null, ...Array.from({length: 6}, (_, i) => document.getElementById(`ls${i + 1}`))];

// ── Character counter ─────────────────────────────────
promptInput.addEventListener('input', () => {
  charCount.textContent = promptInput.value.length;
});

// ── Set example prompt ────────────────────────────────
function setExample(text) {
  promptInput.value = text;
  charCount.textContent = text.length;
  promptInput.focus();
}

// ─────────────────────────────────────────────────────
//  Safe JSON fetch helper
//  Always returns { ok, data, error }
//  – never throws, never tries JSON.parse on HTML
// ─────────────────────────────────────────────────────
async function apiFetch(url, body) {
  let res;
  try {
    res = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
  } catch (networkErr) {
    return { ok: false, data: null, error: `Network error — is the server running? (${networkErr.message})` };
  }

  // Try to parse as JSON; fall back to raw text
  const text = await res.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (_) {
    // Server returned non-JSON (e.g. HTML 500 page)
    const snippet = text.slice(0, 200).replace(/</g, '&lt;');
    return {
      ok: false,
      data: null,
      error: `Server returned HTTP ${res.status} with non-JSON body. ` +
             `Make sure all Python packages are installed and the server started cleanly.\n\n` +
             `Raw response:\n${snippet}`,
    };
  }

  if (!res.ok) {
    // JSON error body – FastAPI uses { "detail": "..." } or we use { "error": "..." }
    const msg = data?.error || data?.detail || `HTTP ${res.status}`;
    return { ok: false, data, error: msg };
  }

  return { ok: true, data, error: null };
}

// ── Show inline error banner ──────────────────────────
function showError(message) {
  loadingSection.classList.add('hidden');

  // Remove any old error banner
  document.getElementById('errorBanner')?.remove();

  const banner = document.createElement('div');
  banner.id = 'errorBanner';
  banner.className = 'error-banner glass-card';
  banner.innerHTML = `
    <div class="error-icon">⚠️</div>
    <div class="error-body">
      <h3>Something went wrong</h3>
      <pre class="error-message">${escHtml(message)}</pre>
      <p class="error-hint">
        If you see <strong>"Missing Python package"</strong>, install dependencies first:<br>
        <code>pip install gensim google-generativeai rouge-score scikit-learn sentence-transformers numpy</code><br>
        Then restart the server with <code>python3 app.py</code>.
      </p>
    </div>
    <button class="error-close" onclick="this.parentElement.remove()">✕</button>`;
  document.querySelector('.app-shell').insertBefore(
    banner,
    document.getElementById('loadingSection')
  );
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Loading animation helpers ─────────────────────────
let loaderTimer = null;

function startLoader(steps) {
  document.getElementById('errorBanner')?.remove();
  resultsSection.classList.add('hidden');
  loadingSection.classList.remove('hidden');

  for (let i = 1; i <= 6; i++) LS[i].className = 'loader-step';

  let current = 0;
  const delays = [300, 700, 500, 1200, 1200, 600];

  function advanceStep() {
    if (current > 0) LS[current].className = 'loader-step done';
    current++;
    if (current <= steps) {
      LS[current].className = 'loader-step active';
      if (current < steps) loaderTimer = setTimeout(advanceStep, delays[current - 1]);
    }
  }
  advanceStep();
}

function stopLoader(stepsCompleted) {
  clearTimeout(loaderTimer);
  for (let i = 1; i <= stepsCompleted; i++) LS[i].className = 'loader-step done';
  setTimeout(() => loadingSection.classList.add('hidden'), 400);
}

// ── Simple markdown renderer ──────────────────────────
function renderMarkdown(text) {
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/^\s*[-*•] (.+)$/gm, '<li>$1</li>')
    .replace(/^\s*\d+\.\s(.+)$/gm, '<li>$1</li>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, '<br>');
  html = html.replace(/(<li>(?:.|\n)*?<\/li>)/g, '<ul>$1</ul>');
  return `<p>${html}</p>`;
}

// ── Gauge SVG ─────────────────────────────────────────
function makeGauge(value, label, color) {
  const r = 40, cx = 50, cy = 50;
  const angle = Math.PI - value * Math.PI;
  const lx = cx + r * Math.cos(Math.PI), ly = cy + r * Math.sin(Math.PI);
  const ex = cx + r * Math.cos(0),       ey = cy + r * Math.sin(0);
  const vx = cx + r * Math.cos(angle),   vy = cy + r * Math.sin(angle);

  const el = document.createElement('div');
  el.className = 'gauge-card';
  el.innerHTML = `
    <svg class="gauge-svg" viewBox="0 0 100 55">
      <path d="M ${lx} ${ly} A ${r} ${r} 0 0 1 ${ex} ${ey}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="6" stroke-linecap="round"/>
      <path d="M ${lx} ${ly} A ${r} ${r} 0 0 1 ${vx} ${vy}" fill="none" stroke="${color}" stroke-width="6" stroke-linecap="round"/>
      <circle cx="${vx}" cy="${vy}" r="4" fill="${color}"/>
    </svg>
    <div class="gauge-val" style="color:${color}">${(value * 100).toFixed(1)}%</div>
    <div class="gauge-label">${label}</div>`;
  return el;
}

// ── Embedding grid ────────────────────────────────────
function showEmbeddings(keyword_map) {
  const grid = document.getElementById('embeddingGrid');
  grid.innerHTML = '';

  if (!keyword_map || Object.keys(keyword_map).length === 0) {
    grid.innerHTML = '<p style="color:var(--muted);font-size:.85rem">No keywords extracted or no embeddings found.</p>';
    return;
  }

  for (const [kw, words] of Object.entries(keyword_map)) {
    const card = document.createElement('div');
    card.className = 'kw-card';
    const maxScore = words[0]?.score || 1;

    const rows = words.map(({ word, score }) => {
      const pct = Math.min(100, (score / Math.max(maxScore, 0.01)) * 100);
      return `<div class="sim-word-row">
        <span class="sim-word">${escHtml(word)}</span>
        <div class="sim-score-bar-wrap"><div class="sim-score-bar" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="sim-score-val">${score.toFixed(3)}</span>
      </div>`;
    }).join('');

    card.innerHTML = `<div class="kw-title">${escHtml(kw)}</div><div class="sim-word-list">${rows}</div>`;
    grid.appendChild(card);
  }
}

// ── Metrics ───────────────────────────────────────────
const METRIC_META = {
  rouge1_f:    { label: 'ROUGE-1 F1',       color: '#4f8ef7', interp: 'Unigram overlap (0–1); higher = more shared content' },
  rouge2_f:    { label: 'ROUGE-2 F1',       color: '#9b59f7', interp: 'Bigram overlap (0–1); higher = more coherence overlap' },
  rougeL_f:    { label: 'ROUGE-L F1',       color: '#22d3ee', interp: 'Longest common subsequence overlap' },
  cosine_sim:  { label: 'Cosine Similarity', color: '#34d399', interp: 'Semantic similarity (0–1); 1 = identical meaning' },
  orig_words:  { label: 'Original Words',    color: '#fbbf24', interp: 'Word count of original response' },
  enrch_words: { label: 'Enriched Words',    color: '#fbbf24', interp: 'Word count of enriched response' },
  word_delta:  { label: 'Word Δ (delta)',    color: '#f87171', interp: 'Extra words added by prompt enrichment' },
};

function showMetrics(metrics) {
  const gaugeRow = document.getElementById('gaugeRow');
  gaugeRow.innerHTML = '';
  [['rouge1_f','#4f8ef7'],['rouge2_f','#9b59f7'],['rougeL_f','#22d3ee'],['cosine_sim','#34d399']]
    .forEach(([k, c]) => gaugeRow.appendChild(makeGauge(metrics[k], METRIC_META[k].label, c)));

  const tbody = document.getElementById('metricsTableBody');
  tbody.innerHTML = '';
  for (const [k, v] of Object.entries(metrics)) {
    const meta = METRIC_META[k] || { label: k, color: '#e8eaf0', interp: '' };
    const display = typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(4) : v;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="metric-key">${meta.label}</span></td>
      <td><span class="metric-val" style="color:${meta.color}">${display}</span></td>
      <td><span class="metric-interp">${meta.interp}</span></td>`;
    tbody.appendChild(tr);
  }
}

// ── Summary cards ─────────────────────────────────────
function showSummary(metrics) {
  const container = document.getElementById('summaryContent');
  container.innerHTML = '';
  const { word_delta, cosine_sim, rouge1_f, orig_words, enrch_words } = metrics;

  const cards = [];

  if (word_delta > 0)
    cards.push({ type:'positive', icon:'📝', title:`${word_delta} more words`, body:`Enriched response is <strong>${word_delta} words longer</strong> (${enrch_words} vs ${orig_words}), providing more detail.` });
  else if (word_delta < 0)
    cards.push({ type:'neutral',  icon:'✂️', title:`${Math.abs(word_delta)} fewer words`, body:`Enriched response is shorter but may be more focused.` });
  else
    cards.push({ type:'neutral',  icon:'⚖️', title:'Equal length', body:'Both responses have the same word count.' });

  if (cosine_sim > 0.90)
    cards.push({ type:'positive', icon:'🎯', title:'Very high relevance', body:`Cosine similarity = <strong>${cosine_sim.toFixed(4)}</strong>. Enrichment stayed on-topic.` });
  else if (cosine_sim > 0.75)
    cards.push({ type:'positive', icon:'✅', title:'Good semantic overlap', body:`Cosine similarity = <strong>${cosine_sim.toFixed(4)}</strong>. Enrichment added breadth while keeping relevance.` });
  else
    cards.push({ type:'negative', icon:'⚠️', title:'Significant semantic shift', body:`Cosine similarity = <strong>${cosine_sim.toFixed(4)}</strong>. Enrichment shifted topic noticeably.` });

  if (rouge1_f > 0.5)
    cards.push({ type:'positive', icon:'🔗', title:'High ROUGE overlap', body:`ROUGE-1 F1 = <strong>${rouge1_f.toFixed(4)}</strong>. Strong shared vocabulary.` });
  else if (rouge1_f > 0.3)
    cards.push({ type:'neutral',  icon:'🔍', title:'Moderate ROUGE overlap', body:`ROUGE-1 F1 = <strong>${rouge1_f.toFixed(4)}</strong>. New terminology introduced while core concepts retained.` });
  else
    cards.push({ type:'negative', icon:'📊', title:'Low ROUGE overlap', body:`ROUGE-1 F1 = <strong>${rouge1_f.toFixed(4)}</strong>. Responses diverge in vocabulary.` });

  const ok = cosine_sim > 0.75 && word_delta >= 0;
  cards.push({ type: ok ? 'positive' : 'neutral', icon: ok ? '🚀' : '💡',
    title: ok ? 'Enrichment effective!' : 'Mixed results',
    body:  ok ? 'Word embeddings successfully enriched the prompt — richer, more detailed response.' : 'Enrichment had mixed impact. Try adjusting keyword count.' });

  for (const c of cards) {
    const div = document.createElement('div');
    div.className = `summary-card ${c.type}`;
    div.innerHTML = `<div class="summary-icon">${c.icon}</div><div class="summary-text"><h4>${c.title}</h4><p>${c.body}</p></div>`;
    container.appendChild(div);
  }
}

// ── Preview enrichment only ───────────────────────────
async function enrichOnly() {
  const prompt = promptInput.value.trim();
  if (!prompt) { alert('Please enter a prompt first.'); return; }

  setButtonsDisabled(true);
  startLoader(3);

  const { ok, data, error } = await apiFetch('/api/enrich', { prompt });
  stopLoader(3);

  if (!ok) { showError(error); setButtonsDisabled(false); return; }

  showPartialResults(data);
  setButtonsDisabled(false);
}

function showPartialResults(data) {
  document.getElementById('originalPromptBox').textContent = data.original;
  document.getElementById('enrichedPromptBox').textContent = data.enriched;
  showEmbeddings(data.keyword_map);
  document.getElementById('responseSection').classList.add('hidden');
  document.getElementById('summaryCard').classList.add('hidden');
  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Full pipeline ─────────────────────────────────────
async function runPipeline() {
  const prompt = promptInput.value.trim();
  if (!prompt) { alert('Please enter a prompt first.'); return; }

  setButtonsDisabled(true);
  startLoader(6);

  const { ok, data, error } = await apiFetch('/api/generate', { prompt });
  stopLoader(6);

  if (!ok) { showError(error); setButtonsDisabled(false); return; }

  showFullResults(data);
  setButtonsDisabled(false);
}

function showFullResults(data) {
  document.getElementById('originalPromptBox').textContent = data.original_prompt;
  document.getElementById('enrichedPromptBox').textContent = data.enriched_prompt;
  showEmbeddings(data.keyword_map);
  document.getElementById('originalResponseBox').innerHTML  = renderMarkdown(data.original_response);
  document.getElementById('enrichedResponseBox').innerHTML  = renderMarkdown(data.enriched_response);
  document.getElementById('origWordCount').textContent      = `${data.metrics.orig_words} words`;
  document.getElementById('enrichedWordCount').textContent  = `${data.metrics.enrch_words} words`;
  showMetrics(data.metrics);
  showSummary(data.metrics);
  document.getElementById('responseSection').classList.remove('hidden');
  document.getElementById('summaryCard').classList.remove('hidden');
  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Helper ────────────────────────────────────────────
function setButtonsDisabled(disabled) {
  enrichBtn.disabled   = disabled;
  generateBtn.disabled = disabled;
}

// ── On page load: ping /api/status ───────────────────
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const res  = await fetch('/api/status');
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch (_) { return; }

    if (!data.ready && data.error) {
      showError(data.error);
    }
  } catch (_) { /* server not running yet – silent */ }
});
