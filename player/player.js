/**
 * player.js – SegmentPlayer interactive logic
 *
 * Features:
 * - Load video (local file) and JSON segmentation metadata
 * - Color-coded timeline bar and progress overlay
 * - Segment list panel with play/skip per segment
 * - Content Map (percentage breakdown by type)
 * - Chapter navigation from JSON chapters array
 * - Play Content Only / Play Non-Content Only modes
 * - Review mode: click segment to reclassify, export corrected JSON
 * - Keyboard shortcuts (Space, N, P, arrows, R, E)
 */

'use strict';

// ─── State ──────────────────────────────────────────────────────────────────
const state = {
  segments: [],
  chapters: [],
  duration: 0,
  meta: null,
  mode: 'normal',       // 'normal' | 'content-only' | 'noncontent-only'
  activeSegIdx: -1,
  reviewMode: false,
  reviewTarget: -1,     // segment index being reclassified
};

const ALL_TYPES = [
  'content', 'intro', 'outro', 'ad', 'self_promotion',
  'recap', 'transition', 'dead_air',
];

// ─── DOM refs ───────────────────────────────────────────────────────────────
const video        = document.getElementById('video-el');
const overlay      = document.getElementById('video-overlay');
const badge        = document.getElementById('segment-badge');
const badgeLabel   = document.getElementById('segment-badge-label');
const progressBg   = document.getElementById('progress-bar-bg');
const progressFill = document.getElementById('progress-fill');
const progressThumb= document.getElementById('progress-thumb');
const segOverlays  = document.getElementById('segment-overlays');
const timeDisplay  = document.getElementById('time-display');
const segList      = document.getElementById('segment-list');
const emptyPanel   = document.getElementById('empty-panel');
const panelStats   = document.getElementById('panel-stats');
const statDuration = document.getElementById('stat-duration');
const statNc       = document.getElementById('stat-noncontent');
const statSegs     = document.getElementById('stat-segments');
const timelineBar  = document.getElementById('timeline-bar');
const btnPlay      = document.getElementById('btn-play');
const btnPrev      = document.getElementById('btn-prev-seg');
const btnNext      = document.getElementById('btn-next-seg');
const btnMute      = document.getElementById('btn-mute');
const volSlider    = document.getElementById('volume-slider');
const btnContent   = document.getElementById('btn-play-content');
const btnSkipNc    = document.getElementById('btn-skip-noncontent');
const btnNormal    = document.getElementById('btn-normal-mode');
const btnReview    = document.getElementById('btn-review-toggle');
const btnExport    = document.getElementById('btn-export-json');
const contentMapWrap  = document.getElementById('content-map-wrapper');
const contentMapBar   = document.getElementById('content-map-bar');
const contentMapStats = document.getElementById('content-map-stats');
const chapterNav   = document.getElementById('chapter-nav');
const chapterList  = document.getElementById('chapter-list');
const reviewModal  = document.getElementById('review-modal');

// ─── Segment type → CSS class mapping ───────────────────────────────────────
function segClass(type) {
  const map = {
    video_content:  'content',
    content:        'content',
    intro:          'intro',
    outro:          'outro',
    ad:             'ad',
    self_promotion: 'self_promotion',
    recap:          'recap',
    transition:     'transition',
    dead_air:       'dead_air',
    non_content:    'ad',
  };
  return map[type] || 'ad';
}

function isNonContent(seg) {
  if (typeof seg.is_content === 'boolean') return !seg.is_content;
  return seg.type === 'non_content' || seg.type !== 'video_content';
}

function formatTime(sec) {
  sec = Math.max(0, sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${pad(h)}:${pad(m)}:${pad(s)}`;
  return `${pad(m)}:${pad(s)}`;
}
function pad(n) { return String(Math.floor(n)).padStart(2, '0'); }

function segLabel(seg) {
  return (seg.segment_label || seg.type || 'unknown').replace(/_/g, ' ');
}

function getColorForClass(cls) {
  const map = {
    content:        '#22c55e',
    intro:          '#f59e0b',
    outro:          '#a78bfa',
    ad:             '#ef4444',
    recap:          '#38bdf8',
    transition:     '#64748b',
    dead_air:       '#1e293b',
    holding_screen: '#374151',
    self_promotion: '#f97316',
  };
  return map[cls] || '#6b7280';
}

// ─── File Loading ────────────────────────────────────────────────────────────
document.getElementById('load-video').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  video.src = URL.createObjectURL(file);
  overlay.classList.add('hidden');
  toast(`Loaded: ${file.name}`);
});

document.getElementById('load-json').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      const meta = JSON.parse(ev.target.result);
      loadMetadata(meta);
      toast(`Metadata loaded: ${meta.video_filename || file.name}`);
    } catch (err) {
      toast(`Invalid JSON: ${err.message}`);
    }
  };
  reader.readAsText(file);
});

function loadMetadata(meta) {
  state.meta = meta;
  state.segments = meta.timeline_segments || [];
  state.chapters = meta.chapters || [];
  state.duration = meta.output_duration_seconds || meta.original_video_duration_seconds || 0;

  const nc_segments = state.segments.filter(s => isNonContent(s));
  const total_nc_dur = nc_segments.reduce((acc, s) => acc + (s.duration_seconds || 0), 0);

  panelStats.classList.remove('hidden');
  emptyPanel.style.display = 'none';
  statDuration.textContent = formatTime(state.duration);
  statNc.textContent = nc_segments.length > 0
    ? `${nc_segments.length} (${formatTime(total_nc_dur)})`
    : '0';
  statSegs.textContent = state.segments.length;

  renderSegmentList();
  renderTimeline();
  renderProgressOverlays();
  renderContentMap();
  renderChapters();
}

// ─── Segment List Panel ──────────────────────────────────────────────────────
function renderSegmentList() {
  segList.innerHTML = '';
  state.segments.forEach((seg, i) => {
    const cls = segClass(seg.segment_label || seg.type);
    const nc  = isNonContent(seg);
    const card = document.createElement('div');
    card.className = `seg-card is-${cls}`;
    card.dataset.idx = i;

    const start = seg.final_video_start_seconds || 0;
    const end   = seg.final_video_end_seconds || 0;
    const dur   = seg.duration_seconds || (end - start);
    const label = segLabel(seg);
    const skip  = seg.skip_suggestion ? `<span class="seg-skip-hint">${seg.skip_reason || 'Skip'}</span>` : '';
    const conf  = seg.confidence ? ` &middot; ${Math.round(seg.confidence * 100)}%` : '';

    card.innerHTML = `
      <div class="seg-card-top">
        <span class="seg-type-badge badge-${cls}">${label}</span>
        <button class="seg-skip-btn" data-start="${start}" data-idx="${i}">&#x25B6; Play</button>
      </div>
      <div class="seg-time">${formatTime(start)} &rarr; ${formatTime(end)}</div>
      <div class="seg-duration">${dur.toFixed(1)}s${nc ? ' &middot; Skip' : ''}${conf}</div>
      ${skip}
    `;

    card.addEventListener('click', () => {
      if (state.reviewMode) {
        openReviewModal(i);
      } else {
        seekToSegment(i);
      }
    });
    card.querySelector('.seg-skip-btn').addEventListener('click', ev => {
      ev.stopPropagation();
      seekToSegment(i);
      video.play();
    });

    segList.appendChild(card);
  });
}

function highlightActiveCard(idx) {
  document.querySelectorAll('.seg-card').forEach((c, i) => {
    c.classList.toggle('active', i === idx);
  });
  const active = segList.querySelector('.seg-card.active');
  if (active) active.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Content Map ─────────────────────────────────────────────────────────────
function renderContentMap() {
  if (!state.segments.length) return;
  contentMapWrap.classList.remove('hidden');

  // Aggregate by type
  const byType = {};
  let total = 0;
  state.segments.forEach(seg => {
    const label = seg.segment_label || seg.type || 'unknown';
    const dur = seg.duration_seconds || 0;
    byType[label] = (byType[label] || 0) + dur;
    total += dur;
  });

  // Render stacked bar
  contentMapBar.innerHTML = '';
  const sortedTypes = Object.keys(byType).sort((a, b) => byType[b] - byType[a]);
  sortedTypes.forEach(type => {
    const pct = (byType[type] / total) * 100;
    const cls = segClass(type);
    const el = document.createElement('div');
    el.className = `content-map-seg cm-${cls}`;
    el.style.width = `${pct}%`;
    el.style.background = getColorForClass(cls);
    el.title = `${type.replace(/_/g, ' ')}: ${formatTime(byType[type])} (${pct.toFixed(1)}%)`;
    contentMapBar.appendChild(el);
  });

  // Stats text
  contentMapStats.innerHTML = sortedTypes.map(type => {
    const pct = ((byType[type] / total) * 100).toFixed(1);
    const cls = segClass(type);
    const color = getColorForClass(cls);
    return `<span class="cm-stat" style="color:${color}">&#x25A0; ${type.replace(/_/g, ' ')} ${pct}%</span>`;
  }).join('');
}

// ─── Chapter Navigation ─────────────────────────────────────────────────────
function renderChapters() {
  if (!state.chapters.length) {
    chapterNav.classList.add('hidden');
    return;
  }
  chapterNav.classList.remove('hidden');
  chapterList.innerHTML = '';

  state.chapters.forEach((ch, i) => {
    const cls = segClass(ch.segment_label || 'content');
    const el = document.createElement('div');
    el.className = `chapter-item ch-${cls}`;
    el.innerHTML = `
      <span class="chapter-time">${ch.start_formatted || formatTime(ch.start_seconds)}</span>
      <span class="chapter-title">${ch.title}</span>
    `;
    el.addEventListener('click', () => {
      video.currentTime = ch.start_seconds || 0;
      video.play();
    });
    chapterList.appendChild(el);
  });
}

// ─── Timeline Bar ────────────────────────────────────────────────────────────
function renderTimeline() {
  timelineBar.innerHTML = '';
  if (!state.segments.length || !state.duration) return;
  state.segments.forEach((seg, i) => {
    const start = seg.final_video_start_seconds || 0;
    const dur   = seg.duration_seconds || 1;
    const pct   = (dur / state.duration) * 100;
    const cls   = segClass(seg.segment_label || seg.type);

    const el = document.createElement('div');
    el.className = `timeline-seg tl-${cls}`;
    el.style.width = `${pct}%`;
    el.title = `${segLabel(seg)}: ${formatTime(start)} (${dur.toFixed(1)}s)`;
    el.addEventListener('click', () => { seekToSegment(i); video.play(); });
    timelineBar.appendChild(el);
  });
}

// ─── Progress Overlays ──────────────────────────────────────────────────────
function renderProgressOverlays() {
  segOverlays.innerHTML = '';
  if (!state.segments.length || !state.duration) return;
  state.segments.forEach(seg => {
    const start = seg.final_video_start_seconds || 0;
    const dur   = seg.duration_seconds || 0;
    const cls   = segClass(seg.segment_label || seg.type);
    const left  = (start / state.duration) * 100;
    const width = (dur   / state.duration) * 100;
    const el    = document.createElement('div');
    el.style.cssText = `position:absolute;left:${left}%;width:${width}%;top:0;bottom:0;opacity:0.35;pointer-events:none;`;
    el.className = `tl-${cls}`;
    el.style.background = getColorForClass(cls);
    segOverlays.appendChild(el);
  });
}

// ─── Seeking ────────────────────────────────────────────────────────────────
function seekToSegment(idx) {
  if (!state.segments[idx]) return;
  video.currentTime = state.segments[idx].final_video_start_seconds || 0;
  state.activeSegIdx = idx;
  highlightActiveCard(idx);
  updateBadge(state.segments[idx]);
}

function updateBadge(seg) {
  if (!seg) { badge.classList.add('hidden'); return; }
  const cls = segClass(seg.segment_label || seg.type);
  badge.classList.remove('hidden');
  badgeLabel.textContent = segLabel(seg).toUpperCase();
  badge.style.borderColor = getColorForClass(cls);
  badge.style.color = getColorForClass(cls);
}

function getCurrentSegment(t) {
  for (let i = 0; i < state.segments.length; i++) {
    const seg = state.segments[i];
    const start = seg.final_video_start_seconds || 0;
    const end   = seg.final_video_end_seconds || Infinity;
    if (t >= start && t < end) return i;
  }
  return -1;
}

// ─── Video Events ───────────────────────────────────────────────────────────
video.addEventListener('timeupdate', () => {
  const t   = video.currentTime;
  const dur = video.duration || state.duration || 1;
  const pct = (t / dur) * 100;

  progressFill.style.width = `${pct}%`;
  progressThumb.style.left = `${pct}%`;
  timeDisplay.textContent  = `${formatTime(t)} / ${formatTime(dur)}`;

  const idx = getCurrentSegment(t);
  if (idx !== state.activeSegIdx) {
    state.activeSegIdx = idx;
    highlightActiveCard(idx);
    updateBadge(state.segments[idx]);
  }

  // Mode logic
  if (state.mode === 'content-only') {
    const seg = state.segments[idx];
    if (seg && isNonContent(seg)) {
      video.currentTime = seg.final_video_end_seconds || t;
    }
  } else if (state.mode === 'noncontent-only') {
    const seg = state.segments[idx];
    if (seg && !isNonContent(seg)) {
      video.currentTime = seg.final_video_end_seconds || t;
    }
  }
});

video.addEventListener('loadedmetadata', () => {
  if (!state.duration) state.duration = video.duration;
});
video.addEventListener('play',  () => { btnPlay.textContent = '⏸'; });
video.addEventListener('pause', () => { btnPlay.textContent = '▶'; });
video.addEventListener('ended', () => { btnPlay.textContent = '▶'; });

// ─── Controls ───────────────────────────────────────────────────────────────
btnPlay.addEventListener('click', () => { video.paused ? video.play() : video.pause(); });
btnPrev.addEventListener('click', () => seekToSegment(Math.max(0, state.activeSegIdx - 1)));
btnNext.addEventListener('click', () => seekToSegment(Math.min(state.activeSegIdx + 1, state.segments.length - 1)));
btnMute.addEventListener('click', () => { video.muted = !video.muted; btnMute.textContent = video.muted ? '\u{1F507}' : '\u{1F50A}'; });
volSlider.addEventListener('input', () => { video.volume = volSlider.value; });

progressBg.addEventListener('click', e => {
  const rect = progressBg.getBoundingClientRect();
  video.currentTime = ((e.clientX - rect.left) / rect.width) * (video.duration || state.duration);
});

// ─── Mode Buttons ───────────────────────────────────────────────────────────
function setMode(m) {
  state.mode = m;
  btnContent.classList.toggle('active', m === 'content-only');
  btnSkipNc.classList.toggle('active', m === 'noncontent-only');
  btnNormal.classList.toggle('active', m === 'normal');
  toast(`Mode: ${{ normal: 'Normal', 'content-only': 'Content Only', 'noncontent-only': 'Non-Content Only' }[m]}`);
}

btnContent.addEventListener('click', () => {
  setMode(state.mode === 'content-only' ? 'normal' : 'content-only');
  if (state.mode === 'content-only') {
    const first = state.segments.findIndex(s => !isNonContent(s));
    if (first >= 0) seekToSegment(first);
    video.play();
  }
});
btnSkipNc.addEventListener('click', () => {
  setMode(state.mode === 'noncontent-only' ? 'normal' : 'noncontent-only');
  if (state.mode === 'noncontent-only') {
    const first = state.segments.findIndex(s => isNonContent(s));
    if (first >= 0) seekToSegment(first);
    video.play();
  }
});
btnNormal.addEventListener('click', () => setMode('normal'));

// ─── Review Mode ────────────────────────────────────────────────────────────
btnReview.addEventListener('click', () => {
  state.reviewMode = !state.reviewMode;
  btnReview.classList.toggle('active', state.reviewMode);
  btnExport.style.display = state.reviewMode ? '' : 'none';
  document.body.classList.toggle('review-active', state.reviewMode);
  toast(state.reviewMode ? 'Review mode ON: click a segment to reclassify' : 'Review mode OFF');
});

function openReviewModal(segIdx) {
  state.reviewTarget = segIdx;
  const seg = state.segments[segIdx];
  const info = document.getElementById('review-modal-info');
  info.textContent = `Segment ${segIdx + 1}: ${formatTime(seg.final_video_start_seconds)} - ${formatTime(seg.final_video_end_seconds)} (${seg.duration_seconds.toFixed(1)}s)`;

  const opts = document.getElementById('review-type-options');
  opts.innerHTML = '';
  ALL_TYPES.forEach(type => {
    const btn = document.createElement('button');
    const cls = segClass(type);
    btn.className = `review-type-btn ${(seg.segment_label || seg.type) === type ? 'selected' : ''}`;
    btn.style.borderColor = getColorForClass(cls);
    btn.textContent = type.replace(/_/g, ' ');
    btn.dataset.type = type;
    btn.addEventListener('click', () => {
      opts.querySelectorAll('.review-type-btn').forEach(b => b.classList.remove('selected'));
      btn.classList.add('selected');
    });
    opts.appendChild(btn);
  });

  reviewModal.classList.remove('hidden');
}

document.getElementById('review-cancel').addEventListener('click', () => {
  reviewModal.classList.add('hidden');
});

document.getElementById('review-apply').addEventListener('click', () => {
  const selected = document.querySelector('#review-type-options .review-type-btn.selected');
  if (!selected) return;
  const newType = selected.dataset.type;
  const seg = state.segments[state.reviewTarget];

  // Update the segment in state
  seg.segment_label = newType;
  seg.type = newType === 'content' ? 'video_content' : newType;
  seg.is_content = newType === 'content';

  // Re-render everything
  renderSegmentList();
  renderTimeline();
  renderProgressOverlays();
  renderContentMap();
  reviewModal.classList.add('hidden');
  toast(`Segment ${state.reviewTarget + 1} reclassified as "${newType.replace(/_/g, ' ')}"`);
});

// Export corrected JSON
btnExport.addEventListener('click', () => {
  if (!state.meta) return;
  const corrected = JSON.parse(JSON.stringify(state.meta));
  corrected.timeline_segments = state.segments;
  corrected._corrected = true;
  corrected._corrected_at = new Date().toISOString();

  const blob = new Blob([JSON.stringify(corrected, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${corrected.video_filename || 'segments'}_corrected.json`;
  a.click();
  URL.revokeObjectURL(url);
  toast('Corrected JSON exported');
});

// ─── Keyboard shortcuts ─────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.code) {
    case 'Space': e.preventDefault(); video.paused ? video.play() : video.pause(); break;
    case 'ArrowRight': video.currentTime += 5; break;
    case 'ArrowLeft':  video.currentTime -= 5; break;
    case 'KeyN': btnNext.click(); break;
    case 'KeyP': btnPrev.click(); break;
    case 'KeyR': btnReview.click(); break;
    case 'KeyE': if (state.reviewMode) btnExport.click(); break;
    case 'Escape': reviewModal.classList.add('hidden'); break;
  }
});

// ─── Toasts ─────────────────────────────────────────────────────────────────
function toast(msg, ms = 2800) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

toast('Load a video and JSON to begin');
