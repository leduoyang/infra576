/**
 * player.js – SegmentPlayer interactive logic
 *
 * Features:
 * - Load video (local file) and JSON segmentation metadata
 * - Color-coded timeline bar and progress overlay
 * - Segment list panel with play/skip per segment
 * - Play Content Only mode (auto-skips non-content)
 * - Skip Non-Content mode (skips ahead if non-content plays)
 * - Keyboard shortcuts
 */

'use strict';

// ─── State ──────────────────────────────────────────────────────────────────
const state = {
  segments: [],        // parsed timeline_segments from JSON
  duration: 0,
  meta: null,          // full JSON metadata
  mode: 'normal',      // 'normal' | 'content-only' | 'skip-noncontent'
  activeSegIdx: -1,
  dragging: false,
};

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

// ─── Segment type → CSS class mapping ───────────────────────────────────────
function segClass(type) {
  const map = {
    video_content: 'content',
    content:       'content',
    intro:         'intro',
    outro:         'outro',
    ad:            'ad',
    non_content:   'ad',
  };
  return map[type] || 'ad';
}


function isNonContent(seg) {
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

// ─── File Loading ────────────────────────────────────────────────────────────
document.getElementById('load-video').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  video.src = url;
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
      toast(`❌ Invalid JSON: ${err.message}`);
    }
  };
  reader.readAsText(file);
});

function loadMetadata(meta) {
  state.meta = meta;
  state.segments = meta.timeline_segments || [];
  // Crucial: Use output_duration_seconds to align with the actual video file
  state.duration = meta.output_duration_seconds || meta.original_video_duration_seconds || 0;

  // Derive stats from timeline segments to ensure consistency with the view
  const nc_segments = state.segments.filter(s => isNonContent(s));
  const nc_count = nc_segments.length;
  const total_nc_dur = nc_segments.reduce((acc, s) => acc + (s.duration_seconds || 0), 0);
  
  panelStats.classList.remove('hidden');
  emptyPanel.style.display = 'none';
  statDuration.textContent = formatTime(state.duration);
  statNc.textContent = nc_count > 0 ? `${nc_count} (${formatTime(total_nc_dur)})` : '0';
  statSegs.textContent = state.segments.length;

  renderSegmentList();
  renderTimeline();
  renderProgressOverlays();
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

    card.innerHTML = `
      <div class="seg-card-top">
        <span class="seg-type-badge badge-${cls}">${label}</span>
        <button class="seg-skip-btn" data-start="${start}" data-idx="${i}">▶ Play</button>
      </div>
      <div class="seg-time">${formatTime(start)} → ${formatTime(end)}</div>
      <div class="seg-duration">${dur.toFixed(1)}s${nc ? ' · Skip' : ''}</div>
    `;

    card.addEventListener('click', () => seekToSegment(i));
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
  // Scroll into view
  const active = segList.querySelector('.seg-card.active');
  if (active) active.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Timeline Bar ─────────────────────────────────────────────────────────────
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
    el.addEventListener('click', () => {
      seekToSegment(i);
      video.play();
    });
    timelineBar.appendChild(el);
  });

  // Clicking on timeline seeks
  timelineBar.addEventListener('click', e => {
    const rect = timelineBar.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    video.currentTime = pct * state.duration;
  });
}

// ─── Progress Overlays (on progress bar) ────────────────────────────────────
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
    el.style.cssText = `
      position:absolute; left:${left}%; width:${width}%;
      top:0; bottom:0; opacity:0.35; pointer-events:none;
    `;
    el.className = `tl-${cls}`;
    // color by class
    el.style.background = getColorForClass(cls);
    segOverlays.appendChild(el);
  });
}

function getColorForClass(cls) {
  const map = {
    content:       '#22c55e',
    intro:         '#f59e0b',
    outro:         '#a78bfa',
    ad:            '#ef4444',
    recap:         '#38bdf8',
    transition:    '#64748b',
    dead_air:      '#1e293b',
    holding_screen:'#374151',
    self_promotion:'#f97316',
  };
  return map[cls] || '#6b7280';
}

// ─── Seeking ─────────────────────────────────────────────────────────────────
function seekToSegment(idx) {
  if (!state.segments[idx]) return;
  const seg = state.segments[idx];
  video.currentTime = seg.final_video_start_seconds || 0;
  state.activeSegIdx = idx;
  highlightActiveCard(idx);
  updateBadge(seg);
}

function updateBadge(seg) {
  if (!seg) { badge.classList.add('hidden'); return; }
  const cls = segClass(seg.segment_label || seg.type);
  const label = segLabel(seg);
  badge.classList.remove('hidden');
  badgeLabel.textContent = label.toUpperCase();
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

  // Progress bar
  const pct = (t / dur) * 100;
  progressFill.style.width = `${pct}%`;
  progressThumb.style.left = `${pct}%`;
  timeDisplay.textContent  = `${formatTime(t)} / ${formatTime(dur)}`;

  // Active segment detection
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
      // Skip to end of this segment
      const endTime = seg.final_video_end_seconds || t;
      video.currentTime = endTime;
    }
  } else if (state.mode === 'noncontent-only') {
    const seg = state.segments[idx];
    if (seg && !isNonContent(seg)) {
      // Skip to end of this segment
      const endTime = seg.final_video_end_seconds || t;
      video.currentTime = endTime;
    }
  }
});


video.addEventListener('loadedmetadata', () => {
  if (!state.duration) state.duration = video.duration;
});

video.addEventListener('play',  () => { btnPlay.textContent = '⏸'; });
video.addEventListener('pause', () => { btnPlay.textContent = '▶'; });
video.addEventListener('ended', () => { btnPlay.textContent = '▶'; });

// ─── Controls ────────────────────────────────────────────────────────────────
btnPlay.addEventListener('click', () => { video.paused ? video.play() : video.pause(); });

btnPrev.addEventListener('click', () => {
  const prev = state.activeSegIdx > 0 ? state.activeSegIdx - 1 : 0;
  seekToSegment(prev);
});
btnNext.addEventListener('click', () => {
  const next = Math.min(state.activeSegIdx + 1, state.segments.length - 1);
  seekToSegment(next);
});

btnMute.addEventListener('click', () => {
  video.muted = !video.muted;
  btnMute.textContent = video.muted ? '🔇' : '🔊';
});
volSlider.addEventListener('input', () => { video.volume = volSlider.value; });

// Progress bar click/drag
progressBg.addEventListener('click', e => {
  const rect = progressBg.getBoundingClientRect();
  const pct  = (e.clientX - rect.left) / rect.width;
  video.currentTime = pct * (video.duration || state.duration);
});

// ─── Mode Buttons ────────────────────────────────────────────────────────────
function setMode(m) {
  state.mode = m;
  btnContent.classList.toggle('active', m === 'content-only');
  btnSkipNc.classList.toggle('active', m === 'noncontent-only');
  btnNormal.classList.toggle('active', m === 'normal');
  const labels = {
    'normal': 'Normal',
    'content-only': 'Content Only',
    'noncontent-only': 'Non-Content Only'
  };
  toast(`Mode: ${labels[m]}`);
}


btnContent.addEventListener('click', () => {
  setMode(state.mode === 'content-only' ? 'normal' : 'content-only');
  if (state.mode === 'content-only') {
    // Jump to first content segment
    const first = state.segments.findIndex(s => !isNonContent(s));
    if (first >= 0) seekToSegment(first);
    video.play();
  }
});

btnSkipNc.addEventListener('click', () => {
  setMode(state.mode === 'noncontent-only' ? 'normal' : 'noncontent-only');
  if (state.mode === 'noncontent-only') {
    // Jump to first ad segment
    const first = state.segments.findIndex(s => isNonContent(s));
    if (first >= 0) seekToSegment(first);
    video.play();
  }
});


btnNormal.addEventListener('click', () => setMode('normal'));

// ─── Keyboard shortcuts ───────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.code) {
    case 'Space': e.preventDefault(); video.paused ? video.play() : video.pause(); break;
    case 'ArrowRight': video.currentTime += 5; break;
    case 'ArrowLeft':  video.currentTime -= 5; break;
    case 'KeyN': btnNext.click(); break;
    case 'KeyP': btnPrev.click(); break;
  }
});

// ─── Toasts ───────────────────────────────────────────────────────────────────
function toast(msg, ms = 2800) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

// Init
toast('Load a video and JSON to begin ✨');
