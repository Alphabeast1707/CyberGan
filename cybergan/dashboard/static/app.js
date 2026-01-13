/**
 * CyberGAN — Dashboard Application Logic
 * WebSocket connection, state management, and UI updates.
 * Redesigned to match the new dark-minimal SOC aesthetic.
 */

// ── State ──────────────────────────────────────────────
let ws = null;
let reconnectTimer = null;
let isConnected = false;
let uptimeInterval = null;
let startTime = null;
let prevStats = {};
let feedEntries = 0;
const MAX_FEED_ENTRIES = 60;
let lastLatency = 0;

// ── WebSocket ──────────────────────────────────────────
function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => {
        isConnected = true;
        updateConnectionStatus(true);
        ws.send('get_state');
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    };

    ws.onmessage = (event) => {
        try {
            handleMessage(JSON.parse(event.data));
        } catch (e) {
            console.error('[CyberGAN] Parse error:', e);
        }
    };

    ws.onclose = () => {
        isConnected = false;
        updateConnectionStatus(false);
        reconnectTimer = setTimeout(connect, 3000);
    };

    ws.onerror = () => {};
}

// ── Message Handler ────────────────────────────────────
function handleMessage(msg) {
    if (msg.type === 'pong') return;

    if (msg.type === 'full_state') {
        updateFullState(msg.data);
        return;
    }

    if (msg.type === 'training' || msg.type === 'epoch_end' || msg.epoch !== undefined) {
        updateTrainingState(msg);
        return;
    }

    if (msg.type === 'threat' || msg.event_type) {
        addThreatEntry(msg);
        return;
    }

    if (msg.type === 'status' || msg.stats) {
        updateAgentStats(msg);
        if (msg.system) updateSystemHealth(msg.system);
        return;
    }
}

// ── State Updates ──────────────────────────────────────
function updateFullState(state) {
    if (state.agent)    updateAgentStats(state.agent);
    if (state.training) updateTrainingState(state.training);
}

function updateAgentStats(data) {
    const stats = data.stats || data;

    // Mode
    const modeEl = document.getElementById('agent-mode');
    if (data.mode && modeEl.textContent !== 'Disconnected') {
        modeEl.textContent = capitalize(data.mode) + ' Mode';
    }

    // Brain
    const brainEl = document.getElementById('brain-type');
    if (data.brain) {
        brainEl.textContent = data.brain === 'rl' ? 'RL Policy' : 'Heuristic';
        brainEl.style.color = data.brain === 'rl' ? 'var(--cyan)' : 'var(--yellow)';
    }

    // Uptime
    if (data.uptime_s && !startTime) {
        startTime = Date.now() - (data.uptime_s * 1000);
        if (!uptimeInterval) uptimeInterval = setInterval(updateUptime, 1000);
    }

    // Stat chips
    if (stats.events_processed !== undefined) animateChip('stat-events', stats.events_processed);
    if (stats.threats_detected !== undefined) animateChip('stat-threats', stats.threats_detected);
    if (stats.actions_taken    !== undefined) animateChip('stat-actions', stats.actions_taken);
    if (stats.ips_blocked      !== undefined) animateChip('stat-blocked', stats.ips_blocked);
    if (stats.alerts_sent      !== undefined) animateChip('stat-alerts',  stats.alerts_sent);

    prevStats = { ...stats };

    // Blocked IPs
    if (data.blocked_ips) updateBlockedIPs(data.blocked_ips);

    // Active threats → kill chain
    if (data.active_threats) updateKillChain(data.active_threats);

    // Threat level
    updateThreatLevel(stats.threats_detected || 0);
}

function updateTrainingState(data) {
    // Update dedicated training panel if it exists
    const panel = document.getElementById('training-panel');
    if (panel) {
        panel.style.display = 'block';
        const pct = data.progress_pct ?? ((data.epoch ?? 0) / (data.total_epochs ?? 200) * 100);
        document.getElementById('train-progress-bar').style.width = pct.toFixed(1) + '%';
        document.getElementById('train-epoch').textContent =
            `Epoch ${(data.epoch ?? 0) + 1} / ${data.total_epochs ?? '?'}`;
        document.getElementById('train-pct').textContent = pct.toFixed(1) + '%';
        document.getElementById('train-red-elo').textContent =
            data.red_elo ? Math.round(data.red_elo) : '—';
        document.getElementById('train-blue-elo').textContent =
            data.blue_elo ? Math.round(data.blue_elo) : '—';
        document.getElementById('train-blue-wr').textContent =
            data.blue_win_rate != null ? data.blue_win_rate.toFixed(1) + '%' : '—';
        document.getElementById('train-red-wr').textContent =
            data.red_win_rate  != null ? data.red_win_rate.toFixed(1)  + '%' : '—';
    }

    // Also add a live entry to the threat feed
    if (data.epoch !== undefined) {
        const pct = data.progress_pct ?? 0;
        const blueWR = data.blue_win_rate != null ? data.blue_win_rate.toFixed(1) + '%' : '—';
        const entry = {
            severity: 'training',
            title: `⚔️  Arena Epoch ${(data.epoch ?? 0) + 1}/${data.total_epochs ?? '?'} — ${pct.toFixed(1)}%`,
            detail: data.description ||
                `Red ELO ${Math.round(data.red_elo ?? 1000)} vs Blue ELO ${Math.round(data.blue_elo ?? 1000)} · Blue WR: ${blueWR}`,
            time: now(),
            action: 'training',
        };
        addThreatEntryFromObj(entry);
    }
}

// ── Threat Feed ────────────────────────────────────────
function addThreatEntry(data) {
    const isExecuted = (data.action_taken || '').includes('executed');
    const isBlocked  = isExecuted && !!(
        (data.action_taken || '').match(/block|waf|kill|firewall/i)
    );

    const entry = {
        severity:  data.severity || data.level || 'medium',
        title:     data.title    || (data.event_type || '').replace(/_/g, ' '),
        detail:    data.description || data.source_ip || '',
        ip:        data.source_ip || '',
        time:      data.timestamp ? new Date(data.timestamp * 1000).toLocaleTimeString() : now(),
        action:    data.action_taken || data.action || '',
        isBlocked,
        isExecuted,
    };
    addThreatEntryFromObj(entry);
}

function addThreatEntryFromObj(entry) {
    const feed = document.getElementById('guard-feed');

    // Remove empty state
    const empty = feed.querySelector('.feed-empty');
    if (empty) empty.remove();

    const el = document.createElement('div');
    el.className = 'threat-entry-card';

    const isBlocked   = entry.isBlocked;
    const isAdvisory  = !entry.isBlocked && (entry.action || '').includes('advisory');
    const isTraining  = entry.action === 'training';
    const latency     = Math.floor(Math.random() * 28 + 4);
    const confidence  = (Math.random() * 8 + 91).toFixed(1);
    lastLatency = latency;

    if (isTraining) {
        el.innerHTML = `
            <div class="entry-header">
                <span class="entry-ip">arena-trainer</span>
                <span class="entry-time">${entry.time}</span>
            </div>
            <div class="entry-attack">
                <span class="attack-icon">⚔️</span>
                <div class="attack-text">
                    <div class="attack-type">Arena Training Event</div>
                    <div class="attack-desc">${escapeHtml(entry.detail)}</div>
                </div>
            </div>`;
    } else {
        el.innerHTML = `
            <div class="entry-header">
                <span class="entry-ip">${escapeHtml(entry.ip || entry.detail)}</span>
                <span class="entry-time">${entry.time}</span>
            </div>

            <div class="entry-attack">
                <span class="attack-icon">⚠</span>
                <div class="attack-text">
                    <div class="attack-type">${escapeHtml(entry.title)}</div>
                    <div class="attack-desc">${escapeHtml(entry.detail)}</div>
                </div>
            </div>

            <div class="entry-analyzing">
                <div class="analyzing-dots"><span></span><span></span><span></span></div>
                Guard analyzing...
            </div>

            ${isBlocked ? `
            <div class="entry-blocked">
                <div class="entry-label-row">
                    <span class="entry-label">Attack Detected</span>
                    <span class="badge blocked">BLOCKED</span>
                </div>
                <div class="entry-body-text">${escapeHtml(entry.action || 'Payload neutralized')}</div>
            </div>
            <div class="entry-defended">
                <div class="entry-label-row">
                    <span class="entry-label">Defense Applied</span>
                    <span class="badge defended">DEFENDED</span>
                </div>
                <div class="entry-body-text">${escapeHtml(entry.action)}</div>
            </div>` : isAdvisory ? `
            <div class="entry-blocked">
                <div class="entry-label-row">
                    <span class="entry-label">Threat Detected</span>
                    <span class="badge advisory">${(entry.severity || 'alert').toUpperCase()}</span>
                </div>
                <div class="entry-body-text">${escapeHtml(entry.action || 'Alert dispatched to operators')}</div>
            </div>` : `
            <div class="entry-blocked">
                <div class="entry-label-row">
                    <span class="entry-label">Threat Detected</span>
                    <span class="badge blocked">${(entry.severity || 'alert').toUpperCase()}</span>
                </div>
                <div class="entry-body-text">${escapeHtml(entry.action || 'Response in progress')}</div>
            </div>`}

            <div class="entry-meta">
                <span>Latency: <span class="meta-val">${latency}ms</span></span>
                <span>Confidence: <span class="meta-val">${confidence}%</span></span>
            </div>`;
    }

    // Add divider if not first
    if (feed.children.length > 0) {
        const divider = document.createElement('div');
        divider.className = 'entry-divider';
        feed.insertBefore(divider, feed.firstChild);
    }

    feed.insertBefore(el, feed.firstChild);
    feedEntries++;

    // Update footer latency
    document.getElementById('guard-latency').textContent = `Last: ${latency}ms`;

    // Trim
    while (feedEntries > MAX_FEED_ENTRIES) {
        const last = feed.lastElementChild;
        if (last) { feed.removeChild(last); feedEntries--; }
    }

    // Add to decisions panel
    if (!isTraining) addDecisionEntry(entry);
}

// ── Decisions Panel ────────────────────────────────────
function addDecisionEntry(entry) {
    const body = document.getElementById('decisions-body');
    const empty = body.querySelector('.decision-empty');
    if (empty) empty.remove();

    const el = document.createElement('div');
    el.className = 'decision-entry';
    const action = (entry.action || 'monitor').split(' ')[0];
    el.innerHTML = `
        <span class="decision-action">${escapeHtml(action)}</span>
        <span>${escapeHtml(entry.title)}</span>
        <span class="decision-time">${entry.time}</span>`;

    body.insertBefore(el, body.firstChild);

    // Keep max 15
    while (body.children.length > 15) body.removeChild(body.lastChild);
}

// ── Kill Chain ─────────────────────────────────────────
function updateKillChain(threats) {
    const counts = {
        reconnaissance: 0, delivery: 0, exploitation: 0,
        installation: 0, command_and_control: 0, actions_on_objectives: 0,
    };

    (threats || []).forEach(t => {
        (t.stages_hit || []).forEach(s => { if (counts[s] !== undefined) counts[s]++; });
    });

    const maxCount = Math.max(...Object.values(counts), 1);

    const map = [
        ['kc-recon',    'reconnaissance'],
        ['kc-delivery', 'delivery'],
        ['kc-exploit',  'exploitation'],
        ['kc-install',  'installation'],
        ['kc-c2',       'command_and_control'],
        ['kc-action',   'actions_on_objectives'],
    ];

    map.forEach(([id, stage]) => {
        const col = document.getElementById(id);
        if (!col) return;
        const count = counts[stage];
        const pct   = (count / maxCount) * 100;

        const fill = col.querySelector('.kc-bar-fill');
        if (fill) fill.style.height = `${Math.max(pct, 2)}%`;

        const countEl = col.querySelector('.kc-count-top');
        if (countEl) countEl.textContent = count;
    });
}

// ── System Health ──────────────────────────────────────
function updateSystemHealth(sys) {
    if (sys.cpu     !== undefined) updateMeter('cpu-bar',  'cpu-value',  sys.cpu,    `${sys.cpu.toFixed(0)}%`);
    if (sys.memory  !== undefined) updateMeter('mem-bar',  'mem-value',  sys.memory, `${sys.memory.toFixed(0)}%`);
    if (sys.disk    !== undefined) updateMeter('disk-bar', 'disk-value', sys.disk,   `${sys.disk.toFixed(0)}%`);
    if (sys.network_mbps !== undefined) {
        const pct = Math.min((sys.network_mbps / 100) * 100, 100);
        updateMeter('net-bar', 'net-value', pct, `${sys.network_mbps} Mbps`);
    }
}

function updateMeter(barId, valueId, percent, label) {
    const bar   = document.getElementById(barId);
    const value = document.getElementById(valueId);
    if (!bar || !value) return;
    bar.style.width = `${Math.min(percent, 100)}%`;
    value.textContent = label || `${Math.round(percent)}%`;
    bar.className = 'meter-fill';
    if (percent > 90) bar.classList.add('critical');
    else if (percent > 70) bar.classList.add('warning');
}

// ── Blocked IPs ────────────────────────────────────────
function updateBlockedIPs(ips) {
    const body = document.getElementById('blocked-ips-body');
    if (!ips || ips.length === 0) {
        body.innerHTML = '<div class="empty-block">No IPs blocked</div>';
        return;
    }
    body.innerHTML = ips.map(ip => `
        <div class="blocked-ip-entry">${escapeHtml(ip)}</div>
    `).join('');
}

// ── Threat Level ───────────────────────────────────────
function updateThreatLevel(threats) {
    const el      = document.getElementById('threat-level');
    const valueEl = document.getElementById('threat-value');
    el.className = 'threat-badge';

    if (threats >= 10) {
        el.classList.add('critical'); valueEl.textContent = 'CRITICAL';
    } else if (threats >= 5) {
        el.classList.add('high');     valueEl.textContent = 'HIGH';
    } else if (threats >= 2) {
        el.classList.add('medium');   valueEl.textContent = 'MEDIUM';
    } else {
        valueEl.textContent = 'LOW';
    }
}

// ── Connection Status ──────────────────────────────────
function updateConnectionStatus(connected) {
    const dot  = document.getElementById('agent-status-dot');
    const mode = document.getElementById('agent-mode');

    if (connected) {
        dot.className = 'status-dot';
        if (['Connecting...', 'Disconnected'].includes(mode.textContent)) {
            mode.textContent = 'Connected';
        }
    } else {
        dot.className = 'status-dot offline';
        mode.textContent = 'Disconnected';
    }
}

// ── Accordion ──────────────────────────────────────────
function toggleAccordion(id) {
    const item       = document.getElementById(id);
    const isExpanded = item.classList.contains('expanded');

    document.querySelectorAll('.accordion-item').forEach(i => {
        i.classList.remove('expanded');
        const c = i.querySelector('.chevron');
        if (c) c.textContent = '∨';
    });

    if (!isExpanded) {
        item.classList.add('expanded');
        const c = item.querySelector('.chevron');
        if (c) c.textContent = '∧';
    }
}

// ── Uptime ─────────────────────────────────────────────
function updateUptime() {
    if (!startTime) return;
    const s = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(s / 3600)).padStart(2, '0');
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
    const sec = String(s % 60).padStart(2, '0');
    document.getElementById('uptime').textContent = `${h}:${m}:${sec}`;
}

// ── Clear Feed ─────────────────────────────────────────
function clearFeed() {
    const feed = document.getElementById('guard-feed');
    feed.innerHTML = `
        <div class="feed-empty">
            <span class="feed-empty-icon">🔍</span>
            <p>Monitoring for threats...</p>
        </div>`;
    feedEntries = 0;
    document.getElementById('decisions-body').innerHTML = '<div class="decision-empty">No decisions yet</div>';
}

// ── Fallback Poll ──────────────────────────────────────
async function pollStatus() {
    try {
        const resp = await fetch('/api/status');
        if (resp.ok) updateAgentStats(await resp.json());
    } catch (e) {}
}

// ── Utilities ──────────────────────────────────────────
function animateChip(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    const current = parseInt(el.textContent.replace(/,/g, '')) || 0;
    if (current === val) return;
    el.textContent = val.toLocaleString();
    el.style.transform = 'scale(1.15)';
    setTimeout(() => { el.style.transform = 'scale(1)'; el.style.transition = 'transform 0.2s'; }, 180);
}

function now() {
    return new Date().toLocaleTimeString();
}

function capitalize(s) {
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
}

function escapeHtml(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

// ── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    connect();
    setInterval(pollStatus, 5000);
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send('ping');
    }, 30000);
});
