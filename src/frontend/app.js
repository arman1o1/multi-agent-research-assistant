/**
 * Multi-Agent Research Assistant — Client Application
 * Handles SSE streaming, WebSocket approval, pipeline visualization, and report rendering.
 */

(function () {
  'use strict';

  // ── Constants ──────────────────────────────────────────────────────────
  const AGENTS = ['planner', 'researcher', 'critic', 'fact_checker', 'writer'];
  const AGENT_LABELS = {
    planner: 'Planner',
    researcher: 'Researcher',
    critic: 'Critic',
    fact_checker: 'Fact-Check',
    writer: 'Writer',
    system: 'System',
  };

  const SSE_RECONNECT_DELAY = 3000;
  const SSE_MAX_RETRIES = 10;
  const ERROR_TOAST_DURATION = 5000;

  // ── DOM References ─────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const dom = {
    topicInput:      $('topicInput'),
    modeAuto:        $('modeAuto'),
    modeApproval:    $('modeApproval'),
    btnSubmit:       $('btnSubmit'),
    btnCancel:       $('btnCancel'),
    pipelineTrack:   $('pipelineTrack'),
    liveDot:         $('liveDot'),
    activityFeed:    $('activityFeed'),
    approvalPanel:   $('approvalPanel'),
    approvalText:    $('approvalText'),
    btnApprove:      $('btnApprove'),
    btnReject:       $('btnReject'),
    reportPanel:     $('reportPanel'),
    reportRendered:  $('reportRendered'),
    btnDownloadMd:   $('btnDownloadMd'),
    btnDownloadPdf:  $('btnDownloadPdf'),
    btnDownloadHtml: $('btnDownloadHtml'),
    errorToast:      $('errorToast'),
    btnThemeToggle:  $('btnThemeToggle'),
    highlightTheme:  $('highlightTheme'),
    timerContainer:  $('timerContainer'),
    sessionTimer:    $('sessionTimer'),
    progressBarContainer: $('progressBarContainer'),
    progressBarFill: $('progressBarFill'),
  };

  // ── State ──────────────────────────────────────────────────────────────
  const state = {
    sessionId: null,
    mode: 'auto',           // 'auto' | 'approval'
    pipelineStatus: 'idle', // 'idle' | 'running' | 'complete' | 'error'
    activeAgent: null,
    completedAgents: new Set(),
    sseRetries: 0,
    timerInterval: null,
    elapsedSeconds: 0,
    wsRetries: 0,
  };

  let eventSource = null;
  let ws = null;
  let errorToastTimer = null;

  // ── Configure Marked.js ────────────────────────────────────────────────
  marked.setOptions({
    highlight: function (code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try { return hljs.highlight(code, { language: lang }).value; } catch (_) {}
      }
      return hljs.highlightAuto(code).value;
    },
    breaks: false,
    gfm: true,
  });

  // ── Initialization ─────────────────────────────────────────────────────
  function init() {
    bindEvents();
    resetPipeline();
    initTheme();
  }

  function bindEvents() {
    // Mode toggle
    dom.modeAuto.addEventListener('click', () => setMode('auto'));
    dom.modeApproval.addEventListener('click', () => setMode('approval'));

    // Submit
    dom.btnSubmit.addEventListener('click', handleSubmit);

    // Cancel
    if (dom.btnCancel) {
      dom.btnCancel.addEventListener('click', handleCancel);
    }

    // Theme toggle
    if (dom.btnThemeToggle) {
      dom.btnThemeToggle.addEventListener('click', toggleTheme);
    }

    // Approval buttons
    dom.btnApprove.addEventListener('click', () => handleApproval(true));
    dom.btnReject.addEventListener('click', () => handleApproval(false));

    // Download buttons
    dom.btnDownloadMd.addEventListener('click', () => downloadReport('md'));
    dom.btnDownloadPdf.addEventListener('click', () => downloadReport('pdf'));
    dom.btnDownloadHtml.addEventListener('click', () => downloadReport('html'));

    // Keyboard: Ctrl+Enter or Enter (when not typing newline) to submit
    dom.topicInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleSubmit();
      }
    });
  }

  // ── Theme Management ───────────────────────────────────────────────────
  function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;

    if (savedTheme === 'light' || (!savedTheme && systemPrefersLight)) {
      setTheme('light');
    } else {
      setTheme('dark');
    }
  }

  function setTheme(theme) {
    if (theme === 'light') {
      document.body.classList.add('light-theme');
      if (dom.highlightTheme) {
        dom.highlightTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';
      }
      const themeIcon = dom.btnThemeToggle ? dom.btnThemeToggle.querySelector('.theme-icon') : null;
      if (themeIcon) {
        themeIcon.textContent = '☀️';
      }
      localStorage.setItem('theme', 'light');
    } else {
      document.body.classList.remove('light-theme');
      if (dom.highlightTheme) {
        dom.highlightTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark-dimmed.min.css';
      }
      const themeIcon = dom.btnThemeToggle ? dom.btnThemeToggle.querySelector('.theme-icon') : null;
      if (themeIcon) {
        themeIcon.textContent = '🌙';
      }
      localStorage.setItem('theme', 'dark');
    }
  }

  function toggleTheme() {
    const isLight = document.body.classList.contains('light-theme');
    setTheme(isLight ? 'dark' : 'light');
  }

  // ── Mode Toggle ────────────────────────────────────────────────────────
  function setMode(mode) {
    state.mode = mode;
    dom.modeAuto.classList.toggle('active', mode === 'auto');
    dom.modeApproval.classList.toggle('active', mode === 'approval');
    dom.modeAuto.setAttribute('aria-checked', mode === 'auto');
    dom.modeApproval.setAttribute('aria-checked', mode === 'approval');
  }

  // ── Submit Handler ─────────────────────────────────────────────────────
  async function handleSubmit() {
    const topic = dom.topicInput.value.trim();
    if (!topic) {
      showError('Please enter a research topic.');
      dom.topicInput.focus();
      return;
    }

    if (state.pipelineStatus === 'running') return;

    state.pipelineStatus = 'running';
    updateButtonStates();
    resetPipeline();
    hideApprovalPanel();
    hideReportPanel();
    clearActivityFeed();

    try {
      const data = await startResearch(topic, state.mode);
      state.sessionId = data.session_id;
      dom.liveDot.classList.add('visible');
      addActivityItem('system', `Research started: session ${data.session_id.slice(0, 8)}...`, new Date().toISOString());

      startTimer();
      connectSSE(data.session_id);

      if (state.mode === 'approval') {
        connectWebSocket(data.session_id);
      }
    } catch (err) {
      showError(err.message || 'Failed to start research.');
      state.pipelineStatus = 'error';
      updateButtonStates();
    }
  }

  // ── Cancel Handler ─────────────────────────────────────────────────────
  async function handleCancel() {
    if (!state.sessionId) return;

    // Disconnect immediately so no further server events are processed
    disconnectSSE();
    disconnectWebSocket();
    state.pipelineStatus = 'error';
    stopTimer();
    updateButtonStates();
    dom.liveDot.classList.remove('visible');

    addActivityItem('system', 'Cancelling research...', new Date().toISOString());
    try {
      const res = await fetch(`/api/research/${state.sessionId}/cancel`, {
        method: 'POST'
      });
      if (!res.ok) {
        throw new Error('Failed to cancel research');
      }
    } catch (err) {
      showError(err.message || 'Failed to cancel research.');
    }
  }

  // ── API: Start Research ────────────────────────────────────────────────
  async function startResearch(topic, mode) {
    const res = await fetch('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, mode }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Server error (${res.status})`);
    }

    return res.json();
  }

  // ── SSE Connection ─────────────────────────────────────────────────────
  function connectSSE(sessionId) {
    disconnectSSE();
    state.sseRetries = 0;

    const url = `/api/research/${sessionId}/stream`;
    eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleSSEEvent(data);
      } catch (err) {
        console.warn('SSE parse error:', err);
      }
    };

    eventSource.onerror = () => {
      if (state.pipelineStatus === 'complete') {
        disconnectSSE();
        return;
      }

      state.sseRetries++;
      if (state.sseRetries > SSE_MAX_RETRIES) {
        disconnectSSE();
        showError('Lost connection to server. Please try again.');
        state.pipelineStatus = 'error';
        updateButtonStates();
        dom.liveDot.classList.remove('visible');
        return;
      }

      // Auto-reconnect
      disconnectSSE();
      setTimeout(() => {
        if (state.pipelineStatus === 'running') {
          connectSSE(sessionId);
        }
      }, SSE_RECONNECT_DELAY);
    };

    eventSource.onopen = () => {
      state.sseRetries = 0;
    };
  }

  function disconnectSSE() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }

  // ── SSE Event Handler ──────────────────────────────────────────────────
  function handleSSEEvent(data) {
    switch (data.type) {
      case 'agent_event':
        handleAgentEvent(data);
        break;

      case 'pipeline_status':
        handlePipelineStatus(data);
        break;

      default:
        console.log('Unknown SSE event type:', data.type);
    }
  }

  function handleAgentEvent(data) {
    const { agent, status, content, timestamp } = data;
    const agentKey = normalizeAgentKey(agent);

    if (status === 'running') {
      updatePipelineViz(agentKey, 'active');
    } else if (status === 'complete') {
      updatePipelineViz(agentKey, 'complete');
    }

    if (content) {
      addActivityItem(agentKey, content, timestamp);
    }
  }

  function handlePipelineStatus(data) {
    const { status, stage, data: payload } = data;

    if (status === 'awaiting_approval') {
      showApprovalPanel(stage, payload);
      addActivityItem('system', `Pipeline paused. Awaiting approval for: ${stage}`, new Date().toISOString());
    }

    if (status === 'complete') {
      state.pipelineStatus = 'complete';
      stopTimer();
      if (dom.progressBarFill) dom.progressBarFill.style.width = '100%';
      updateButtonStates();
      dom.liveDot.classList.remove('visible');
      disconnectSSE();
      disconnectWebSocket();

      if (payload && payload.report) {
        renderReport(payload.report);
        showReportPanel();
        addActivityItem('system', 'Research complete. Report ready.', new Date().toISOString());
      } else {
        addActivityItem('system', 'Research pipeline complete.', new Date().toISOString());
        // Try fetching the report
        fetchAndShowReport();
      }
    }

    if (status === 'error') {
      state.pipelineStatus = 'error';
      stopTimer();
      updateButtonStates();
      dom.liveDot.classList.remove('visible');
      disconnectSSE();
      const msg = (payload && payload.message) || 'Pipeline encountered an error.';
      showError(msg);
      addActivityItem('system', `Error: ${msg}`, new Date().toISOString());
    }
  }

  // ── Pipeline Visualization ─────────────────────────────────────────────
  function updatePipelineViz(agentKey, status) {
    const node = $(`node-${agentKey}`);
    if (!node) return;

    updateProgressBarBasedOnAgent(agentKey, status);

    if (status === 'active') {
      // Deactivate previous active agent
      if (state.activeAgent && state.activeAgent !== agentKey) {
        const prevNode = $(`node-${state.activeAgent}`);
        if (prevNode) {
          prevNode.classList.remove('active');
          if (!state.completedAgents.has(state.activeAgent)) {
            prevNode.classList.add('complete');
            state.completedAgents.add(state.activeAgent);
          }
        }
      }

      node.classList.remove('complete');
      node.classList.add('active');
      state.activeAgent = agentKey;
      updateConnectors();

    } else if (status === 'complete') {
      node.classList.remove('active');
      node.classList.add('complete');
      state.completedAgents.add(agentKey);

      if (state.activeAgent === agentKey) {
        state.activeAgent = null;
      }
      updateConnectors();
    }
  }

  function updateConnectors() {
    // Light up connectors between completed agents
    for (let i = 0; i < AGENTS.length - 1; i++) {
      const conn = $(`conn-${i}`);
      if (!conn) continue;

      const isLit = state.completedAgents.has(AGENTS[i]);
      conn.classList.toggle('lit', isLit);
    }
  }

  function resetPipeline() {
    state.activeAgent = null;
    state.completedAgents.clear();

    AGENTS.forEach((agent) => {
      const node = $(`node-${agent}`);
      if (node) {
        node.classList.remove('active', 'complete');
      }
    });

    for (let i = 0; i < AGENTS.length - 1; i++) {
      const conn = $(`conn-${i}`);
      if (conn) conn.classList.remove('lit');
    }

    stopTimer();
    state.elapsedSeconds = 0;
    if (dom.sessionTimer) dom.sessionTimer.textContent = '00:00';
    if (dom.timerContainer) dom.timerContainer.style.display = 'none';
    if (dom.progressBarFill) dom.progressBarFill.style.width = '0%';
    if (dom.progressBarContainer) dom.progressBarContainer.style.display = 'none';
  }

  // ── Activity Feed ──────────────────────────────────────────────────────
  function addActivityItem(agent, content, timestamp) {
    const item = document.createElement('div');
    item.className = 'activity-item';
    item.setAttribute('data-agent', agent);

    const badge = document.createElement('span');
    badge.className = `agent-badge ${agent}`;
    badge.textContent = AGENT_LABELS[agent] || agent;

    const contentWrap = document.createElement('div');
    contentWrap.className = 'activity-content';

    const text = document.createElement('div');
    text.className = 'activity-text';
    text.innerHTML = DOMPurify.sanitize(marked.parse(content));

    // Apply syntax highlighting to code blocks in the activity feed
    text.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block);
    });

    const ts = document.createElement('div');
    ts.className = 'activity-timestamp';
    ts.textContent = formatTimestamp(timestamp);

    contentWrap.appendChild(text);
    contentWrap.appendChild(ts);
    item.appendChild(badge);
    item.appendChild(contentWrap);

    dom.activityFeed.appendChild(item);

    // Auto-scroll to bottom
    requestAnimationFrame(() => {
      dom.activityFeed.scrollTop = dom.activityFeed.scrollHeight;
    });
  }

  function clearActivityFeed() {
    dom.activityFeed.innerHTML = '';
  }

  function formatTimestamp(iso) {
    if (!iso) return '';
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return iso;
    }
  }

  // ── Approval Panel ─────────────────────────────────────────────────────
  function showApprovalPanel(stage, data) {
    let rawText = '';
    if (typeof data === 'string') {
      rawText = data;
    } else if (data && data.plan) {
      rawText = data.plan;
    } else if (data) {
      rawText = JSON.stringify(data, null, 2);
    }

    if (!rawText) {
      rawText = `Stage: ${stage}\n\nPlease review and approve or reject.`;
    }

    dom.approvalText.innerHTML = DOMPurify.sanitize(marked.parse(rawText));
    
    // Apply syntax highlighting
    dom.approvalText.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block);
    });

    dom.approvalPanel.classList.add('visible');
    dom.approvalPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function hideApprovalPanel() {
    dom.approvalPanel.classList.remove('visible');
  }

  // ── WebSocket (Approval Mode) ──────────────────────────────────────────
  function connectWebSocket(sessionId) {
    disconnectWebSocket();
    state.wsRetries = 0;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/${sessionId}`;

    ws = new WebSocket(url);

    ws.onopen = () => {
      console.log('WebSocket connected');
      state.wsRetries = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleSSEEvent(data); // reuse the same handler
      } catch (err) {
        console.warn('WS parse error:', err);
      }
    };

    ws.onerror = (err) => {
      console.warn('WebSocket error:', err);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      ws = null;

      // Auto-reconnect if research is still running
      if (state.pipelineStatus === 'running' && state.mode === 'approval') {
        state.wsRetries++;
        if (state.wsRetries <= 10) {
          console.log(`Reconnecting WebSocket (attempt ${state.wsRetries})...`);
          setTimeout(() => {
            if (state.pipelineStatus === 'running') {
              connectWebSocket(sessionId);
            }
          }, 3000);
        }
      }
    };
  }

  function disconnectWebSocket() {
    if (ws) {
      ws.close();
      ws = null;
    }
  }

  function handleApproval(approved) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'approval', approved }));
    } else {
      // Fallback: try HTTP
      if (state.sessionId) {
        fetch(`/api/research/${state.sessionId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ approved }),
        }).catch((err) => {
          showError('Failed to send approval.');
          console.error(err);
        });
      }
    }

    hideApprovalPanel();
    addActivityItem('system', approved ? 'Approved. Resuming pipeline.' : 'Rejected. Revising.', new Date().toISOString());
  }

  // ── Report Panel ───────────────────────────────────────────────────────
  function renderReport(markdown) {
    dom.reportRendered.innerHTML = DOMPurify.sanitize(marked.parse(markdown));

    // Apply syntax highlighting to code blocks
    dom.reportRendered.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block);
    });
  }

  function showReportPanel() {
    dom.reportPanel.classList.add('visible');
    dom.reportPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function hideReportPanel() {
    dom.reportPanel.classList.remove('visible');
  }

  async function fetchAndShowReport() {
    if (!state.sessionId) return;
    try {
      const res = await fetch(`/api/research/${state.sessionId}/report`);
      if (res.ok) {
        const md = await res.text();
        renderReport(md);
        showReportPanel();
      }
    } catch (err) {
      console.warn('Could not fetch report:', err);
    }
  }

  // ── Download ───────────────────────────────────────────────────────────
  function downloadReport(format) {
    if (!state.sessionId) {
      showError('No active research session.');
      return;
    }

    let url, filename;
    if (format === 'pdf') {
      url = `/api/research/${state.sessionId}/report/pdf`;
      filename = 'report.pdf';
    } else if (format === 'html') {
      url = `/api/research/${state.sessionId}/report/html`;
      filename = 'report.html';
    } else {
      url = `/api/research/${state.sessionId}/report`;
      filename = 'report.md';
    }

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // ── UI Helpers ─────────────────────────────────────────────────────────
  function updateButtonStates() {
    const isRunning = state.pipelineStatus === 'running';
    dom.btnSubmit.disabled = isRunning;
    dom.btnSubmit.classList.toggle('loading', isRunning);
    const btnText = dom.btnSubmit.querySelector('.btn-text');
    if (btnText) {
      btnText.textContent = isRunning ? 'Researching...' : 'Start Research';
    }

    if (dom.btnCancel) {
      if (isRunning) {
        dom.btnCancel.classList.add('visible');
      } else {
        dom.btnCancel.classList.remove('visible');
      }
    }
  }

  function showError(message) {
    dom.errorToast.textContent = message;
    dom.errorToast.classList.add('visible');

    clearTimeout(errorToastTimer);
    errorToastTimer = setTimeout(() => {
      dom.errorToast.classList.remove('visible');
    }, ERROR_TOAST_DURATION);
  }

  function normalizeAgentKey(raw) {
    if (!raw) return 'system';
    const key = raw.toLowerCase().replace(/[\s-]+/g, '_');
    // Map common variants
    if (key.includes('plan')) return 'planner';
    if (key.includes('research')) return 'researcher';
    if (key.includes('critic') || key.includes('review')) return 'critic';
    if (key.includes('fact') || key.includes('check') || key.includes('verify')) return 'fact_checker';
    if (key.includes('writ') || key.includes('author')) return 'writer';
    return AGENTS.includes(key) ? key : 'system';
  }

  // Helper functions for Timer & Progress Bar
  function startTimer() {
    stopTimer();
    state.elapsedSeconds = 0;
    if (dom.timerContainer) dom.timerContainer.style.display = 'inline-flex';
    if (dom.progressBarContainer) dom.progressBarContainer.style.display = 'block';
    
    state.timerInterval = setInterval(() => {
      state.elapsedSeconds++;
      const minutes = String(Math.floor(state.elapsedSeconds / 60)).padStart(2, '0');
      const seconds = String(state.elapsedSeconds % 60).padStart(2, '0');
      if (dom.sessionTimer) {
        dom.sessionTimer.textContent = `${minutes}:${seconds}`;
      }
    }, 1000);
  }

  function stopTimer() {
    if (state.timerInterval) {
      clearInterval(state.timerInterval);
      state.timerInterval = null;
    }
  }

  function updateProgressBarBasedOnAgent(agentKey, status) {
    if (!dom.progressBarFill) return;
    let percent = 0;
    
    if (agentKey === 'planner') {
      percent = status === 'active' ? 10 : 20;
    } else if (agentKey === 'researcher') {
      percent = status === 'active' ? 30 : 45;
    } else if (agentKey === 'critic') {
      percent = status === 'active' ? 50 : 65;
    } else if (agentKey === 'fact_checker') {
      percent = status === 'active' ? 75 : 85;
    } else if (agentKey === 'writer') {
      percent = status === 'active' ? 90 : 98;
    }
    
    dom.progressBarFill.style.width = `${percent}%`;
  }

  // ── Bootstrap ──────────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
