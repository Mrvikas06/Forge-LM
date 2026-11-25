/**
 * ForgeLM Frontend Studio v2.0
 * Next-generation interactive client for causal language model inference.
 */

const STORY_PRESETS = {
  story: "Once upon a time in a sunny meadow, a playful puppy named Barnaby",
  car: "Once upon a time, there was a little car named Beep who loved to go fast",
  autumn: "Trees lose their leaves in autumn to prepare for winter because",
  friendship: "A good friend listens when you speak, shares toys, and"
};

const SAMPLING_PROFILES = {
  storyteller: {
    temperature: 0.70,
    top_k: 40,
    top_p: 0.90,
    repetition_penalty: 1.15,
    max_tokens: 160
  },
  precise: {
    temperature: 0.15,
    top_k: 20,
    top_p: 0.80,
    repetition_penalty: 1.05,
    max_tokens: 140
  },
  creative: {
    temperature: 1.05,
    top_k: 70,
    top_p: 0.95,
    repetition_penalty: 1.20,
    max_tokens: 200
  }
};

let currentMode = "chat";
let isGenerating = false;
let chatMessages = [];

// DOM References
const elements = {
  tabChat: document.getElementById("tab-chat"),
  tabPlayground: document.getElementById("tab-playground"),
  viewChat: document.getElementById("view-chat"),
  viewPlayground: document.getElementById("view-playground"),
  statusDot: document.getElementById("status-dot"),
  statusLabel: document.getElementById("status-label"),
  chatThread: document.getElementById("chat-thread"),
  chatInput: document.getElementById("chat-input"),
  chatSendBtn: document.getElementById("chat-send-btn"),
  playgroundPrompt: document.getElementById("playground-prompt"),
  playgroundOutput: document.getElementById("playground-stream-output"),
  playgroundGenBtn: document.getElementById("playground-gen-btn"),
  metricTokens: document.getElementById("metric-tokens"),
  metricSpeed: document.getElementById("metric-speed"),
  telemetryDevice: document.getElementById("telemetry-device"),
  telemetryVram: document.getElementById("telemetry-vram"),
  telemetryParams: document.getElementById("telemetry-params"),
  telemetryLayers: document.getElementById("telemetry-layers"),
  backendUrl: document.getElementById("input-backend-url")
};

function getBackendBaseUrl() {
  const url = elements.backendUrl ? elements.backendUrl.value.trim() : "http://127.0.0.1:8000";
  return url.replace(/\/+$/, "");
}

function switchMode(mode) {
  currentMode = mode;
  if (mode === "chat") {
    elements.tabChat.classList.add("active");
    elements.tabPlayground.classList.remove("active");
    elements.viewChat.classList.add("active");
    elements.viewPlayground.classList.remove("active");
  } else {
    elements.tabPlayground.classList.add("active");
    elements.tabChat.classList.remove("active");
    elements.viewPlayground.classList.add("active");
    elements.viewChat.classList.remove("active");
  }
}

function updateSliderVal(name) {
  const slider = document.getElementById(`slider-${name}`);
  const display = document.getElementById(`val-${name}`);
  if (slider && display) {
    display.textContent = parseFloat(slider.value).toFixed(name === "topk" || name === "maxtokens" ? 0 : 2);
  }
}

function applySamplingProfile(profileKey) {
  const profile = SAMPLING_PROFILES[profileKey];
  if (!profile) return;

  document.getElementById("slider-temperature").value = profile.temperature;
  document.getElementById("slider-topk").value = profile.top_k;
  document.getElementById("slider-topp").value = profile.top_p;
  document.getElementById("slider-reppen").value = profile.repetition_penalty;
  document.getElementById("slider-maxtokens").value = profile.max_tokens;

  updateSliderVal("temperature");
  updateSliderVal("topk");
  updateSliderVal("topp");
  updateSliderVal("reppen");
  updateSliderVal("maxtokens");
}

function getSamplingParams() {
  return {
    temperature: parseFloat(document.getElementById("slider-temperature").value),
    top_k: parseInt(document.getElementById("slider-topk").value, 10),
    top_p: parseFloat(document.getElementById("slider-topp").value),
    repetition_penalty: parseFloat(document.getElementById("slider-reppen").value),
    max_tokens: parseInt(document.getElementById("slider-maxtokens").value, 10),
    use_cache: document.getElementById("toggle-kvcache").checked
  };
}

function applyPreset(key) {
  if (STORY_PRESETS[key]) {
    elements.playgroundPrompt.value = STORY_PRESETS[key];
    elements.playgroundOutput.textContent = "";
  }
}

function applyQuickPrompt(promptText) {
  elements.chatInput.value = promptText;
  sendChatMessage();
}

function clearPlayground() {
  elements.playgroundPrompt.value = "";
  elements.playgroundOutput.textContent = "";
  elements.metricTokens.textContent = "Tokens: 0";
  elements.metricSpeed.textContent = "Throughput: 0 tok/s";
}

function clearActiveSession() {
  chatMessages = [];
  elements.chatThread.innerHTML = `
    <div class="message assistant welcome-card">
      <div class="message-avatar">⚡</div>
      <div class="message-content">
        <div class="message-header">
          <span class="sender-name">ForgeLM</span>
          <span class="msg-time">Session Reset</span>
        </div>
        <div class="message-body">
          Session cleared! Ready for new prompts or questions.
        </div>
      </div>
    </div>
  `;
}

// Telemetry Poller
async function fetchTelemetry() {
  const base = getBackendBaseUrl();
  try {
    const res = await fetch(`${base}/health`);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    elements.statusDot.className = "status-dot online";
    elements.statusLabel.textContent = `Online (${data.device.toUpperCase()})`;

    elements.telemetryDevice.textContent = data.device.toUpperCase();
    elements.telemetryVram.textContent = data.vram_allocated_mb ? `${data.vram_allocated_mb} MB` : "N/A (CPU)";
  } catch (err) {
    elements.statusDot.className = "status-dot offline";
    elements.statusLabel.textContent = "Offline";
    elements.telemetryDevice.textContent = "Disconnected";
    elements.telemetryVram.textContent = "--";
  }
}

async function fetchModelInfo() {
  const base = getBackendBaseUrl();
  try {
    const res = await fetch(`${base}/model/info`);
    if (res.ok) {
      const data = await res.json();
      const paramsM = (data.num_parameters / 1e6).toFixed(1);
      elements.telemetryParams.textContent = `${paramsM}M Params`;
      elements.telemetryLayers.textContent = `${data.n_layers} Layers / ${data.n_heads} Heads`;
    }
  } catch (err) {
    // Ignore if offline
  }
}

// Chat Mode Handler
function handleChatKeyDown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
}

async function sendChatMessage() {
  const text = elements.chatInput.value.trim();
  if (!text || isGenerating) return;

  // Add User Message
  appendMessage("user", text);
  chatMessages.push({ role: "user", content: text });
  elements.chatInput.value = "";

  // Prepare Assistant Bubble for live streaming
  const { bodyEl, footerEl } = appendAssistantContainer();
  bodyEl.classList.add("cursor-blink");
  isGenerating = true;
  elements.chatSendBtn.disabled = true;

  const base = getBackendBaseUrl();
  const params = getSamplingParams();
  let tokenCount = 0;
  const startTime = performance.now();

  try {
    const response = await fetch(`${base}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatMessages,
        max_tokens: params.max_tokens,
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        repetition_penalty: params.repetition_penalty,
        use_cache: params.use_cache
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let assistantText = "";
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6).trim();
          try {
            const data = JSON.parse(jsonStr);
            if (data.token) {
              tokenCount++;
              assistantText += data.token;
              bodyEl.textContent = assistantText;
              elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
            }
          } catch (e) {
            // Incomplete JSON
          }
        }
      }
    }

    chatMessages.push({ role: "assistant", content: assistantText });

    const totalSeconds = ((performance.now() - startTime) / 1000).toFixed(2);
    const tokSec = tokenCount > 0 ? (tokenCount / totalSeconds).toFixed(1) : 0;
    footerEl.innerHTML = `
      <span>${tokenCount} tokens in ${totalSeconds}s (${tokSec} tok/s)</span>
      <button class="copy-btn" onclick="copyText(this, \`${escapeForAttr(assistantText)}\`)">Copy</button>
    `;

  } catch (err) {
    bodyEl.textContent += `\n[Generation Error: ${err.message}]`;
  } finally {
    bodyEl.classList.remove("cursor-blink");
    isGenerating = false;
    elements.chatSendBtn.disabled = false;
    elements.chatInput.focus();
  }
}

function appendMessage(role, text) {
  const msgEl = document.createElement("div");
  msgEl.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "👤" : "⚡";

  const content = document.createElement("div");
  content.className = "message-content";

  const header = document.createElement("div");
  header.className = "message-header";
  header.innerHTML = `<span class="sender-name">${role === "user" ? "You" : "ForgeLM"}</span>`;

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;

  content.appendChild(header);
  content.appendChild(body);
  msgEl.appendChild(avatar);
  msgEl.appendChild(content);
  elements.chatThread.appendChild(msgEl);
  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
  return body;
}

function appendAssistantContainer() {
  const msgEl = document.createElement("div");
  msgEl.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "⚡";

  const content = document.createElement("div");
  content.className = "message-content";

  const header = document.createElement("div");
  header.className = "message-header";
  header.innerHTML = `<span class="sender-name">ForgeLM</span><span class="msg-time">Generating...</span>`;

  const body = document.createElement("div");
  body.className = "message-body";

  const footer = document.createElement("div");
  footer.className = "message-footer";

  content.appendChild(header);
  content.appendChild(body);
  content.appendChild(footer);
  msgEl.appendChild(avatar);
  msgEl.appendChild(content);
  elements.chatThread.appendChild(msgEl);
  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;

  return { bodyEl: body, footerEl: footer };
}

function copyText(button, text) {
  navigator.clipboard.writeText(text).then(() => {
    const original = button.textContent;
    button.textContent = "Copied!";
    setTimeout(() => { button.textContent = original; }, 1500);
  });
}

function escapeForAttr(str) {
  return str.replace(/`/g, "\\`").replace(/\$/g, "\\$");
}

// Playground Mode Handler
async function runPlaygroundGeneration() {
  const prompt = elements.playgroundPrompt.value.trim();
  if (!prompt || isGenerating) return;

  isGenerating = true;
  elements.playgroundGenBtn.disabled = true;
  elements.playgroundOutput.textContent = "";
  elements.playgroundOutput.classList.add("cursor-blink");

  const base = getBackendBaseUrl();
  const params = getSamplingParams();
  let tokenCount = 0;
  const startTime = performance.now();

  try {
    const response = await fetch(`${base}/generate/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        max_tokens: params.max_tokens,
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        repetition_penalty: params.repetition_penalty,
        use_cache: params.use_cache
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6).trim();
          try {
            const data = JSON.parse(jsonStr);
            if (data.token) {
              tokenCount++;
              elements.playgroundOutput.textContent += data.token;
              const elapsedSec = (performance.now() - startTime) / 1000;
              const speed = (tokenCount / elapsedSec).toFixed(1);
              elements.metricTokens.textContent = `Tokens: ${tokenCount}`;
              elements.metricSpeed.textContent = `Throughput: ${speed} tok/s`;
            }
          } catch (e) {
            // Incomplete chunk
          }
        }
      }
    }
  } catch (err) {
    elements.playgroundOutput.textContent += `\n[Generation Error: ${err.message}]`;
  } finally {
    elements.playgroundOutput.classList.remove("cursor-blink");
    isGenerating = false;
    elements.playgroundGenBtn.disabled = false;
  }
}

// Telemetry Poller Loop
fetchTelemetry();
fetchModelInfo();
setInterval(fetchTelemetry, 3000);
