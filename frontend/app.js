/**
 * ForgeLM Frontend Client Application
 * Handles chat messaging, playground completion, SSE token streaming, and telemetry polling.
 */

const PRESETS = {
  story: "Once upon a time in a sunny meadow, a little bunny named Barnaby discovered a hidden path.",
  adventure: "Deep in the whispering forest, a brave young explorer stumbled upon an ancient wooden door.",
  question: "Why does the sun shine warm in the summer?",
  friendship: "Leo and Mia were playing with building blocks when suddenly one of the tall towers fell down."
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
  const url = elements.backendUrl ? elements.backendUrl.value.trim() : "http://localhost:8000";
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
  if (PRESETS[key]) {
    elements.playgroundPrompt.value = PRESETS[key];
    elements.playgroundOutput.textContent = "";
  }
}

function clearPlayground() {
  elements.playgroundPrompt.value = "";
  elements.playgroundOutput.textContent = "";
  elements.metricTokens.textContent = "Tokens: 0";
  elements.metricSpeed.textContent = "Speed: 0 tok/s";
}

// Telemetry Poller
async function fetchTelemetry() {
  const base = getBackendBaseUrl();
  try {
    const res = await fetch(`${base}/health`);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    elements.statusDot.className = "status-dot online";
    elements.statusLabel.textContent = `Online (${data.device})`;

    elements.telemetryDevice.textContent = data.device.toUpperCase();
    elements.telemetryVram.textContent = data.vram_allocated_mb ? `${data.vram_allocated_mb} MB` : "N/A";
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
      elements.telemetryParams.textContent = `${paramsM}M`;
      elements.telemetryLayers.textContent = `${data.n_layers} / ${data.n_heads}`;
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

  // Add User Message to UI
  appendMessage("user", text);
  chatMessages.push({ role: "user", content: text });
  elements.chatInput.value = "";

  // Prepare Assistant Bubble for streaming
  const assistantMsgBody = appendMessage("assistant", "");
  assistantMsgBody.classList.add("cursor-blink");
  isGenerating = true;
  elements.chatSendBtn.disabled = true;

  const base = getBackendBaseUrl();
  const params = getSamplingParams();

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
      throw new Error(`Server returned status ${response.status}`);
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
      buffer = lines.pop(); // Retain incomplete chunk

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6).trim();
          try {
            const data = JSON.parse(jsonStr);
            if (data.token) {
              assistantText += data.token;
              assistantMsgBody.textContent = assistantText;
              elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
            }
          } catch (e) {
            // Malformed line
          }
        }
      }
    }

    chatMessages.push({ role: "assistant", content: assistantText });
  } catch (err) {
    assistantMsgBody.textContent += `\n[Error: ${err.message}]`;
  } finally {
    assistantMsgBody.classList.remove("cursor-blink");
    isGenerating = false;
    elements.chatSendBtn.disabled = false;
    elements.chatInput.focus();
  }
}

function appendMessage(role, text) {
  const msgEl = document.createElement("div");
  msgEl.className = `message ${role}`;

  const header = document.createElement("div");
  header.className = "message-header";
  header.innerHTML = `<span class="sender-name">${role === "user" ? "You" : "ForgeLM"}</span>`;

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;

  msgEl.appendChild(header);
  msgEl.appendChild(body);
  elements.chatThread.appendChild(msgEl);
  elements.chatThread.scrollTop = elements.chatThread.scrollHeight;
  return body;
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
      throw new Error(`Server returned status ${response.status}`);
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
              elements.metricSpeed.textContent = `Speed: ${speed} tok/s`;
            }
          } catch (e) {
            // Malformed line
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

// Initial Telemetry Polling & Intervals
fetchTelemetry();
fetchModelInfo();
setInterval(fetchTelemetry, 3000);
