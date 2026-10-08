/**
 * HeatGuard AI - Assistant Chat Controller
 * Manages conversational state, message rendering, prompt chip insertion,
 * typing indicators, and REST communication with /api/chat.
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const chips = document.querySelectorAll(".prompt-chip");

  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      sendMessage();
    });
  }

  // Handle Prompt Chips
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (input && prompt) {
        input.value = prompt;
        sendMessage();
      }
    });
  });
});

async function sendMessage() {
  const input = document.getElementById("chat-input");
  const messagesContainer = document.getElementById("chat-messages");
  const citySelect = document.getElementById("chat-city-select");
  const dateSelect = document.getElementById("chat-date-select");
  const sendBtn = document.getElementById("chat-send-btn");

  const messageText = input.value.trim();
  if (!messageText) return;

  const city = citySelect ? citySelect.value : "Delhi";
  const date = dateSelect ? dateSelect.value : "2024-05-28";

  // 1. Append User Message
  appendUserMessage(messageText);
  input.value = "";
  input.disabled = true;
  if (sendBtn) sendBtn.disabled = true;

  // 2. Show Typing Indicator
  const typingIndicator = appendTypingIndicator();
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // 3. Call API
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: messageText,
        city: city,
        date: date,
      }),
    });

    const json = await res.json();
    typingIndicator.remove();

    if (json.status === "success") {
      appendBotMessage(json.chat);
    } else {
      appendErrorMessage(json.message || "An error occurred while answering.");
    }
  } catch (err) {
    console.error("Chat error:", err);
    typingIndicator.remove();
    appendErrorMessage("Network connection error. Please try again.");
  } finally {
    input.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    input.focus();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

function appendUserMessage(text) {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble message-user";
  bubble.innerText = text;
  container.appendChild(bubble);
}

function appendTypingIndicator() {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble message-bot";
  bubble.style.width = "70px";
  bubble.innerHTML = `
    <div style="display:flex; gap:4px; align-items:center; justify-content:center; padding:4px 0;">
      <span style="width:7px; height:7px; background:#f97316; border-radius:50%; animation:pulse-dot 1.2s infinite ease-in-out;"></span>
      <span style="width:7px; height:7px; background:#f97316; border-radius:50%; animation:pulse-dot 1.2s infinite ease-in-out 0.2s;"></span>
      <span style="width:7px; height:7px; background:#f97316; border-radius:50%; animation:pulse-dot 1.2s infinite ease-in-out 0.4s;"></span>
    </div>
  `;
  container.appendChild(bubble);
  return bubble;
}

function appendBotMessage(chatData) {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble message-bot";

  const sources = chatData.sources || ["Two-Stage Hybrid Engine", "heatguard_clean.csv"];
  const sourceTags = sources.map(s => `<span class="source-tag">${s}</span>`).join("");

  const formattedHtml = markdownToHtml(chatData.reply);

  bubble.innerHTML = `
    <div style="display:flex; align-items:center; gap:0.4rem; font-size:0.8rem; color:var(--accent-ember); font-weight:700; margin-bottom:0.4rem;">
      <i data-feather="shield" style="width:14px;height:14px;"></i>
      <span>HeatGuard AI Assistant</span>
    </div>
    <div>${formattedHtml}</div>
    <div class="message-sources">
      <span>Grounded Sources:</span>
      ${sourceTags}
    </div>
  `;

  container.appendChild(bubble);
  feather.replace();
}

function appendErrorMessage(errorText) {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble message-bot";
  bubble.style.borderColor = "rgba(239, 68, 68, 0.4)";
  bubble.innerHTML = `
    <div style="color:#ef4444; font-weight:600; display:flex; align-items:center; gap:0.4rem;">
      <i data-feather="alert-circle" style="width:14px;height:14px;"></i>
      <span>${errorText}</span>
    </div>
  `;
  container.appendChild(bubble);
  feather.replace();
}

// Lightweight Markdown Renderer
function markdownToHtml(md) {
  if (!md) return "";
  let html = md
    .replace(/^### (.*$)/gim, '<h3 style="color:#fb923c; font-size:1.05rem; margin:0.4rem 0;">$1</h3>')
    .replace(/^## (.*$)/gim, '<h2 style="color:#fff; font-size:1.15rem; margin:0.5rem 0;">$1</h2>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/gim, '<em>$1</em>')
    .replace(/^\-\s+(.*$)/gim, '<li>$1</li>')
    .replace(/^\d+\.\s+(.*$)/gim, '<li>$1</li>')
    .replace(/\n\n+/g, '<br><br>');

  if (html.includes('<li>')) {
    html = html.replace(/(<li>.*?<\/li>)/gims, '<ul style="padding-left:1.25rem; margin:0.4rem 0;">$1</ul>');
  }
  return html;
}
