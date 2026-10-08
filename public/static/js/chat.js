/**
 * HeatGuard AI - Assistant Chat Controller
 * Manages conversational state, message rendering, prompt chip insertion,
 * typing indicators, multi-turn history tracking, and REST communication with /api/chat.
 */

const chatHistory = [];

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
  const sendBtn = document.getElementById("chat-send-btn");

  const messageText = input.value.trim();
  if (!messageText) return;

  // 1. Append User Message to UI & History
  const userBubble = appendUserMessage(messageText);
  chatHistory.push({ role: "user", content: messageText });

  input.value = "";
  input.disabled = true;
  if (sendBtn) sendBtn.disabled = true;

  // 2. Show Typing Indicator
  const typingIndicator = appendTypingIndicator();
  
  // Smoothly scroll to anchor the user's prompt at the top of the viewing area
  scrollChatToElement(userBubble);

  // 3. Call API with message and history context
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: messageText,
        history: chatHistory.slice(-6), // Send last 6 turns for context
      }),
    });

    const json = await res.json();
    typingIndicator.remove();

    if (json.status === "success" && json.chat) {
      appendBotMessage(json.chat);
      chatHistory.push({ role: "assistant", content: json.chat.reply });
      // Keep view anchored to the user prompt & top of the bot reply
      scrollChatToElement(userBubble);
    } else {
      appendErrorMessage(json.message || "An error occurred while answering.");
      scrollChatToElement(userBubble);
    }
  } catch (err) {
    console.error("Chat error:", err);
    typingIndicator.remove();
    appendErrorMessage("Network connection error. Please try again.");
    scrollChatToElement(userBubble);
  } finally {
    input.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    input.focus();
  }
}

function scrollChatToElement(element) {
  if (!element) return;
  const container = document.getElementById("chat-messages");
  if (!container) return;

  // Position the user prompt cleanly at the top of the chat scroll container with comfortable padding
  const containerRect = container.getBoundingClientRect();
  const elementRect = element.getBoundingClientRect();
  const relativeTop = elementRect.top - containerRect.top + container.scrollTop;

  container.scrollTo({
    top: Math.max(0, relativeTop - 12),
    behavior: "smooth"
  });
}

function appendUserMessage(text) {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble message-user";
  bubble.innerText = text;
  container.appendChild(bubble);
  return bubble;
}

function appendTypingIndicator() {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble message-bot";
  bubble.style.width = "70px";
  bubble.innerHTML = `
    <div style="display:flex; gap:4px; align-items:center; justify-content:center; padding:4px 0;">
      <span style="width:7px; height:7px; background:var(--brand-600); border-radius:50%; animation:pulse-dot 1.2s infinite ease-in-out;"></span>
      <span style="width:7px; height:7px; background:var(--brand-600); border-radius:50%; animation:pulse-dot 1.2s infinite ease-in-out 0.2s;"></span>
      <span style="width:7px; height:7px; background:var(--brand-600); border-radius:50%; animation:pulse-dot 1.2s infinite ease-in-out 0.4s;"></span>
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
    <div style="display:flex; align-items:center; gap:0.4rem; font-size:0.82rem; color:var(--brand-800); font-weight:700; margin-bottom:0.4rem;">
      <i data-feather="shield" style="width:14px;height:14px; color:var(--brand-600);"></i>
      <span>HeatGuard Copilot</span>
    </div>
    <div>${formattedHtml}</div>
    <div class="message-sources">
      <span>Grounded Sources:</span>
      ${sourceTags}
    </div>
  `;

  container.appendChild(bubble);
  feather.replace();
  return bubble;
}

function appendErrorMessage(errorText) {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble message-bot";
  bubble.style.borderColor = "#fecaca";
  bubble.style.backgroundColor = "#fef2f2";
  bubble.innerHTML = `
    <div style="color:#dc2626; font-weight:600; display:flex; align-items:center; gap:0.4rem;">
      <i data-feather="alert-circle" style="width:14px;height:14px;"></i>
      <span>${errorText}</span>
    </div>
  `;
  container.appendChild(bubble);
  feather.replace();
}

// Lightweight Markdown & Math Notation Renderer
function markdownToHtml(md) {
  if (!md) return "";
  let text = md;

  // Sanitize any raw LaTeX/Math delimiters into clean readable Unicode
  text = text
    .replace(/\\pm\s*/g, '±')
    .replace(/\\circ\s*/g, '°')
    .replace(/\\text\{([^\}]+)\}/g, '$1')
    .replace(/\\Delta\s*/g, 'Δ')
    .replace(/\\tau\s*\*?/g, 'τ*')
    .replace(/T_\{?\\?max\}?/gi, 'Tmax')
    .replace(/T_\{?\\?min\}?/gi, 'Tmin')
    .replace(/T_\{?\\?3d\}?/gi, 'T3d')
    .replace(/T_\{?\\?7d\}?/gi, 'T7d')
    .replace(/\(\$Tmax\s*[-−]\s*Tmin\$\)/gi, '(Tmax − Tmin)')
    .replace(/\$Tmax\s*[-−]\s*Tmin\$/gi, 'Tmax − Tmin')
    .replace(/\$\Delta T\(t\+1\)\$/gi, 'ΔT(t+1)')
    .replace(/\$R\^2\$/gi, 'R²')
    .replace(/\$([^$]+)\$/g, '$1')
    .replace(/\\([a-zA-Z]+)/g, '$1');

  let html = text
    .replace(/^### (.*$)/gim, '<h3 style="color:var(--text-primary); font-size:1.02rem; font-weight:700; margin:0.4rem 0;">$1</h3>')
    .replace(/^## (.*$)/gim, '<h2 style="color:var(--text-primary); font-size:1.1rem; font-weight:700; margin:0.5rem 0;">$1</h2>')
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
