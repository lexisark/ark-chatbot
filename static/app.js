const API = '/api';
let currentChatId = null;
let isStreaming = false;

// ── DOM refs ───────────────────────────────────────────

const chatList = document.getElementById('chat-list');
const emptyState = document.getElementById('empty-state');
const chatView = document.getElementById('chat-view');
const chatTitle = document.getElementById('chat-title');
const chatScope = document.getElementById('chat-scope');
const messagesEl = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const deleteChatBtn = document.getElementById('delete-chat-btn');
const editPromptBtn = document.getElementById('edit-prompt-btn');

// Modals
const newChatModal = document.getElementById('new-chat-modal');
const modalTitle = document.getElementById('modal-title');
const modalPrompt = document.getElementById('modal-prompt');
const modalScope = document.getElementById('modal-scope');
const modalCancel = document.getElementById('modal-cancel');
const modalCreate = document.getElementById('modal-create');

const editPromptModal = document.getElementById('edit-prompt-modal');
const editPromptText = document.getElementById('edit-prompt-text');
const editPromptCancel = document.getElementById('edit-prompt-cancel');
const editPromptSave = document.getElementById('edit-prompt-save');

// ── API helpers ────────────────────────────────────────

async function api(path, opts = {}) {
    const res = await fetch(API + path, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    if (res.status === 204) return null;
    return res.json();
}

// ── Chat list ──────────────────────────────────────────

async function loadChats() {
    const chats = await api('/chats');
    chatList.innerHTML = '';
    chats.forEach(chat => {
        const el = document.createElement('div');
        el.className = 'chat-item' + (chat.id === currentChatId ? ' active' : '');
        const time = new Date(chat.created_at).toLocaleDateString();
        el.innerHTML = `
            <span class="chat-item-title">${escapeHtml(chat.title || 'Untitled')}</span>
            <span class="chat-item-time">${time}</span>
        `;
        el.onclick = () => openChat(chat.id);
        chatList.appendChild(el);
    });
}

// ── Open chat ──────────────────────────────────────────

async function openChat(chatId) {
    currentChatId = chatId;
    emptyState.style.display = 'none';
    chatView.style.display = 'flex';

    const chat = await api(`/chats/${chatId}`);
    chatTitle.textContent = chat.title || 'Untitled';
    chatScope.textContent = chat.scope_id ? `scope: ${chat.scope_id}` : '';

    // Store prompt for editing
    chatView.dataset.systemPrompt = chat.system_prompt || '';

    await loadMessages(chatId);
    loadChats();
    messageInput.focus();
}

async function loadMessages(chatId) {
    const messages = await api(`/chats/${chatId}/messages`);
    messagesEl.innerHTML = '';
    messages.forEach(msg => appendMessage(msg.role, msg.content));
    scrollToBottom();
}

function appendMessage(role, content) {
    const el = document.createElement('div');
    el.className = `message ${role}`;
    el.textContent = content;
    messagesEl.appendChild(el);
    return el;
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Send message (streaming) ───────────────────────────

async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || !currentChatId || isStreaming) return;

    messageInput.value = '';
    messageInput.style.height = 'auto';
    isStreaming = true;
    sendBtn.disabled = true;

    // Show user message immediately
    appendMessage('user', content);
    scrollToBottom();

    // Create streaming assistant bubble
    const assistantEl = appendMessage('assistant', '');
    assistantEl.classList.add('streaming');
    scrollToBottom();

    try {
        const res = await fetch(`${API}/chats/${currentChatId}/messages/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                if (data.error) {
                    assistantEl.textContent += `\n[Error: ${data.error}]`;
                } else if (data.done) {
                    // Stream complete
                } else if (data.delta) {
                    assistantEl.textContent += data.delta;
                    scrollToBottom();
                }
            }
        }
    } catch (err) {
        assistantEl.textContent += `\n[Connection error]`;
    }

    assistantEl.classList.remove('streaming');
    isStreaming = false;
    sendBtn.disabled = false;
    messageInput.focus();
}

// ── New chat ───────────────────────────────────────────

function showNewChatModal() {
    modalTitle.value = '';
    modalPrompt.value = '';
    modalScope.value = 'default';
    newChatModal.style.display = 'flex';
    modalTitle.focus();
}

async function createChat() {
    const chat = await api('/chats', {
        method: 'POST',
        body: JSON.stringify({
            title: modalTitle.value || 'New Chat',
            system_prompt: modalPrompt.value || null,
            scope_id: modalScope.value || 'default',
        }),
    });
    newChatModal.style.display = 'none';
    await openChat(chat.id);
}

// ── Delete chat ────────────────────────────────────────

async function deleteChat() {
    if (!currentChatId) return;
    if (!confirm('Delete this chat?')) return;

    await api(`/chats/${currentChatId}`, { method: 'DELETE' });
    currentChatId = null;
    chatView.style.display = 'none';
    emptyState.style.display = 'flex';
    await loadChats();
}

// ── Edit system prompt ─────────────────────────────────

function showEditPrompt() {
    editPromptText.value = chatView.dataset.systemPrompt || '';
    editPromptModal.style.display = 'flex';
    editPromptText.focus();
}

async function savePrompt() {
    if (!currentChatId) return;
    await api(`/chats/${currentChatId}`, {
        method: 'PATCH',
        body: JSON.stringify({ system_prompt: editPromptText.value }),
    });
    chatView.dataset.systemPrompt = editPromptText.value;
    editPromptModal.style.display = 'none';
}

// ── Textarea auto-resize ───────────────────────────────

messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
});

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ── Event listeners ────────────────────────────────────

sendBtn.addEventListener('click', sendMessage);
newChatBtn.addEventListener('click', showNewChatModal);
deleteChatBtn.addEventListener('click', deleteChat);
editPromptBtn.addEventListener('click', showEditPrompt);

modalCancel.addEventListener('click', () => newChatModal.style.display = 'none');
modalCreate.addEventListener('click', createChat);
modalTitle.addEventListener('keydown', (e) => { if (e.key === 'Enter') createChat(); });

editPromptCancel.addEventListener('click', () => editPromptModal.style.display = 'none');
editPromptSave.addEventListener('click', savePrompt);

// Close modals on backdrop click
newChatModal.addEventListener('click', (e) => { if (e.target === newChatModal) newChatModal.style.display = 'none'; });
editPromptModal.addEventListener('click', (e) => { if (e.target === editPromptModal) editPromptModal.style.display = 'none'; });

// ── Utilities ──────────────────────────────────────────

function escapeHtml(text) {
    const el = document.createElement('span');
    el.textContent = text;
    return el.innerHTML;
}

// ── Init ───────────────────────────────────────────────

loadChats();
