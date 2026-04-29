/**
 * script.js — Lógica del chat RAG Plastitec
 * Conecta la UI con la API FastAPI local.
 */

// Iconos por posición (orden desc de frecuencia)
const FAQ_ICONS = ['📅', '💰', '🏥', '🎉', '📖'];
const FAQ_LABELS = [
    'Vacaciones y Licencias',
    'Nómina y Pagos',
    'Seguros Médicos',
    'Días Festivos',
    'Normas Internas'
];

const API_BASE = window.location.origin;

const chatWindow   = document.getElementById('chat-window');
const chatForm     = document.getElementById('chat-form');
const chatInput    = document.getElementById('chat-input');
const sendBtn      = document.getElementById('send-btn');
const faqContainer = document.getElementById('faq-container');

// ── Init ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadFAQs();
    chatForm.addEventListener('submit', handleSubmit);
});

// ── FAQ Loading ───────────────────────────────────────────────────────
async function loadFAQs() {
    try {
        const res = await fetch(`${API_BASE}/faq`);
        if (!res.ok) throw new Error('No se pudo cargar FAQs');
        const faqs = await res.json();
        renderFAQs(faqs);
    } catch (e) {
        console.error('Error loading FAQs:', e);
        faqContainer.innerHTML = '<p style="color:#6B7280;font-size:0.85rem;">No se pudieron cargar las preguntas frecuentes.</p>';
    }
}

function renderFAQs(faqs) {
    faqContainer.innerHTML = '';

    faqs.forEach((faq, idx) => {
        const card = document.createElement('button');
        card.className = 'faq-card';
        card.innerHTML = `
            <div class="faq-card-icon">${FAQ_ICONS[idx] ?? '❓'}</div>
            <div class="faq-card-num">${idx + 1}. ${FAQ_LABELS[idx] ?? ''}</div>
            <div class="faq-card-title">${faq.pregunta}</div>
            <div class="faq-card-freq">Popularidad, frecuencia: ${faq.frecuencia}</div>
        `;

        card.addEventListener('click', () => {
            // Incrementar contador en background
            fetch(`${API_BASE}/faq/increment`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id_faq: faq.id })
            }).then(() => loadFAQs()).catch(console.error);

            // Enviar al chat
            appendMessage(faq.pregunta, 'user');
            sendToAPI(faq.pregunta, true);
        });

        faqContainer.appendChild(card);
    });
}

// ── Chat Form ─────────────────────────────────────────────────────────
function handleSubmit(e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    appendMessage(text, 'user');
    chatInput.value = '';
    sendToAPI(text);
}

// ── API Call ──────────────────────────────────────────────────────────
async function sendToAPI(pregunta, skipFaqIncrement = false) {
    sendBtn.disabled = true;
    const typingId = appendTyping();

    try {
        const res = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pregunta, skip_faq_increment: skipFaqIncrement })
        });

        removeEl(typingId);
        sendBtn.disabled = false;

        if (!res.ok) {
            appendMessage('Hubo un error al procesar tu solicitud. Inténtalo de nuevo.', 'assistant');
            return;
        }

        const data = await res.json();

        // Construir contenido del bubble con meta al final
        let metaLine = '';
        if (data.fuente && data.fuente !== 'No aplica') {
            metaLine = `Fuente: ${data.fuente} | Confianza: ${data.confianza}`;
        }
        appendMessage(data.respuesta, 'assistant', metaLine);

        // Refrescar FAQs por si hubo coincidencia
        loadFAQs();

    } catch (err) {
        console.error('Error en /ask:', err);
        removeEl(typingId);
        sendBtn.disabled = false;
        appendMessage('No me pude conectar con el servidor. Verifica que esté activo.', 'assistant');
    }
}

// ── DOM Helpers ───────────────────────────────────────────────────────
function appendMessage(text, sender, metaText = '') {
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    // Texto principal
    const p = document.createElement('span');
    p.textContent = text;
    bubble.appendChild(p);

    // Meta (fuente y confianza)
    if (metaText) {
        const meta = document.createElement('div');
        meta.className = 'msg-meta';
        meta.textContent = metaText;
        bubble.appendChild(meta);
    }

    msg.appendChild(bubble);
    chatWindow.appendChild(msg);
    scrollBottom();
}

function appendTyping() {
    const id = 'typing-' + Date.now();
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.id = id;
    msg.innerHTML = `
        <div class="bubble">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    chatWindow.appendChild(msg);
    scrollBottom();
    return id;
}

function removeEl(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
}
