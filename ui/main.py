"""WhatsApp-style Streamlit chat UI for the agentic RAG API.

Run locally:  streamlit run ui/main.py        (API_URL defaults to http://localhost:8000)
"""
import html
import os
import uuid
from datetime import datetime

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
ESCALATION_TEXT = "escalated this to our human support team"
QUICK_REPLIES = ["How do I cancel my order?", "Where is my refund?", "What payment methods do you accept?",
                 "I want to talk to a human agent"]
WELCOME = ("Hi! 👋 I'm your support assistant. Ask me about orders, refunds, payments, delivery or your "
           "account, or tap a suggestion below.")

st.set_page_config(page_title="Support Assistant", page_icon="💬", layout="centered")

# ---------------- WhatsApp look & feel ----------------
st.markdown("""
<style>
  #MainMenu, footer, header[data-testid="stHeader"], .stDeployButton {visibility:hidden; height:0;}
  .stApp { background-color:#efeae2;
           background-image: radial-gradient(#d9d2c5 1px, transparent 1px),
                             radial-gradient(#e2dccf 1px, transparent 1px);
           background-size: 22px 22px, 22px 22px; background-position: 0 0, 11px 11px; }
  .block-container { padding-top: 1rem; padding-bottom: 6rem; max-width: 760px; }
  .wa-header { position:sticky; top:0; z-index:10; display:flex; align-items:center; gap:12px;
               background:#008069; color:#fff; padding:10px 16px; border-radius:10px; margin-bottom:10px;
               box-shadow:0 2px 4px rgba(0,0,0,.15); }
  .wa-avatar { width:42px; height:42px; border-radius:50%; background:#dfe5e7; display:flex;
               align-items:center; justify-content:center; font-size:22px; }
  .wa-name { font-weight:600; font-size:16px; line-height:1.2; }
  .wa-status { font-size:12px; opacity:.9; }
  .wa-day { text-align:center; margin:8px 0 12px; }
  .wa-day span { background:#ffffffd9; color:#54656f; font-size:12px; padding:5px 12px; border-radius:8px;
                 box-shadow:0 1px .5px rgba(0,0,0,.13); }
  .row { display:flex; margin:2px 0; }
  .row.user { justify-content:flex-end; } .row.bot { justify-content:flex-start; }
  .bubble { position:relative; max-width:80%; padding:7px 10px 6px 10px; border-radius:8px; font-size:14.5px;
            line-height:1.45; color:#111b21; box-shadow:0 1px .5px rgba(0,0,0,.13); word-wrap:break-word;
            animation: pop .18s ease-out; }
  @keyframes pop { from {transform:scale(.96); opacity:.4;} to {transform:scale(1); opacity:1;} }
  .user .bubble { background:#d9fdd3; border-top-right-radius:0; }
  .bot .bubble  { background:#ffffff; border-top-left-radius:0; }
  .user .bubble:after { content:""; position:absolute; top:0; right:-8px; border-width:0 0 10px 10px;
                        border-style:solid; border-color:transparent transparent transparent #d9fdd3; }
  .bot .bubble:before { content:""; position:absolute; top:0; left:-8px; border-width:0 10px 10px 0;
                        border-style:solid; border-color:transparent #ffffff transparent transparent; }
  .time { float:right; font-size:11px; color:#667781; margin:6px 0 -4px 10px; white-space:nowrap; }
  .ticks { color:#53bdeb; margin-left:3px; }
  .src { background:#f5f6f6; border-left:3px solid #00a884; font-size:13px; color:#3b4a54; }
  .src b { color:#00a884; font-size:11px; }
  .pill { text-align:center; margin:6px 0; }
  .pill span { background:#fff3c4; color:#54656f; font-size:12px; padding:4px 10px; border-radius:8px; }
  .typing { display:inline-flex; gap:4px; padding:4px 2px; }
  .typing i { width:7px; height:7px; border-radius:50%; background:#8696a0; display:block;
              animation: blink 1.2s infinite ease-in-out; }
  .typing i:nth-child(2){animation-delay:.2s} .typing i:nth-child(3){animation-delay:.4s}
  @keyframes blink { 0%,80%,100% {opacity:.25; transform:translateY(0);} 40% {opacity:1; transform:translateY(-3px);} }
  .meta { font-size:11px; color:#8696a0; margin:2px 0 0 4px; }
  div[data-testid="stButton"] button { border-radius:18px; border:1px solid #00a884; color:#008069;
                                       background:#fff; padding:2px 12px; font-size:13px; }
  div[data-testid="stButton"] button:hover { background:#e7f8f3; border-color:#008069; color:#008069; }
  div[data-testid="stChatInput"] textarea { font-size:15px; }
  div[data-testid="stExpander"] { background:transparent; border:none; max-width:80%; }
  div[data-testid="stExpander"] summary { font-size:12.5px; color:#008069; padding:0 4px; }
</style>
""", unsafe_allow_html=True)


def esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def bubble(who: str, inner: str, time: str = "", extra: str = ""):
    ticks = '<span class="ticks">✓✓</span>' if who == "user" and time else ""
    stamp = f'<span class="time">{time}{ticks}</span>' if time else ""
    st.markdown(f'<div class="row {who}"><div class="bubble {extra}">{inner}{stamp}</div></div>',
                unsafe_allow_html=True)


def now() -> str:
    return datetime.now().strftime("%H:%M")


# ---------------- Session state ----------------
def new_chat():
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.messages = [{"role": "bot", "text": WELCOME, "time": now()}]
    st.session_state.pending = None


if "session_id" not in st.session_state:
    new_chat()

with st.sidebar:
    st.markdown("### 💬 Chats")
    st.caption(f"Session `{st.session_state.session_id}`")
    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()
    show_details = st.toggle("Show agent details", value=True)
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"🟢 Connected · {h['llm_model']}")
        online = True
    except Exception:
        st.error(f"🔴 API not reachable at {API_URL}")
        online = False

status = "typing…" if st.session_state.pending else ("online" if online else "offline")
st.markdown(f"""<div class="wa-header"><div class="wa-avatar">🎧</div>
  <div><div class="wa-name">Customer Support</div><div class="wa-status">{status}</div></div></div>
  <div class="wa-day"><span>TODAY</span></div>""", unsafe_allow_html=True)


def send_feedback(i: int, rating: str, correction: str | None = None):
    m = st.session_state.messages[i]
    try:
        requests.post(f"{API_URL}/feedback", timeout=10, json={
            "session_id": st.session_state.session_id, "query": m["query"], "answer": m["text"],
            "rating": rating, "correction": correction or None})
        m["feedback"] = rating
        st.toast("Thanks for the feedback! 🙏" if rating == "up" else "Feedback saved, we'll improve this answer.")
    except Exception as e:
        st.toast(f"Could not send feedback: {e}")


def ask(text: str):
    st.session_state.messages.append({"role": "user", "text": text, "time": now()})
    st.session_state.pending = text


# ---------------- Chat history ----------------
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "user":
        bubble("user", esc(m["text"]), m["time"])
        continue

    d = m.get("data")
    text = m["text"]
    escalated = d is not None and ESCALATION_TEXT in text
    if escalated:  # show the hand-off as a WhatsApp-style system pill instead of inside the bubble
        text = text.split("\n\nI've escalated")[0]
    bubble("bot", esc(text), m["time"])

    if d is None:
        continue
    if escalated:
        st.markdown('<div class="pill"><span>🧑‍💼 Chat escalated to a human agent, '
                    'they will contact you shortly</span></div>', unsafe_allow_html=True)
    if show_details:
        st.markdown(f'<div class="meta">confidence {d["confidence"]:.0%} · '
                    f'{" → ".join(d["agents_invoked"])}</div>', unsafe_allow_html=True)
    with st.expander("📎 Sources"):
        for src, ctx in zip(d["sources"], d["retrieved_context"]):
            bubble("bot", f"<b>{esc(src)}</b><br>{esc(ctx)}", extra="src")

    if m.get("feedback"):
        st.markdown(f'<div class="meta">{"👍" if m["feedback"] == "up" else "👎"} feedback sent</div>',
                    unsafe_allow_html=True)
    else:
        c1, c2, _ = st.columns([1, 1, 7])
        c1.button("👍", key=f"up{i}", on_click=send_feedback, args=(i, "up"))
        if c2.button("👎", key=f"down{i}"):
            m["show_fix"] = True
        if m.get("show_fix"):
            with st.form(key=f"fix{i}", border=False):
                fix = st.text_input("What should the answer have said? (optional)", key=f"fixtxt{i}")
                if st.form_submit_button("Send feedback"):
                    send_feedback(i, "down", fix)
                    st.rerun()

# Quick-reply chips (only before the first question)
if len(st.session_state.messages) == 1 and not st.session_state.pending:
    cols = st.columns(2)
    for k, q in enumerate(QUICK_REPLIES):
        if cols[k % 2].button(q, key=f"qr{k}", use_container_width=True):
            ask(q)
            st.rerun()

# ---------------- Pending question: typing indicator, then call the API ----------------
if st.session_state.pending:
    bubble("bot", '<div class="typing"><i></i><i></i><i></i></div>')
    q = st.session_state.pending
    try:
        r = requests.post(f"{API_URL}/query", timeout=180,
                          json={"session_id": st.session_state.session_id, "query": q})
        r.raise_for_status()
        data = r.json()
        st.session_state.messages.append({"role": "bot", "text": data["answer"], "data": data,
                                          "query": q, "time": now()})
    except Exception as e:
        st.session_state.messages.append({"role": "bot", "text": f"⚠️ Sorry, I couldn't reach the server ({e}).",
                                          "time": now()})
    st.session_state.pending = None
    st.rerun()

# ---------------- Input ----------------
if prompt := st.chat_input("Type a message"):
    ask(prompt)
    st.rerun()
