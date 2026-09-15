"""
CyberLawGPT
------------
A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly
based on Pakistan's Cyber Law - the Prevention of Electronic Crimes Act (PECA), 2016.

Stack:
  - Streamlit        -> UI
  - PyPDF             -> PDF text extraction
  - sentence-transformers -> free, local embeddings (no API key needed)
  - FAISS (faiss-cpu) -> vector similarity search
  - Groq API          -> LLM inference (model: openai/gpt-oss-20b)

Designed to run for free on:
  - Google Colab (via streamlit + localtunnel/pyngrok, see README)
  - Streamlit Community Cloud

On startup, the app downloads the configured PDF, extracts + chunks the text,
embeds the chunks, and builds a FAISS index — all cached so it only happens once.
"""

import os
import io
import re
import time
import requests
import numpy as np
import streamlit as st
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
APP_NAME = "CyberLawGPT"
DEFAULT_PDF_URL = "https://www.na.gov.pk/uploads/documents/1470910659_707.pdf"  # PECA 2016 (official, National Assembly of Pakistan)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"          # small, free, fast, runs on CPU
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 60
GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

TECH_LEVEL_PROMPTS = {
    "Beginner (plain language)": (
        "Explain the answer in simple, everyday language for someone with no legal "
        "background. Avoid jargon; when you must use a legal term, briefly explain it "
        "in parentheses."
    ),
    "Intermediate (balanced)": (
        "Explain the answer clearly for an educated non-lawyer. You may use legal "
        "terminology but briefly clarify any technical or legal term the first time "
        "it appears."
    ),
    "Expert (legal/technical)": (
        "Answer as you would for a lawyer or compliance officer. Use precise legal "
        "terminology, cite exact section numbers and clause references from the Act "
        "wherever possible, and be formally worded."
    ),
}

RESPONSE_LENGTH_TOKENS = {
    "Short": 300,
    "Medium": 650,
    "Detailed": 1200,
}

ANSWER_LANGUAGE_PROMPTS = {
    "English": "Respond in clear English.",
    "Roman Urdu": "Respond in Roman Urdu (Urdu written using English letters), keeping legal terms in English where there is no common Roman Urdu equivalent.",
}

SYSTEM_PROMPT_TEMPLATE = """You are {app_name}, an assistant specialized ONLY in Pakistan's cyber law,
primarily the Prevention of Electronic Crimes Act (PECA), 2016, based strictly on the
official Act text provided to you as context below.

Rules you MUST follow:
1. Answer ONLY using the provided context chunks from the Act. Do not use outside knowledge
   of other countries' laws unless the user explicitly asks for a comparison, and in that case
   clearly separate "According to Pakistani law (PECA 2016)" from any general/comparative remarks.
2. If the context does not contain enough information to answer confidently, say so plainly
   instead of guessing or inventing section numbers or penalties.
3. Each context excerpt below is labeled with "[Detected Section(s): ...]" — this is a
   best-effort automatic detection, not guaranteed complete. ONLY cite a section number if
   it appears in one of these labels for the excerpt you are using. If an excerpt has no
   detected section number, describe its content without inventing a number, e.g. say
   "the Act states..." instead of attributing it to a specific, unconfirmed section.
4. You are not a substitute for a licensed lawyer. For anything involving an active legal
   case, filing an FIR/complaint, or high-stakes decisions, remind the user to consult a
   qualified lawyer or the relevant authority (e.g. FIA Cyber Crime Wing / NR3C).
5. Be factual, neutral, and avoid alarmist or exaggerated language.

Style instructions for this answer:
- {tech_level_instruction}
- {language_instruction}
"""

# ----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(page_title=APP_NAME, page_icon="⚖️", layout="wide")


# ----------------------------------------------------------------------------
# HELPERS: PDF DOWNLOAD, PARSING, CHUNKING, EMBEDDING, INDEX
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_pdf(url: str) -> bytes:
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.content


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n".join(pages_text)


# Matches section headers as they typically appear in the Act, e.g. "3. Unauthorized
# access to information system or data.—" or standalone mentions like "Section 21".
SECTION_HEADER_RE = re.compile(r"(?m)^\s*(\d{1,3})\.\s+[A-Z]")
SECTION_MENTION_RE = re.compile(r"\bSection\s+(\d{1,3})\b", re.IGNORECASE)


def detect_section_numbers(text: str) -> list:
    """Best-effort extraction of Act section numbers present in a piece of text.
    Used to tag chunks so the model can only cite sections that actually appear
    in the retrieved context, instead of guessing."""
    found = set()
    for m in SECTION_HEADER_RE.finditer(text):
        num = int(m.group(1))
        if 1 <= num <= 60:  # PECA 2016 has fewer than 60 sections; keeps false positives out
            found.add(num)
    for m in SECTION_MENTION_RE.finditer(text):
        found.add(int(m.group(1)))
    return sorted(found)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    """Splits text into overlapping word-chunks and tags each chunk with any
    section numbers detected inside it, e.g. {"text": ..., "sections": [21, 22]}."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if chunk_words:
            chunk_str = " ".join(chunk_words)
            if len(chunk_str.strip()) > 30:
                chunks.append({"text": chunk_str, "sections": detect_section_numbers(chunk_str)})
        start += chunk_size - overlap
    return chunks


@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def build_knowledge_base(pdf_url: str):
    """Downloads the PDF, chunks it, embeds it, and builds a FAISS index.
    Cached by pdf_url so it only reruns when the source changes."""
    pdf_bytes = download_pdf(pdf_url)
    raw_text = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_text(raw_text)

    embedder = load_embedder()
    embeddings = embedder.encode(
        [c["text"] for c in chunks], show_progress_bar=False, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)

    return {
        "chunks": chunks,
        "index": index,
        "num_chunks": len(chunks),
        "source_url": pdf_url,
    }


def retrieve(kb: dict, query: str, k: int = 4):
    embedder = load_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, idxs = kb["index"].search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        chunk = kb["chunks"][idx]
        results.append({"text": chunk["text"], "sections": chunk.get("sections", []), "score": float(score)})
    return results


def call_groq(api_key: str, model: str, system_prompt: str, chat_history: list,
               user_message: str, context: str, temperature: float, max_tokens: int):
    client = Groq(api_key=api_key)

    full_user_prompt = (
        f"CONTEXT FROM THE ACT (use this to answer):\n{context}\n\n"
        f"USER QUESTION:\n{user_message}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    # include prior turns for conversational context (text only, no retrieved context re-sent)
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": full_user_prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


# ----------------------------------------------------------------------------
# SIDEBAR — SETTINGS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ " + APP_NAME)
    st.caption("RAG chatbot for Pakistan's Cyber Law (PECA 2016)")

    st.subheader("🔑 Groq API")
    default_key = os.environ.get("GROQ_API_KEY", "")
    try:
        default_key = st.secrets.get("GROQ_API_KEY", default_key)
    except Exception:
        pass
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys. "
             "You can also set it as an environment variable or in .streamlit/secrets.toml as GROQ_API_KEY.",
    )

    selected_model = st.selectbox("Model", GROQ_MODELS, index=0)

    st.subheader("📄 Knowledge Source")
    pdf_url = st.text_input("Cyber Law PDF URL", value=DEFAULT_PDF_URL)
    uploaded_pdf = st.file_uploader("...or upload a PDF instead", type=["pdf"])

    st.subheader("🎓 Response Settings")
    tech_level = st.selectbox("Technical level", list(TECH_LEVEL_PROMPTS.keys()), index=1)
    response_length = st.select_slider("Response size", options=list(RESPONSE_LENGTH_TOKENS.keys()), value="Medium")
    answer_language = st.selectbox("Answer language", list(ANSWER_LANGUAGE_PROMPTS.keys()), index=0)

    with st.expander("⚙️ Advanced settings"):
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
        top_k = st.slider("Chunks retrieved (top-k)", 1, 10, 4, 1)
        max_tokens_override = st.number_input(
            "Max tokens (0 = use response size preset)", min_value=0, max_value=4000, value=0, step=50
        )
        show_sources = st.checkbox("Show retrieved source excerpts", value=True)

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.pop("messages", None)
        st.rerun()

    st.caption(
        "⚠️ This app provides general information based on the Prevention of Electronic "
        "Crimes Act, 2016. It is not legal advice. Consult a qualified lawyer for specific cases."
    )

# ----------------------------------------------------------------------------
# BUILD / LOAD KNOWLEDGE BASE
# ----------------------------------------------------------------------------
st.title("⚖️ CyberLawGPT")
st.caption("Ask questions about Pakistan's cyber law (Prevention of Electronic Crimes Act, 2016)")

kb = None
try:
    with st.spinner("Preparing knowledge base (downloading + embedding the Act)... this runs once."):
        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.read()
            raw_text = extract_text_from_pdf(pdf_bytes)
            chunks = chunk_text(raw_text)
            embedder = load_embedder()
            embeddings = np.asarray(
                embedder.encode([c["text"] for c in chunks], show_progress_bar=False, normalize_embeddings=True),
                dtype="float32",
            )
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            kb = {"chunks": chunks, "index": index, "num_chunks": len(chunks), "source_url": "uploaded file"}
        else:
            kb = build_knowledge_base(pdf_url)
except Exception as e:
    st.error(f"Failed to build knowledge base: {e}")
    st.stop()

st.success(f"Knowledge base ready — {kb['num_chunks']} chunks indexed from: {kb['source_url']}", icon="✅")

# ----------------------------------------------------------------------------
# CHAT UI
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Retrieved sources"):
                for i, s in enumerate(msg["sources"], 1):
                    sec_label = f"Section(s) {', '.join(str(x) for x in s['sections'])}" if s.get("sections") else "no section number detected"
                    st.markdown(f"**Chunk {i}** · {sec_label} · similarity: {s['score']:.2f}")
                    st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

user_input = st.chat_input("Ask about Pakistan's cyber law, e.g. 'What is the penalty for hacking?'")

if user_input:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Act and drafting a response..."):
            retrieved = retrieve(kb, user_input, k=top_k)
            context_text = "\n\n---\n\n".join(
                f"[Detected Section(s): {', '.join(str(s) for s in r['sections']) or 'none detected'}]\n{r['text']}"
                for r in retrieved
            )

            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                app_name=APP_NAME,
                tech_level_instruction=TECH_LEVEL_PROMPTS[tech_level],
                language_instruction=ANSWER_LANGUAGE_PROMPTS[answer_language],
            )

            max_tokens = max_tokens_override if max_tokens_override > 0 else RESPONSE_LENGTH_TOKENS[response_length]

            try:
                answer = call_groq(
                    api_key=groq_api_key,
                    model=selected_model,
                    system_prompt=system_prompt,
                    chat_history=st.session_state.messages[:-1],
                    user_message=user_input,
                    context=context_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                answer = f"⚠️ Error calling Groq API: {e}"

            st.markdown(answer)
            if show_sources and retrieved:
                with st.expander("📚 Retrieved sources"):
                    for i, s in enumerate(retrieved, 1):
                        sec_label = f"Section(s) {', '.join(str(x) for x in s['sections'])}" if s.get("sections") else "no section number detected"
                        st.markdown(f"**Chunk {i}** · {sec_label} · similarity: {s['score']:.2f}")
                        st.caption(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved if show_sources else None,
    })
