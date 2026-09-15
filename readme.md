# ⚖️ CyberLawGPT

A free, open-source **RAG (Retrieval-Augmented Generation)** chatbot that answers
questions strictly based on **Pakistan's Cyber Law** — the *Prevention of Electronic
Crimes Act (PECA), 2016*.

- 🧠 **LLM:** Groq API, model `openai/gpt-oss-20b` (free tier, very fast)
- 🔎 **Retrieval:** FAISS (local, in-memory vector index)
- ✂️ **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (runs free, locally, on CPU)
- 📄 **Source document:** downloaded automatically on startup from the official
  National Assembly of Pakistan PDF (URL is editable in the sidebar; you can also
  upload your own PDF, e.g. amendments or a different Act)

> ⚠️ **Disclaimer:** This app gives general information based on the Act's text.
> It is **not legal advice**. For real cases, consult a licensed lawyer or the
> FIA Cyber Crime Wing (NR3C).

---

## 📁 Files

| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit application (RAG pipeline + chat UI) |
| `requirement.txt` | Python dependencies |
| `readme.md` | This file |

---

## 🚀 Quick Start (Local Machine)

```bash
# 1. Clone / copy the 3 files into a folder, then:
pip install -r requirement.txt

# 2. Set your free Groq API key (get one at https://console.groq.com/keys)
export GROQ_API_KEY="your_key_here"        # macOS/Linux
setx GROQ_API_KEY "your_key_here"          # Windows (new terminal after)

# 3. Run it
streamlit run app.py
```

The app will open at `http://localhost:8501`. On first load it downloads the
PECA 2016 PDF, extracts text, chunks it, embeds it with a local model, and
builds a FAISS index — all cached so it only happens once per session.

You can also paste your Groq API key directly into the sidebar text box
instead of using an environment variable.

---

## 🟢 Run for Free on Google Colab

Streamlit apps can't be viewed directly in Colab's output, so we expose the
local port with **localtunnel** (no signup needed) or **pyngrok**.

```python
# --- Cell 1: setup ---
!pip install -q streamlit groq faiss-cpu sentence-transformers pypdf requests numpy

# Upload app.py, requirement.txt, readme.md to the Colab file browser
# (or git clone your repo containing them)

import os
os.environ["GROQ_API_KEY"] = "your_key_here"   # paste your free Groq key

# --- Cell 2: run + tunnel ---
!npm install -g localtunnel > /dev/null 2>&1
!streamlit run app.py &>/content/logs.txt &
import time; time.sleep(8)
!npx localtunnel --port 8501
```

Click the printed `https://xxxx.loca.lt` URL, then click **"Click to Continue"**
on the localtunnel warning page — the app will load. The tunnel password (if
asked) is your Colab runtime's public IP, printed by running:

```python
!curl https://loca.lt/mytunnelpassword
```

**Alternative (pyngrok):**

```python
!pip install -q pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("your_ngrok_token")  # free at https://dashboard.ngrok.com
public_url = ngrok.connect(8501)
print(public_url)
```

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. Push the three files (`app.py`, `requirement.txt`, `readme.md`) to a public
   (or private) GitHub repository.
   - ⚠️ Streamlit Cloud looks for a file named exactly **`requirements.txt`**
     (with an "s"). Either rename `requirement.txt` → `requirements.txt` before
     deploying, or add a `requirements.txt` that simply contains the same lines.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick
   your repo, branch, and `app.py` as the entry point.
3. In **App settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
4. Click **Deploy**. First load will take a minute while it installs
   dependencies and builds the FAISS index; subsequent loads are fast thanks
   to caching (`st.cache_resource` / `st.cache_data`).

---

## 🎛️ UI Options Available in the App

| Control | What it does |
|---|---|
| **Groq API Key** | Your key (password field), or set via env var / `secrets.toml` |
| **Model** | Choose between `openai/gpt-oss-20b` (default), `openai/gpt-oss-120b`, Llama 3.3/3.1, etc. |
| **Cyber Law PDF URL / Upload** | Swap the source document without editing code |
| **Technical level** | Beginner (plain language) / Intermediate / Expert (legal/technical, cites sections) |
| **Response size** | Short / Medium / Detailed (controls answer length) |
| **Answer language** | English or Roman Urdu |
| **Creativity (temperature)** | How deterministic vs. creative the answer is |
| **Chunks retrieved (top-k)** | How many passages from the Act are fed to the model per question |
| **Max tokens override** | Manually cap response length instead of using the preset |
| **Show retrieved sources** | Toggle to display the exact excerpts used to answer, with similarity scores |
| **Clear chat history** | Resets the conversation |

---

## 🧩 How It Works (RAG Pipeline)

1. **Ingest:** Download the PDF → extract raw text with `pypdf`.
2. **Chunk:** Split into ~350-word overlapping chunks for good retrieval granularity.
3. **Embed:** Encode each chunk locally with `all-MiniLM-L6-v2` (no API cost).
4. **Index:** Store normalized embeddings in a FAISS `IndexFlatIP` (cosine similarity).
5. **Retrieve:** On each question, embed the query and fetch the top-k most similar chunks.
6. **Generate:** Send the retrieved chunks + question + style instructions to
   Groq's `openai/gpt-oss-20b`, which answers grounded strictly in that context.

---

## 🔐 Notes on Free Usage

- Groq's API has a generous **free tier** — get a key at
  [console.groq.com/keys](https://console.groq.com/keys).
- Embeddings and FAISS run **entirely locally/CPU**, so there's no cost or
  external API call for retrieval — only the final answer generation uses Groq.
- Everything fits comfortably within Streamlit Community Cloud's free resource
  limits and Colab's free CPU runtime.

---

## ⚠️ Limitations

- Answers are only as good as the source PDF's text extraction quality (scanned/
  low-quality PDFs may extract poorly — consider running OCR first if needed).
- This is a general-information tool, not a substitute for professional legal counsel.
- Always verify section numbers and penalties against the official gazette text
  for anything consequential.
