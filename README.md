# DocuMind — AI Knowledge Assistant

Upload PDFs, ask questions, get answers grounded strictly in what's in those documents. No hallucinated facts, no outside knowledge bleeding in. Built with RAG (Retrieval-Augmented Generation) on top of Groq's inference API and a fully local embedding pipeline.

---

## What it does

You upload one or more PDFs and DocuMind indexes them into a local vector store. From there you can ask anything — it retrieves the most relevant passages using a hybrid BM25 + semantic search approach and generates an answer that cites exactly where in the document it came from.

A few things that aren't obvious from the description:

- **Conversation history is persistent and resumable.** Every conversation is saved to disk. You can close the app, come back later, click a past conversation in the sidebar, and continue asking questions exactly where you left off — the document index is restored too, not just the messages.
- **Diagram generation.** Ask it to "draw a flowchart of the process described in section 3" and it generates a Mermaid.js diagram grounded in the actual document content.
- **Follow-up suggestions.** After each answer it suggests 3 follow-up questions that it can actually answer from the retrieved context — not generic ones.
- **Confidence scoring.** Each answer gets a High/Medium/Low confidence badge based on cosine similarity between your query and the best-matching chunk. Useful for knowing when to trust the answer.
- **Source highlighting.** Each cited source shows the specific sentence from that chunk that best matches the answer, so you can verify it without reading the whole page.

---

## Stack

| Component | What's used |
|---|---|
| UI | Streamlit |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (runs locally) |
| Vector store | ChromaDB (persistent, per-conversation collections) |
| Retrieval | Hybrid BM25 (40%) + MMR semantic (60%) via EnsembleRetriever |
| PDF parsing | PyMuPDF |
| Conversation storage | JSON files on disk |

Embeddings run entirely on-device — no embedding API, no cost per document.

---

## Project structure

```
documind/
├── app.py                  # Streamlit entrypoint
├── src/
│   ├── ingestion.py        # PDF extraction and chunking
│   ├── vectorstore.py      # ChromaDB wrapper, hybrid retriever
│   ├── rag.py              # LangChain chains, prompts, answer logic
│   └── conversation_store.py  # Save/load/delete conversations
├── data/                   # Created at runtime (gitignored)
│   ├── chroma_db/          # Persistent vector indexes
│   └── conversations/      # Saved conversation JSON files
├── requirements.txt
└── .env                    # GROQ_API_KEY goes here (gitignored)
```

---

## Running locally

**1. Clone and install dependencies**

```bash
git clone https://github.com/your-username/documind.git
cd documind
pip install -r requirements.txt
```

**2. Get a Groq API key**

Go to [console.groq.com](https://console.groq.com), create an account, and generate a free API key.

**3. Create a `.env` file**

```
GROQ_API_KEY=your_key_here
```

**4. Run**

```bash
streamlit run app.py
```

The first run downloads the sentence-transformers model (~90 MB). After that it's cached locally.

---

## Deploying to Streamlit Cloud

**Before you push:**

1. Make sure `.env` and `data/` are in your `.gitignore` — you don't want your API key or local indexes in the repo.

```gitignore
.env
data/
__pycache__/
*.pyc
.DS_Store
```

2. Commit everything else including `requirements.txt`.

3. Push to a public (or private) GitHub repo.

**On Streamlit Cloud:**

1. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
2. Click **New app**, select your repo, set the main file to `app.py`.
3. Open **Advanced settings → Secrets** and add:

```toml
GROQ_API_KEY = "your_key_here"
```

4. Deploy.

**One thing to know about cloud deployment:** Streamlit Cloud uses an ephemeral filesystem — the `data/` directory (conversations and ChromaDB indexes) gets wiped whenever the app restarts or redeploys. Conversation history and saved indexes won't survive restarts. For a demo or assignment submission this is fine; for a production app you'd want to swap the local JSON/ChromaDB storage for something like Supabase or Pinecone.

---

## Notes on chunking and retrieval

Chunks are 1500 characters with 300-character overlap. This is intentionally larger than the typical 512-token default — it keeps section headings in the same chunk as the content below them, which matters when documents use abbreviated labels like "P1" or "Table 3.2" that need the surrounding context to be meaningful.

Retrieval uses Reciprocal Rank Fusion to merge BM25 (exact keyword matching) and MMR semantic search. BM25 handles things like author names, table references, and specific abbreviations that semantic search tends to miss. MMR handles paraphrased or conceptually related queries. The 40/60 weighting was chosen empirically — semantic search generally outperforms BM25 on open-ended questions, but the keyword component is essential for technical documents.

---

## License

MIT
