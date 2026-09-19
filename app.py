import os

import streamlit as st
from dotenv import load_dotenv

from src.ingestion import process_uploaded_files
from src.vectorstore import (
    create_vectorstore,
    load_vectorstore,
    delete_collection,
    reset_vectorstore,
)
from src.rag import (
    ask_question,
    generate_document_summary,
    generate_diagram,
    generate_followup_questions,
    is_diagram_request,
    INSUFFICIENT_INFORMATION_RESPONSE,
)
from src.conversation_store import (
    generate_conversation_id,
    collection_name_for,
    save_conversation,
    save_chunks_initial,
    list_conversations,
    load_conversation,
    delete_conversation,
    deserialize_chunks,
    format_display_date,
)


# =========================================================
# Environment
# =========================================================

load_dotenv()


# =========================================================
# Streamlit configuration
# =========================================================

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Styling
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-heading {
        font-size: 2.7rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }

    .main-subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }

    .source-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .stat-label {
        color: #777;
        font-size: 0.9rem;
    }

    .summary-box {
        background: rgba(99, 102, 241, 0.07);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.88rem;
        line-height: 1.5;
        color: inherit;
    }

    /* Follow-up question buttons */
    div[data-testid="stHorizontalBlock"] .stButton button {
        font-size: 0.82rem;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.5);
        background: rgba(99, 102, 241, 0.06);
        color: inherit;
        white-space: normal;
        text-align: left;
        height: auto;
    }
    div[data-testid="stHorizontalBlock"] .stButton button:hover {
        background: rgba(99, 102, 241, 0.14);
        border-color: #6366f1;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Session state initialization
# =========================================================

SESSION_DEFAULTS = {
    "vectorstore": None,
    "messages": [],
    "chat_history": [],
    "document_stats": [],
    "document_names": [],
    "last_evaluation": None,
    "all_chunks": [],
    "document_summaries": {},
    "pending_question": None,
    "current_conv_id": None,          # ID of the active conversation
    "current_collection_name": None,  # ChromaDB collection for active conv
}

for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================================================
# Helpers
# =========================================================

def reset_application():
    reset_vectorstore()
    for key, val in SESSION_DEFAULTS.items():
        st.session_state[key] = val if not isinstance(val, (list, dict)) else type(val)()
    st.session_state.vectorstore = None
    st.session_state.current_conv_id = None
    st.session_state.current_collection_name = None


def persist_current_conversation():
    """
    Auto-save the active conversation to disk after each exchange.
    Chunks are only written on the initial save; subsequent saves
    update messages only (the JSON file already has the chunks).
    """
    conv_id = st.session_state.current_conv_id
    if not conv_id or not st.session_state.messages:
        return

    save_conversation(
        conv_id=conv_id,
        collection_name=st.session_state.current_collection_name or "",
        messages=st.session_state.messages,
        document_names=st.session_state.document_names,
        document_stats=st.session_state.document_stats,
        document_summaries=st.session_state.document_summaries,
        chunks=st.session_state.all_chunks,
    )


def switch_to_conversation(conv_id: str):
    """
    Load a saved conversation and make it the active session.

    Restores: vectorstore, chunks, messages, chat_history,
              document info, and conversation IDs.
    """
    data = load_conversation(conv_id)
    if not data:
        st.error("Could not load that conversation.")
        return

    collection_name = data.get("collection_name", "")
    chunks = deserialize_chunks(data.get("chunks", []))

    # Restore the vectorstore for this conversation's collection.
    try:
        vectorstore = load_vectorstore(collection_name)
    except Exception:
        st.error(
            "Could not restore the knowledge base for this conversation. "
            "The vector index may have been cleared."
        )
        return

    # Build chat_history from the saved messages.
    chat_history = []
    msgs = data.get("messages", [])
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            chat_history.append((msgs[i]["content"], msgs[i + 1]["content"]))

    # Apply to session state.
    st.session_state.vectorstore = vectorstore
    st.session_state.all_chunks = chunks
    st.session_state.messages = msgs
    st.session_state.chat_history = chat_history
    st.session_state.document_names = data.get("document_names", [])
    st.session_state.document_stats = data.get("document_stats", [])
    st.session_state.document_summaries = data.get("document_summaries", {})
    st.session_state.current_conv_id = conv_id
    st.session_state.current_collection_name = collection_name
    st.session_state.last_evaluation = None
    st.session_state.pending_question = None


def build_conversation_text():
    lines = ["DocuMind Conversation", "=" * 60, ""]

    if not st.session_state.messages:
        lines.append("No conversation yet.")
        return "\n".join(lines)

    for message in st.session_state.messages:
        role = message["role"].upper()
        lines.append(f"{role}:")
        lines.append(message["content"])

        if message.get("type") == "diagram" and message.get("mermaid_code"):
            lines += ["", "[Mermaid Diagram]", message["mermaid_code"]]

        sources = message.get("sources", [])
        if sources:
            lines += ["", "Sources:"]
            for source in sources:
                lines.append(f"- {source['source']} — Page {source['page']}")

        lines += ["", "-" * 60, ""]

    return "\n".join(lines)


def render_sources(sources):
    if not sources:
        return

    with st.expander("📚 Sources", expanded=False):
        for source in sources:
            best_sentence = source.get("best_sentence", "")

            def _esc(t):
                return (
                    t.replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;")
                     .replace('"', "&quot;")
                )

            sentence_html = ""
            if best_sentence:
                sentence_html = f"""
                <div style="margin-top:7px; padding:6px 10px;
                            background:rgba(234,179,8,0.10);
                            border-left:3px solid #eab308;
                            border-radius:0 6px 6px 0;
                            font-size:0.82rem; color:inherit;
                            line-height:1.5;">
                    🔍 &ldquo;{_esc(best_sentence)}&rdquo;
                </div>
                """

            st.markdown(
                f"""
                <div class="source-card">
                    📄 <strong>{_esc(source["source"])}</strong>
                    &mdash; Page {source["page"]}
                    {sentence_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_confidence_badge(score: float):
    if score >= 0.45:
        color, label, icon = "#22c55e", "High Confidence", "🟢"
    elif score >= 0.28:
        color, label, icon = "#f59e0b", "Medium Confidence", "🟡"
    else:
        color, label, icon = "#ef4444", "Low Confidence", "🔴"

    st.markdown(
        f"""
        <div style="display:inline-flex; align-items:center; gap:6px;
                    background:{color}18; border:1px solid {color}55;
                    border-radius:6px; padding:3px 12px; margin-top:6px;
                    font-size:0.82rem; color:{color}; font-weight:600;">
            {icon}&nbsp;{label}
            <span style="font-weight:400;opacity:0.75;margin-left:4px;">
                ({score:.2f})
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_followup_buttons(followups: list, msg_index: int):
    if not followups:
        return

    st.markdown(
        "<div style='margin-top:10px; font-size:0.82rem; "
        "color:#888;'>💬 <em>Follow-up questions:</em></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(followups))
    for col_idx, (col, question) in enumerate(zip(cols, followups)):
        if col.button(
            question,
            key=f"followup_{msg_index}_{col_idx}",
            use_container_width=True,
        ):
            st.session_state.pending_question = question
            st.rerun()


def render_mermaid(mermaid_code: str):
    safe_code = mermaid_code.replace("`", "&#96;")
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            body {{ margin:0; padding:8px; background:transparent; font-family:sans-serif; }}
            .mermaid {{ text-align:center; }}
            svg {{ max-width:100%; height:auto; }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{safe_code}
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
                flowchart: {{ useMaxWidth: true, htmlLabels: true }},
            }});
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=540, scrolling=True)


def render_mermaid_raw(mermaid_code: str):
    with st.expander("🔍 View Diagram Code", expanded=False):
        st.code(mermaid_code, language="text")


def render_chat_messages(messages: list):
    for msg_idx, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                msg_type = message.get("type", "text")

                if msg_type == "diagram":
                    mermaid_code = message.get("mermaid_code")
                    if mermaid_code:
                        render_mermaid(mermaid_code)
                        render_mermaid_raw(mermaid_code)
                else:
                    confidence = message.get("confidence_score")
                    if confidence is not None:
                        render_confidence_badge(confidence)

                    followups = message.get("followup_questions", [])
                    render_followup_buttons(followups, msg_idx)

                render_sources(message.get("sources", []))


# =========================================================
# Header
# =========================================================

st.markdown('<div class="main-heading">📚 DocuMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">AI Knowledge Assistant powered by Retrieval-Augmented Generation</div>',
    unsafe_allow_html=True,
)


# =========================================================
# API key validation
# =========================================================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY was not found. "
        "Create a .env file and add your Groq API key."
    )
    st.code("GROQ_API_KEY=your_api_key_here")
    st.stop()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    # ---------------------------------------------------------
    # Conversation history
    # ---------------------------------------------------------

    st.header("🕘 Chat History")

    conversations = list_conversations()

    if not conversations:
        st.caption("No saved conversations yet.")
    else:
        for conv in conversations:
            is_active = conv["id"] == st.session_state.current_conv_id

            date_label = format_display_date(
                conv.get("updated_at", conv.get("created_at", ""))
            )
            q_count = conv["message_count"]
            subtitle = (
                f"{date_label} · {q_count} "
                f"question{'s' if q_count != 1 else ''}"
            )

            title_text = conv["title"]
            if len(title_text) > 36:
                title_text = title_text[:36] + "…"

            prefix = "▸ " if is_active else ""

            col_btn, col_del = st.columns([5, 1])

            with col_btn:
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    f"{prefix}{title_text}\n\n_{subtitle}_",
                    key=f"conv_open_{conv['id']}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    if not is_active:
                        switch_to_conversation(conv["id"])
                        st.rerun()

            with col_del:
                if st.button(
                    "🗑",
                    key=f"conv_del_{conv['id']}",
                    help="Delete this conversation",
                ):
                    # Delete the Chroma collection for this conversation.
                    cname = conv.get("collection_name", "")
                    if cname:
                        delete_collection(cname)

                    delete_conversation(conv["id"])

                    # If this was the active conversation, clear state.
                    if st.session_state.current_conv_id == conv["id"]:
                        st.session_state.vectorstore = None
                        st.session_state.messages = []
                        st.session_state.chat_history = []
                        st.session_state.document_names = []
                        st.session_state.document_stats = []
                        st.session_state.document_summaries = {}
                        st.session_state.all_chunks = []
                        st.session_state.current_conv_id = None
                        st.session_state.current_collection_name = None
                        st.session_state.last_evaluation = None

                    st.rerun()

    st.divider()

    # ---------------------------------------------------------
    # Knowledge base upload
    # ---------------------------------------------------------

    st.header("📁 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Upload one or more text-based PDF files. "
            "Scanned PDFs without selectable text require OCR "
            "and are not supported by this version."
        ),
    )

    process_button = st.button(
        "⚙️ Process Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_button:
        try:
            with st.status(
                "Building knowledge base...", expanded=True
            ) as status:

                st.write("Extracting PDF text and page metadata...")
                chunks, stats = process_uploaded_files(uploaded_files)

                if not chunks:
                    status.update(label="No extractable text found.", state="error")
                    st.error("The uploaded PDFs do not contain extractable text.")
                    st.stop()

                st.write(f"Created {len(chunks)} text chunks.")
                st.write("Generating local embeddings using all-MiniLM-L6-v2...")

                # Each conversation gets its own collection.
                new_conv_id = generate_conversation_id()
                new_collection = collection_name_for(new_conv_id)

                vectorstore = create_vectorstore(chunks, collection_name=new_collection)

                st.write("Saving vectors to persistent ChromaDB...")
                st.session_state.vectorstore = vectorstore
                st.session_state.all_chunks = chunks
                st.session_state.document_stats = stats
                st.session_state.document_names = [
                    stat["filename"] for stat in stats
                ]

                st.write("Generating document summaries...")
                summaries = {}
                for stat in stats:
                    filename = stat["filename"]
                    doc_chunks = [
                        c for c in chunks
                        if c.metadata.get("source") == filename
                    ]
                    summaries[filename] = generate_document_summary(
                        doc_chunks, filename
                    )
                st.session_state.document_summaries = summaries

                # Initialise the new conversation.
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.last_evaluation = None
                st.session_state.pending_question = None
                st.session_state.current_conv_id = new_conv_id
                st.session_state.current_collection_name = new_collection

                # Write the initial record (with chunks) to disk.
                save_chunks_initial(
                    conv_id=new_conv_id,
                    collection_name=new_collection,
                    chunks=chunks,
                    document_names=st.session_state.document_names,
                    document_stats=stats,
                    document_summaries=summaries,
                )

                status.update(label="Knowledge base ready", state="complete", expanded=False)

            st.success(
                f"Indexed {len(stats)} document(s) and {len(chunks)} chunks."
            )

        except Exception as error:
            st.error(f"Document processing failed:\n\n{error}")

    # ---------------------------------------------------------
    # Document statistics + summaries
    # ---------------------------------------------------------

    if st.session_state.document_stats:

        st.divider()
        st.subheader("📊 Document Stats")

        for stat in st.session_state.document_stats:
            with st.container(border=True):
                st.markdown(f"**📄 {stat['filename']}**")
                col1, col2 = st.columns(2)
                col1.metric("Pages", stat["pages"])
                col2.metric("Chunks", stat["chunks"])

                summary = st.session_state.document_summaries.get(
                    stat["filename"]
                )
                if summary:
                    st.markdown(
                        f'<div class="summary-box">💡 {summary}</div>',
                        unsafe_allow_html=True,
                    )

        total_pages = sum(s["pages"] for s in st.session_state.document_stats)
        total_chunks = sum(s["chunks"] for s in st.session_state.document_stats)
        st.caption(f"Total: {total_pages} pages • {total_chunks} chunks")

    # ---------------------------------------------------------
    # Per-document filtering
    # ---------------------------------------------------------

    selected_documents = []

    if st.session_state.document_names:
        st.divider()
        st.subheader("🔎 Search Documents")
        st.caption(
            "Choose which documents may be used to answer questions."
        )

        for index, document_name in enumerate(st.session_state.document_names):
            enabled = st.checkbox(
                document_name,
                value=True,
                key=f"document_toggle_{index}",
            )
            if enabled:
                selected_documents.append(document_name)

    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------

    st.divider()

    st.download_button(
        label="⬇️ Download Conversation",
        data=build_conversation_text(),
        file_name="documind_conversation.txt",
        mime="text/plain",
        use_container_width=True,
    )

    if st.button("🗑️ Clear All & Reset", use_container_width=True):
        reset_application()
        st.success("All conversations and knowledge base cleared.")
        st.rerun()


# =========================================================
# Tabs
# =========================================================

chat_tab, evaluation_tab = st.tabs(["💬 Chat", "📈 Retrieval Evaluation"])


# =========================================================
# Chat tab
# =========================================================

with chat_tab:

    if st.session_state.vectorstore is None:
        st.info(
            "Upload one or more PDFs in the sidebar and click "
            "**Process Documents** to begin, or select a past "
            "conversation from **Chat History**."
        )
    else:
        if not selected_documents:
            st.warning(
                "Select at least one document in the sidebar "
                "before asking a question."
            )

        st.caption(
            "💡 Tip: Ask me to **draw a flowchart** or "
            "**generate a diagram** of any process in your documents."
        )

        # Render the full conversation history for the active session.
        render_chat_messages(st.session_state.messages)

        # -------------------------------------------------
        # Resolve the active question
        # -------------------------------------------------

        direct_input = st.chat_input(
            "Ask a question or request a flowchart / diagram...",
            disabled=not selected_documents,
        )

        pending = st.session_state.pending_question
        if pending:
            st.session_state.pending_question = None

        user_question = direct_input or pending

        # -------------------------------------------------
        # Process the question
        # -------------------------------------------------

        if user_question:
            st.session_state.messages.append(
                {"role": "user", "content": user_question}
            )

            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):

                if is_diagram_request(user_question):
                    with st.spinner("Retrieving content and generating diagram..."):
                        try:
                            diagram_result = generate_diagram(
                                vectorstore=st.session_state.vectorstore,
                                question=user_question,
                                selected_sources=selected_documents,
                                all_chunks=st.session_state.all_chunks,
                            )

                            if diagram_result["is_insufficient"]:
                                answer = (
                                    "I couldn't find enough information "
                                    "in the documents to generate a diagram "
                                    "for that topic."
                                )
                                st.markdown(answer)
                                render_sources(diagram_result["sources"])

                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": answer,
                                        "type": "text",
                                        "sources": diagram_result["sources"],
                                    }
                                )

                            else:
                                mermaid_code = diagram_result["mermaid_code"]
                                intro = (
                                    "Here is the flowchart based on "
                                    "the document content:"
                                )
                                st.markdown(intro)
                                render_mermaid(mermaid_code)
                                render_mermaid_raw(mermaid_code)
                                render_sources(diagram_result["sources"])

                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": intro,
                                        "type": "diagram",
                                        "mermaid_code": mermaid_code,
                                        "sources": diagram_result["sources"],
                                    }
                                )

                        except Exception as error:
                            st.error("An error occurred while generating the diagram.")
                            st.code(str(error))

                else:
                    with st.spinner("Searching the knowledge base..."):
                        try:
                            result = ask_question(
                                vectorstore=st.session_state.vectorstore,
                                question=user_question,
                                chat_history=st.session_state.chat_history,
                                selected_sources=selected_documents,
                                all_chunks=st.session_state.all_chunks,
                            )

                            answer = result["answer"]
                            sources = result["sources"]
                            evaluation = result["evaluation"]
                            confidence = evaluation["top_relevance_score"]

                            answer_found = (
                                INSUFFICIENT_INFORMATION_RESPONSE not in answer
                            )

                            st.markdown(answer)
                            if answer_found:
                                render_confidence_badge(confidence)

                        except Exception as error:
                            st.error("An error occurred while generating the answer.")
                            st.code(str(error))
                            st.stop()

                    followup_questions = []
                    with st.spinner("Generating follow-up questions..."):
                        try:
                            followup_questions = generate_followup_questions(
                                question=user_question,
                                answer=answer,
                                source_documents=result.get("source_documents", []),
                            )
                        except Exception:
                            pass

                    render_followup_buttons(
                        followup_questions,
                        msg_index=len(st.session_state.messages),
                    )
                    render_sources(sources)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "type": "text",
                            "sources": sources,
                            "confidence_score": (
                                confidence if answer_found else None
                            ),
                            "followup_questions": followup_questions,
                        }
                    )

                    st.session_state.chat_history.append(
                        (user_question, answer)
                    )

                    st.session_state.last_evaluation = evaluation

            # Auto-save after every exchange.
            persist_current_conversation()


# =========================================================
# Retrieval evaluation tab
# =========================================================

with evaluation_tab:

    st.subheader("Retrieval Evaluation")
    st.caption("Metrics from the most recent question.")

    evaluation = st.session_state.last_evaluation

    if evaluation is None:
        st.info("Ask a question first to generate retrieval metrics.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Chunks Retrieved", evaluation["chunks_retrieved"])
        top_score = evaluation["top_relevance_score"]
        col2.metric("Top Relevance Score", f"{top_score:.3f}")
        col3.metric("Documents Used", len(evaluation["documents_used"]))

        st.markdown("### Answer Confidence")
        render_confidence_badge(top_score)
        st.caption(
            "Calibrated for all-MiniLM-L6-v2: "
            "≥ 0.45 = High · ≥ 0.28 = Medium · < 0.28 = Low. "
            "Based on cosine similarity between query and best-matching chunk."
        )

        st.divider()

        st.markdown("### Documents Used")
        if evaluation["documents_used"]:
            for doc_name in evaluation["documents_used"]:
                st.markdown(f"📄 `{doc_name}`")
        else:
            st.write("No documents were retrieved.")

        st.divider()

        st.markdown("### Retrieval Configuration")
        st.code(
            """Embedding model  : all-MiniLM-L6-v2 (local)
Vector store     : ChromaDB (cosine similarity)
Retrieval mode   : Hybrid — BM25 (40%) + MMR semantic (60%)
Final chunks     : 4 per sub-retriever
Candidate chunks : 12 (MMR candidate pool)
Chunk size       : 1500 characters
Chunk overlap    : 300 characters"""
        )
        st.caption(
            "BM25 handles exact keyword matches (author names, "
            "labels like 'P1', table references). "
            "MMR handles semantic similarity with diversity. "
            "Results are merged using Reciprocal Rank Fusion."
        )