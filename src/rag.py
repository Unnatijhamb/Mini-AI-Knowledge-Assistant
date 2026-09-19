import re
from typing import List, Tuple

import numpy as np

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_groq import ChatGroq

from src.vectorstore import (
    create_hybrid_retriever,
    get_top_relevance_score,
    get_embedding_model,
)


GROQ_MODEL = "openai/gpt-oss-120b"

INSUFFICIENT_INFORMATION_RESPONSE = (
    "I couldn't find enough information in the uploaded "
    "documents to answer this."
)

# Keywords that indicate the user wants a diagram / flowchart.
_DIAGRAM_KEYWORDS = [
    "flowchart", "flow chart", "flow-chart",
    "diagram", "visualize", "visualise",
    "draw a", "draw the", "chart", "map out",
    "create a visual", "show me a visual",
    "process flow", "workflow",
]


def is_diagram_request(question: str) -> bool:
    """
    Return True when the question is asking for a diagram or flowchart.
    """
    q = question.lower()
    return any(kw in q for kw in _DIAGRAM_KEYWORDS)


# ---------------------------------------------------------
# Query rewriting prompt
# ---------------------------------------------------------

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
You are a search query rewriting assistant for a
Retrieval-Augmented Generation system.

Given the previous conversation and the newest user question,
rewrite the newest question into ONE clear, standalone search
query.

The rewritten query must:
- preserve the user's original meaning
- resolve references such as "it", "they", "this", or "that"
  using the conversation history
- include important keywords from the user's question
- be optimized for semantic retrieval
- NOT answer the question
- NOT add facts that were not provided

Conversation history:
{chat_history}

Newest user question:
{question}

Standalone search query:
""".strip(),
)


# ---------------------------------------------------------
# Grounded-answer prompt
# ---------------------------------------------------------

ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""
You are DocuMind, an AI Knowledge Assistant that answers
questions using uploaded documents.

You MUST follow these rules:

1. Answer using ONLY the information contained in the provided
   document context.

2. Do NOT use outside knowledge.

3. Do NOT invent facts, values, names, dates, conclusions,
   definitions, or explanations that are not supported by the
   context.

4. If the context does not contain enough information to answer
   the question confidently, respond with EXACTLY:

"{INSUFFICIENT_INFORMATION_RESPONSE}"

5. If an answer is available, provide a clear, useful,
   well-structured response.

6. Do not invent source citations inside the answer.
   The application displays verified source information
   separately.

7. If different documents contain different information,
   explain the distinction rather than combining them into an
   unsupported conclusion.

DOCUMENT CONTEXT:
-----------------
{{context}}
-----------------

USER QUESTION:
{{question}}

ANSWER:
""".strip(),
)


# ---------------------------------------------------------
# Document summary prompt
# ---------------------------------------------------------

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "filename"],
    template="""
You are a document analyst. Read the following excerpts from
"{filename}" and write a concise 3-sentence summary covering:
1. What the document is about
2. Key content, methods, or findings
3. Who would benefit from reading it

Be specific and informative. Do not start with "This document".

DOCUMENT EXCERPTS:
{context}

SUMMARY:
""".strip(),
)


# ---------------------------------------------------------
# Mermaid diagram prompt
# ---------------------------------------------------------

MERMAID_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are DocuMind, an AI Knowledge Assistant that creates
Mermaid.js diagrams from uploaded documents.

The user wants a diagram about: {question}

Using ONLY the information in the document context below,
generate a valid Mermaid.js flowchart diagram.

Rules:
1. Output ONLY valid Mermaid.js syntax — no explanation,
   no markdown code fences (no ```), no introductory text,
   no trailing text.
2. Start with "flowchart TD" or "flowchart LR"
   (choose based on what best represents the content).
3. Use actual content, steps, or relationships described in
   the context.
4. Keep node labels concise (under 8 words each).
5. Use double-quoted labels in square brackets: A["Label"]
6. Do NOT use parentheses () in node labels — replace them
   with square brackets [] or angle brackets <>.
7. If the context does not contain enough information to draw
   a meaningful diagram, output exactly:
   INSUFFICIENT_INFORMATION

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

Mermaid diagram (no fences, start directly with flowchart):
""".strip(),
)


# ---------------------------------------------------------
# Follow-up questions prompt
# ---------------------------------------------------------

FOLLOWUP_PROMPT = PromptTemplate(
    input_variables=["context", "question", "answer"],
    template="""
You are helping a user explore a document collection.
Using the document excerpts below, suggest 3 follow-up questions.

CRITICAL RULE: Only suggest questions that can be DIRECTLY and
FULLY answered using the DOCUMENT EXCERPTS provided. Do NOT
suggest questions about topics not present in the excerpts.
If fewer than 3 answerable questions exist, output only those.

Rules:
- Output ONLY the questions, one per line, numbered 1. 2. 3.
- Each question must be under 12 words
- Every question MUST be answerable from the excerpts below
- Do NOT repeat or rephrase the original question
- Do NOT add any explanation or preamble

DOCUMENT EXCERPTS:
{context}

Original question: {question}
Answer given: {answer}

Follow-up questions (only what the excerpts can answer):
""".strip(),
)


def create_llm():
    """
    Create the LLM used for query rewriting, answer generation,
    and document summarisation.

    Uses Groq's inference API.
    GROQ_API_KEY is read automatically from the .env file.
    """
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
    )


def create_conversational_rag_chain(
    vectorstore,
    all_chunks: List[Document],
    selected_sources,
):
    """
    Build the conversational RAG chain with hybrid retrieval.

    Pipeline:
        conversation history
                ↓
        query rewriting
                ↓
        standalone retrieval query
                ↓
        Hybrid BM25 + MMR retrieval
        (exact keyword + semantic similarity)
                ↓
        grounded answer generation
    """
    llm = create_llm()

    retriever = create_hybrid_retriever(
        vectorstore=vectorstore,
        all_chunks=all_chunks,
        selected_sources=selected_sources,
        k=4,
        fetch_k=12,
    )

    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
    )

    document_answer_chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=ANSWER_PROMPT,
    )

    chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=document_answer_chain,
        question_generator=question_generator,
        return_source_documents=True,
        rephrase_question=False,
        get_chat_history=format_chat_history,
    )

    return chain


def generate_document_summary(
    chunks: List[Document],
    filename: str,
) -> str:
    """
    Generate a 3-sentence summary of a document using its
    first few chunks. Called once per document at upload time
    and displayed in the sidebar.
    """
    llm = create_llm()

    # Use up to the first 5 chunks for the summary.
    sample_chunks = chunks[:5]
    context = "\n\n---\n\n".join(
        c.page_content for c in sample_chunks
    )

    chain = LLMChain(llm=llm, prompt=SUMMARY_PROMPT)

    try:
        result = chain.invoke(
            {"context": context, "filename": filename}
        )
        return result.get("text", "").strip()
    except Exception:
        return "Summary could not be generated."


def generate_diagram(
    vectorstore,
    question: str,
    selected_sources,
    all_chunks: List[Document] = None,
) -> dict:
    """
    Retrieve relevant chunks and generate a Mermaid.js diagram
    grounded in the document content.

    Returns:
        {
            "mermaid_code": str or None,
            "sources": list,
            "is_insufficient": bool,
        }
    """
    retriever = create_hybrid_retriever(
        vectorstore=vectorstore,
        all_chunks=all_chunks or [],
        selected_sources=selected_sources,
        k=6,
        fetch_k=18,
    )

    source_documents = retriever.invoke(question)

    if not source_documents:
        return {
            "mermaid_code": None,
            "sources": [],
            "is_insufficient": True,
        }

    context = "\n\n---\n\n".join(
        doc.page_content for doc in source_documents
    )

    llm = create_llm()
    chain = LLMChain(llm=llm, prompt=MERMAID_PROMPT)

    try:
        result = chain.invoke(
            {"context": context, "question": question}
        )
        raw = result.get("text", "").strip()
    except Exception:
        return {
            "mermaid_code": None,
            "sources": unique_sources(source_documents),
            "is_insufficient": True,
        }

    if not raw or "INSUFFICIENT_INFORMATION" in raw:
        return {
            "mermaid_code": None,
            "sources": unique_sources(source_documents),
            "is_insufficient": True,
        }

    # Strip markdown fences in case the model adds them anyway.
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    return {
        "mermaid_code": raw,
        "sources": unique_sources(source_documents),
        "is_insufficient": False,
    }


def generate_followup_questions(
    question: str,
    answer: str,
    source_documents: List[Document] = None,
) -> List[str]:
    """
    Generate up to 3 follow-up questions grounded strictly in
    the retrieved document chunks, so every suggestion is
    guaranteed to be answerable from the knowledge base.

    Returns an empty list if the answer was insufficient or
    if the LLM call fails.
    """
    if INSUFFICIENT_INFORMATION_RESPONSE in answer:
        return []

    # Build context from the same chunks that produced the answer
    # so the LLM can only propose questions it has evidence for.
    context = ""
    if source_documents:
        context = "\n\n---\n\n".join(
            doc.page_content for doc in source_documents[:4]
        )

    if not context:
        return []

    llm = create_llm()
    chain = LLMChain(llm=llm, prompt=FOLLOWUP_PROMPT)

    try:
        result = chain.invoke(
            {
                "context": context,
                "question": question,
                "answer": answer,
            }
        )
        raw = result.get("text", "").strip()
    except Exception:
        return []

    questions = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if cleaned and len(cleaned) > 5:
            questions.append(cleaned)

    return questions[:3]


def find_best_sentence(answer: str, chunk_text: str) -> str:
    """
    Return the sentence from chunk_text that is most semantically
    similar to the answer, using the already-loaded local
    embedding model (no API call — runs entirely on device).

    Sentence similarity is computed as dot product of L2-
    normalised embeddings (equivalent to cosine similarity).
    """
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n", chunk_text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 25]

    if not sentences:
        return ""

    try:
        model = get_embedding_model()
        answer_emb = np.array(model.embed_query(answer))
        sent_embs = np.array(model.embed_documents(sentences))
        # Embeddings are already L2-normalised, so dot = cosine.
        scores = sent_embs @ answer_emb
        best_idx = int(np.argmax(scores))
        return sentences[best_idx]
    except Exception:
        return ""


def get_highlighted_sources(
    answer: str,
    source_documents: List[Document],
) -> List[dict]:
    """
    Build a deduplicated source list where each entry also
    carries the single sentence from that chunk that best
    matches the answer — used for exact-quote highlighting in
    the UI.
    """
    sources = []
    seen = set()

    for doc in source_documents:
        filename = doc.metadata.get("source", "Unknown document")
        page_number = doc.metadata.get("page", "Unknown")
        source_key = (filename, page_number)

        if source_key in seen:
            continue
        seen.add(source_key)

        best_sentence = find_best_sentence(answer, doc.page_content)

        sources.append(
            {
                "source": filename,
                "page": page_number,
                "best_sentence": best_sentence,
            }
        )

    return sources


def format_chat_history(
    chat_history: List[Tuple[str, str]]
):
    """
    Convert stored (user, assistant) tuples to plain text for
    the query-rewriting prompt.
    """
    if not chat_history:
        return "No previous conversation."

    formatted_messages = []
    for user_message, assistant_message in chat_history:
        formatted_messages.append(f"User: {user_message}")
        formatted_messages.append(f"Assistant: {assistant_message}")

    return "\n".join(formatted_messages)


def unique_sources(source_documents: List[Document]):
    """
    Convert retrieved documents into unique page-aware citations
    (without sentence highlighting — used for diagram results).
    """
    sources = []
    seen = set()

    for document in source_documents:
        filename = document.metadata.get("source", "Unknown document")
        page_number = document.metadata.get("page", "Unknown")
        source_key = (filename, page_number)

        if source_key in seen:
            continue

        seen.add(source_key)
        sources.append({"source": filename, "page": page_number})

    return sources


def get_documents_used(source_documents: List[Document]):
    """
    Return unique document filenames represented in the
    retrieved context.
    """
    documents_used = []
    seen = set()

    for document in source_documents:
        filename = document.metadata.get("source", "Unknown document")
        if filename not in seen:
            seen.add(filename)
            documents_used.append(filename)

    return documents_used


def ask_question(
    vectorstore,
    question,
    chat_history,
    selected_sources,
    all_chunks: List[Document] = None,
):
    """
    Execute the complete RAG pipeline with hybrid retrieval.

    Returns:
        {
            "answer": str,
            "sources": list,          # includes best_sentence per source
            "source_documents": list,
            "evaluation": {
                "chunks_retrieved": int,
                "top_relevance_score": float,
                "documents_used": list
            }
        }
    """
    chain = create_conversational_rag_chain(
        vectorstore=vectorstore,
        all_chunks=all_chunks or [],
        selected_sources=selected_sources,
    )

    result = chain.invoke(
        {
            "question": question,
            "chat_history": chat_history,
        }
    )

    answer = result.get(
        "answer", INSUFFICIENT_INFORMATION_RESPONSE
    ).strip()

    source_documents = result.get("source_documents", [])

    # Use highlighted sources (includes best_sentence field).
    sources = get_highlighted_sources(answer, source_documents)
    documents_used = get_documents_used(source_documents)

    top_relevance_score = get_top_relevance_score(
        vectorstore=vectorstore,
        query=question,
        selected_sources=selected_sources,
    )

    evaluation = {
        "chunks_retrieved": len(source_documents),
        "top_relevance_score": top_relevance_score,
        "documents_used": documents_used,
    }

    return {
        "answer": answer,
        "sources": sources,
        "source_documents": source_documents,
        "evaluation": evaluation,
    }