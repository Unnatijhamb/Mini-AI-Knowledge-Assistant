import json
import os
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document


CONVERSATIONS_DIRECTORY = "./data/conversations"


def ensure_conversations_directory():
    os.makedirs(CONVERSATIONS_DIRECTORY, exist_ok=True)


def generate_conversation_id() -> str:
    return str(uuid.uuid4())


def collection_name_for(conv_id: str) -> str:
    """
    Derive a stable, unique ChromaDB collection name from a
    conversation ID.  Chroma collection names must be 3-63
    characters and match [a-zA-Z0-9_-]+.
    """
    return f"dm_{conv_id.replace('-', '')[:24]}"


def get_conversation_title(messages: list, document_names: list) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            question = msg["content"]
            return question[:55] + ("..." if len(question) > 55 else "")

    if document_names:
        names = ", ".join(document_names[:2])
        if len(document_names) > 2:
            names += f" +{len(document_names) - 2} more"
        return names

    return "Untitled Conversation"


def serialize_chunks(chunks: List[Document]) -> list:
    """Convert LangChain Document objects to JSON-serialisable dicts."""
    return [
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]


def deserialize_chunks(raw: list) -> List[Document]:
    """Restore LangChain Document objects from saved dicts."""
    return [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {}),
        )
        for item in raw
    ]


def save_conversation(
    conv_id: str,
    collection_name: str,
    messages: list,
    document_names: list,
    document_stats: list,
    document_summaries: dict,
    chunks: List[Document],
) -> bool:
    """
    Persist a full conversation snapshot to disk.

    chunks are serialised as plain dicts so they can be restored
    later without re-uploading the original PDFs.

    Returns True on success.
    """
    ensure_conversations_directory()

    title = get_conversation_title(messages, document_names)
    now = datetime.now().isoformat()

    data = {
        "id": conv_id,
        "collection_name": collection_name,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "document_names": document_names,
        "document_stats": document_stats,
        "document_summaries": document_summaries,
        "messages": messages,
        "chunks": serialize_chunks(chunks),
    }

    path = os.path.join(CONVERSATIONS_DIRECTORY, f"{conv_id}.json")

    # Preserve original created_at and chunks on subsequent saves
    # (chunks never change after the initial document processing).
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            data["created_at"] = existing.get("created_at", now)
            # Keep the saved chunks rather than re-serialising every time.
            if "chunks" in existing:
                data["chunks"] = existing["chunks"]
        except Exception:
            pass

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def save_chunks_initial(
    conv_id: str,
    collection_name: str,
    chunks: List[Document],
    document_names: list,
    document_stats: list,
    document_summaries: dict,
) -> bool:
    """
    Write the initial conversation record (no messages yet) with
    the serialised chunks.  Called once at document-processing time
    so subsequent saves don't need to re-serialise the chunks.
    """
    return save_conversation(
        conv_id=conv_id,
        collection_name=collection_name,
        messages=[],
        document_names=document_names,
        document_stats=document_stats,
        document_summaries=document_summaries,
        chunks=chunks,
    )


def list_conversations() -> List[dict]:
    """
    Return metadata for all saved conversations, newest first.
    """
    ensure_conversations_directory()

    conversations = []

    for filename in os.listdir(CONVERSATIONS_DIRECTORY):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(CONVERSATIONS_DIRECTORY, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            user_messages = [
                m for m in data.get("messages", [])
                if m.get("role") == "user"
            ]

            conversations.append(
                {
                    "id": data["id"],
                    "collection_name": data.get("collection_name", ""),
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "document_names": data.get("document_names", []),
                    "message_count": len(user_messages),
                }
            )
        except Exception:
            continue

    conversations.sort(key=lambda x: x["updated_at"], reverse=True)

    return conversations


def load_conversation(conv_id: str) -> Optional[dict]:
    """
    Load the full saved conversation including serialised chunks.
    Returns None if the conversation does not exist.
    """
    path = os.path.join(CONVERSATIONS_DIRECTORY, f"{conv_id}.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_conversation(conv_id: str) -> bool:
    """
    Delete the saved conversation JSON file.
    The caller is responsible for also deleting the Chroma collection.
    """
    path = os.path.join(CONVERSATIONS_DIRECTORY, f"{conv_id}.json")

    if not os.path.exists(path):
        return False

    try:
        os.remove(path)
        return True
    except Exception:
        return False


def format_display_date(iso_string: str) -> str:
    if not iso_string:
        return ""

    try:
        dt = datetime.fromisoformat(iso_string)
        today = datetime.now().date()

        if dt.date() == today:
            return "Today"

        delta = (today - dt.date()).days

        if delta == 1:
            return "Yesterday"

        if delta < 7:
            return dt.strftime("%A")

        return dt.strftime("%d %b %Y")

    except Exception:
        return ""