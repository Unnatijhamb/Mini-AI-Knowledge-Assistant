import fitz

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


def extract_pdf(uploaded_file):
    """
    Extract text directly from a Streamlit UploadedFile using PyMuPDF.

    Returns:
        page_documents: one LangChain Document per non-empty page
        page_count: total number of pages in the PDF
    """

    pdf_bytes = uploaded_file.getvalue()

    pdf_document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    page_documents = []
    page_count = len(pdf_document)

    try:
        for page_index in range(page_count):
            page = pdf_document.load_page(page_index)

            text = page.get_text("text").strip()

            if not text:
                continue

            document = Document(
                page_content=text,
                metadata={
                    "source": uploaded_file.name,
                    # Human-readable page numbering starts from 1.
                    "page": page_index + 1,
                },
            )

            page_documents.append(document)

    finally:
        pdf_document.close()

    return page_documents, page_count


def chunk_documents(documents):
    """
    Split page-level documents into smaller overlapping chunks.

    LangChain preserves the metadata of the parent Document,
    therefore every chunk keeps:
        {
            "source": filename,
            "page": page_number
        }

    chunk_size=1500 keeps section headings (e.g. "3.1 P1.") in the
    same chunk as the content below them, so abbreviation-based
    queries can retrieve the right passage.

    chunk_overlap=300 ensures context is not lost at chunk boundaries.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def process_pdf(uploaded_file):
    """
    Process one uploaded PDF.

    Returns:
        chunks
        statistics dictionary
    """

    page_documents, page_count = extract_pdf(uploaded_file)

    chunks = chunk_documents(page_documents)

    statistics = {
        "filename": uploaded_file.name,
        "pages": page_count,
        "chunks": len(chunks),
    }

    return chunks, statistics


def process_uploaded_files(uploaded_files):
    """
    Process multiple uploaded PDF files.

    Returns:
        all_chunks: chunks from all PDFs
        all_statistics: statistics for every uploaded PDF
    """

    all_chunks = []
    all_statistics = []

    for uploaded_file in uploaded_files:
        chunks, statistics = process_pdf(uploaded_file)

        all_chunks.extend(chunks)
        all_statistics.append(statistics)

    return all_chunks, all_statistics