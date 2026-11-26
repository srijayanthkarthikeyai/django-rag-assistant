import os
import django
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

#Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from documents.models import Document

#Chunking helper function
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Splits text into overlapping windows of characters."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# File reader supporting .txt and .pdf
def extract_text_from_file(file_path: Path) -> str:
    """Extracts raw text content from a given file path."""
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif suffix == ".pdf":
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return ""

def main():
    #Locate data folder
    data_dir = Path("data")
    if not data_dir.exists():
        print("Directory 'data/' does not exist. Please create it and add files.")
        return

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    supported_extensions = {".txt", ".pdf"}
    files_to_process = [
        f for f in data_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not files_to_process:
        print("No supported files (.txt, .pdf) found in data/.")
        return

    all_chunks = []
    chunk_metadata = []

    print(f"Found {len(files_to_process)} file(s). Extracting and chunking...")

    for file_path in files_to_process:
        raw_text = extract_text_from_file(file_path)
        if not raw_text.strip():
            print(f"Skipping empty file: {file_path.name}")
            continue

        file_chunks = chunk_text(raw_text, chunk_size=300, overlap=50)
        for idx, chunk in enumerate(file_chunks, start=1):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "title": f"{file_path.name} (Part {idx})",
                "content": chunk
            })

    if not all_chunks:
        print("No readable text chunks extracted.")
        return

    #Batch compute vector embeddings
    print(f"Embedding {len(all_chunks)} chunks in batch...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    #Prepare Django models and bulk insert into PostgreSQL
    docs_to_create = [
        Document(
            title=meta["title"],
            content=meta["content"],
            embedding=emb.tolist()
        )
        for meta, emb in zip(chunk_metadata, embeddings)
    ]

    print("Inserting chunks into PostgreSQL database...")
    Document.objects.bulk_create(docs_to_create)
    print(f"Successfully ingested and indexed {len(docs_to_create)} chunks into PostgreSQL!")

if __name__ == "__main__":
    main()