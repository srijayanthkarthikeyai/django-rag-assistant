import os
import django
from sentence_transformers import SentenceTransformer

#Initialize Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')  # Use your project's settings name
django.setup()

from documents.models import Document

#Chunking helper function
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks of characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

#Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sample text to chunk and insert
raw_document = {
    "title": "PostgreSQL & pgvector Deep Dive",
    "content": (
        "PostgreSQL supports relational data alongside vector embeddings. "
        "With pgvector, users can store high-dimensional vectors and execute "
        "similarity searches using exact nearest neighbor or approximate nearest "
        "neighbor algorithms like HNSW and IVFFlat. HNSW provides fast search "
        "performance at the cost of higher memory usage during index creation."
    )
}

#Split into chunks
chunks = chunk_text(raw_document["content"], chunk_size=150, overlap=30)

#Batch encode
embeddings = model.encode(chunks)

#Create Document instances and bulk insert
docs_to_create = [
    Document(
        title=f"{raw_document['title']} (Part {i+1})",
        content=chunk,
        embedding=vector.tolist()
    )
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings))
]

Document.objects.bulk_create(docs_to_create)
print(f"Successfully saved {len(docs_to_create)} chunks.")