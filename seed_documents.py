import os
import django
from sentence_transformers import SentenceTransformer

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from documents.models import Document

#Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sample data to insert
sample_docs = [
    {
        "title": "PostgreSQL Overview",
        "content": "PostgreSQL is a powerful, open-source object-relational database system."
    },
    {
        "title": "Django Framework",
        "content": "Django is a high-level Python web framework that encourages rapid development."
    }
]

#Generate embeddings and save to the database
for item in sample_docs:
    vector = model.encode(item["content"]).tolist()
    Document.objects.create(
        title=item["title"],
        content=item["content"],
        embedding=vector
    )