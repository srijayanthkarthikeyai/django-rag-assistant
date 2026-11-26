import os
import django
from sentence_transformers import SentenceTransformer
from pgvector.django import CosineDistance
from groq import Groq

#Initialize Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from documents.models import Document

#Retrieve relevant documents
model = SentenceTransformer('all-MiniLM-L6-v2')
SIMILARITY_THRESHOLD = 0.6

#Filter records so only sufficiently close matches are retrieved
results = Document.objects.annotate(
    distance=CosineDistance('embedding', query_vector)
).filter(
    distance__lte=SIMILARITY_THRESHOLD
).order_by('distance')[:3]

#Format the retrieved context
context_text = "\n".join([f"- {doc.title}: {doc.content}" for doc in results])

#Initialize Groq client
client = Groq(api_key=GROQ_KEY)

#Send context and query to the LLM
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question using only the provided context."
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query_text}"
        }
    ],
    temperature=0.2
)

#final answer
print("\n--- LLM Response ---")
print(response.choices[0].message.content)