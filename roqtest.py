from documents.models import Document

# 1. 📊 Count total stored chunks
print(f"Total documents: {Document.objects.count()}")

# 2. 🔍 Filter for records ingested from your PDF (or list the latest 6)
pdf_docs = Document.objects.filter(title__icontains=".pdf")
# If your titles didn't include the extension, view the last 6 created:
# pdf_docs = Document.objects.all().order_by('-id')[:6]

# 3. 📄 Print the titles, chunk sizes, and preview text
for doc in pdf_docs:
    print(f"\nID: {doc.id} | Title: {doc.title}")
    print(f"Content preview: {doc.content[:120]}...")
    print(f"Vector dimensions: {len(doc.embedding)}")