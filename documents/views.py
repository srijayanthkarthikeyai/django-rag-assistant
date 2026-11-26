import io
import json
import os
import re
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
from django.db.models import Q
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
from pgvector.django import CosineDistance
from google import genai
from google.genai import types
from groq import Groq
from pypdf import PdfReader
from .models import Document, Session, Message

SIMILARITY_THRESHOLD = 0.65


# 🧠 Lazy API Client Helpers (Safe for Render builds and .env loading)
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")
    return genai.Client(api_key=api_key)


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")
    return Groq(api_key=api_key)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Fetches 384-d embeddings remotely via Google Gemini API."""
    client = get_gemini_client()
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=384)
    )
    return [emb.values for emb in result.embeddings]


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Splits plain text into overlapping sliding-window character chunks."""
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


def generate_chat_title(first_query: str) -> str:
    """Uses Groq to create a concise 3-5 word conversation title."""
    try:
        client = get_groq_client()
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "Generate a concise 3 to 5 word title summarizing the user query. Output ONLY the title text without quotes, punctuation, or preamble."
                },
                {"role": "user", "content": first_query}
            ],
            temperature=0.3,
            max_tokens=20
        )
        title = completion.choices[0].message.content.strip().strip('"').strip("'")
        return title[:50] if title else first_query[:35]
    except Exception:
        return first_query[:35] + ("..." if len(first_query) > 35 else "")


def retrieve_hybrid_documents(query_text: str, user, threshold: float = SIMILARITY_THRESHOLD, top_k: int = 4):
    """Combines remote vector cosine similarity with PostgreSQL full-text search."""
    query_vector = get_embeddings([query_text])[0]
    search_query = SearchQuery(query_text)

    base_qs = Document.objects.filter(Q(user=user) | Q(user__isnull=True))

    candidates = (
        base_qs.annotate(
            vector_distance=CosineDistance('embedding', query_vector),
            search_rank=SearchRank(SearchVector('content', 'title'), search_query)
        )
        .filter(Q(vector_distance__lte=threshold) | Q(search_rank__gte=0.05))
    )

    scored_results = []
    for doc in candidates:
        vector_sim = max(0.0, 1.0 - doc.vector_distance)
        keyword_score = float(doc.search_rank or 0.0)
        hybrid_score = (0.7 * vector_sim) + (0.3 * keyword_score)

        scored_results.append({
            "doc": doc,
            "hybrid_score": hybrid_score,
            "distance": round(doc.vector_distance, 4)
        })

    scored_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return scored_results[:top_k]


# 🖥️ Chat Interface (Accessible to guests and logged-in users)
@ensure_csrf_cookie
def chat_view(request):
    return render(request, 'documents/chat.html')


# 👤 Registration View
def register_view(request):
    if request.user.is_authenticated:
        return redirect('chat_ui')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('chat_ui')
    else:
        form = UserCreationForm()

    return render(request, 'documents/register.html', {'form': form})


# 🗄️ Session Endpoints
@login_required
def session_list_create_view(request):
    if request.method == "GET":
        sessions = Session.objects.filter(user=request.user).values('id', 'title', 'updated_at')
        return JsonResponse({"sessions": list(sessions)})

    elif request.method == "POST":
        new_session = Session.objects.create(user=request.user)
        return JsonResponse({"id": str(new_session.id), "title": new_session.title})

    return JsonResponse({"error": "Method not allowed"}, status=405)


@login_required
def session_detail_view(request, session_id):
    session = get_object_or_404(Session, id=session_id, user=request.user)

    if request.method == "GET":
        messages = list(session.messages.values('role', 'content', 'sources', 'created_at'))
        return JsonResponse({
            "id": str(session.id),
            "title": session.title,
            "messages": messages
        })

    elif request.method == "DELETE":
        session.delete()
        return JsonResponse({"message": "Session deleted"})

    return JsonResponse({"error": "Method not allowed"}, status=405)


# 🌊 Streaming RAG View
@login_required
def rag_stream_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        query_text = data.get("query", "").strip()
        session_id = data.get("session_id")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not query_text:
        return JsonResponse({"error": "Query cannot be empty"}, status=400)

    if session_id:
        try:
            session = Session.objects.get(id=session_id, user=request.user)
        except (Session.DoesNotExist, ValueError):
            session = Session.objects.create(user=request.user)
    else:
        session = Session.objects.create(user=request.user)

    # 🏷️ Auto-generate title using LLM on the first message
    if session.messages.count() == 0:
        session.title = generate_chat_title(query_text)
        session.save()

    Message.objects.create(session=session, role='user', content=query_text)

    # 🔍 Hybrid retrieval
    try:
        hybrid_results = retrieve_hybrid_documents(query_text, request.user)
    except Exception:
        hybrid_results = []

    sources_data = [
        {"title": item["doc"].title, "distance": item["distance"]}
        for item in hybrid_results
    ]
    context_text = "\n\n".join([
        f"[{item['doc'].title}]\n{item['doc'].content}"
        for item in hybrid_results
    ])

    past_messages = list(session.messages.exclude(id=session.messages.last().id).order_by('created_at'))
    groq_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful knowledge assistant. Use the provided context to answer the question. "
                "If no context is provided or relevant, answer using your general knowledge politely."
            )
        }
    ]
    for msg in past_messages:
        groq_messages.append({"role": msg.role, "content": msg.content})

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query_text}" if context_text else query_text
    groq_messages.append({"role": "user", "content": user_prompt})

    def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

        full_response = []
        try:
            client = get_groq_client()
            groq_stream = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=groq_messages,
                temperature=0.2,
                stream=True
            )

            for chunk in groq_stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_response.append(delta)
                    yield f"data: {json.dumps({'type': 'token', 'token': delta})}\n\n"

        except Exception as e:
            error_msg = f"\n\n*(Error generating response: {str(e)})*"
            full_response.append(error_msg)
            yield f"data: {json.dumps({'type': 'token', 'token': error_msg})}\n\n"

        complete_content = "".join(full_response)
        Message.objects.create(
            session=session,
            role='assistant',
            content=complete_content,
            sources=sources_data
        )

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


# 📤 Document Upload View
@login_required
def upload_document_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()

    try:
        if ext == "txt":
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif ext == "pdf":
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            return JsonResponse({"error": "Only .txt and .pdf supported."}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Failed to extract text: {str(e)}"}, status=500)

    chunks = chunk_text(raw_text, chunk_size=300, overlap=50)
    if not chunks:
        return JsonResponse({"error": "No readable text extracted"}, status=400)

    try:
        embeddings = get_embeddings(chunks)
    except Exception as e:
        return JsonResponse({"error": f"Embedding API failed: {str(e)}"}, status=502)

    docs = [
        Document(
            user=request.user,
            title=f"{filename} (Part {i})",
            content=c,
            embedding=e
        )
        for i, (c, e) in enumerate(zip(chunks, embeddings), start=1)
    ]
    Document.objects.bulk_create(docs)

    return JsonResponse({
        "message": f"Successfully indexed '{filename}'!",
        "chunks_indexed": len(docs)
    })


# 📁 Document Management Views
@login_required
@require_http_methods(["GET"])
def document_list_view(request):
    """Returns unique files and chunk counts for the active user."""
    user_docs = Document.objects.filter(user=request.user).values('title')
    file_summary = {}
    for item in user_docs:
        match = re.match(r"^(.*?)(?:\s+\(Part\s+\d+\))?$", item['title'])
        filename = match.group(1) if match else item['title']
        file_summary[filename] = file_summary.get(filename, 0) + 1

    documents = [
        {"filename": name, "chunks": count}
        for name, count in file_summary.items()
    ]
    return JsonResponse({"documents": documents})


@login_required
@require_http_methods(["POST"])
def document_delete_view(request):
    """Deletes all chunks of an uploaded file."""
    try:
        data = json.loads(request.body)
        filename = data.get("filename")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not filename:
        return JsonResponse({"error": "Filename required"}, status=400)

    deleted_count, _ = Document.objects.filter(
        user=request.user,
        title__startswith=filename
    ).delete()

    return JsonResponse({
        "message": f"Successfully deleted '{filename}'",
        "chunks_removed": deleted_count
    })