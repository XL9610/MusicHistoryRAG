from sentence_transformers import SentenceTransformer
import chromadb
from config import (CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DEVICE, STOP_WORDS, SCORE_THRESHOLD,
                    MAX_CHUNKS_PER_CHAPTER, N_RESULTS)
import re

# Load model and database once
model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

# Filler phrases to strip from queries before retrieval
FILLER_PHRASES = [
    "expand on", "tell me about", "tell me more about",
    "explain more about", "explain", "can you explain",
    "please explain", "what is", "what are", "what was",
    "describe", "discuss", "elaborate on",
]


def normalize_query(query):
    """Strip filler phrases to get a retrieval-friendly query."""
    q = query.lower().strip()
    for phrase in FILLER_PHRASES:
        if q.startswith(phrase):
            q = q[len(phrase):].strip()
    return q if q else query


def extract_keywords(query):
    """Extract meaningful keywords using proper tokenization."""
    tokens = re.findall(r"\b\w+\b", query.lower())
    return [t for t in tokens if len(t) > 3 and t not in STOP_WORDS]


def retrieve(query):
    """Full retrieval pipeline: normalize → vector search → keyword rerank → filter."""

    # 1. Normalize query for retrieval
    retrieval_query = normalize_query(query)
    query_vector = model.encode(retrieval_query).tolist()
    keywords = extract_keywords(retrieval_query)

    # 2. Always start with vector search (wide net)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )

    # 3. Rerank: semantic score + keyword bonus
    candidates = []
    for doc, meta, dist in zip(results["documents"][0],
                               results["metadatas"][0],
                               results["distances"][0]):
        semantic_score = 1 - dist
        doc_tokens = set(re.findall(r"\b\w+\b", doc.lower()))
        matches = sum(1 for kw in keywords if kw in doc_tokens)
        coverage = matches / len(keywords) if keywords else 0
        keyword_bonus = 0.15 * coverage
        final_score = round(semantic_score + keyword_bonus, 3)
        candidates.append((doc, meta, final_score))

    # Sort by final score descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    # 4. Dynamic threshold
    if candidates:
        max_score = candidates[0][2]
        threshold = max(0.08, max_score * 0.5)
    else:
        threshold = SCORE_THRESHOLD

    # 5. Filter: threshold + chapter diversity
    filtered_docs = []
    filtered_metas = []
    chapter_count = {}

    for doc, meta, score in candidates:
        if score < threshold:
            continue

        ch = meta["chapter_num"]
        chapter_count[ch] = chapter_count.get(ch, 0) + 1
        if chapter_count[ch] > MAX_CHUNKS_PER_CHAPTER:
            continue

        filtered_docs.append(doc)
        filtered_metas.append((meta, score))

        if len(filtered_docs) >= 5:
            break

    # 6. Fallback: if nothing passed, retry with pure vector and lower threshold
    if not filtered_docs:
        for doc, meta, score in candidates[:5]:
            if score >= 0.08:
                filtered_docs.append(doc)
                filtered_metas.append((meta, score))

    return filtered_docs, filtered_metas, retrieval_query
