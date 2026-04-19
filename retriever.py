from sentence_transformers import SentenceTransformer
import chromadb
from config import (CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DEVICE, STOP_WORDS, SCORE_THRESHOLD,
                    MAX_CHUNKS_PER_CHAPTER, N_RESULTS, SUMMARY_COLLECTION_NAME, TOP_CHAPTERS)
import re
from rank_bm25 import BM25Okapi

# Load model and database once
model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chunk_collection = client.get_collection(COLLECTION_NAME)
summary_collection = client.get_collection(SUMMARY_COLLECTION_NAME)

# Filler phrases to strip from queries before retrieval
FILLER_PHRASES = [
    "expand on", "tell me about", "tell me more about",
    "explain more about", "explain", "can you explain",
    "please explain", "what is", "what are", "what was",
    "describe", "discuss", "elaborate on",
]

def get_top_chapters(query_vector):
    """Get top relevant chapters based on query vector first from the summary collection."""
    results = summary_collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_CHAPTERS,
        include=["metadatas", "distances"]
    )
    #Creating top chapter number list
    top_chapters = []

    for meta in results["metadatas"][0]:
        top_chapters.append(meta["chapter_num"])

    return set(top_chapters)


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

def get_phrase_bonus(query, doc):
    """Give extra bonus if the exact query phrase appears in the document"""
    q=query.lower().strip()
    d = doc.lower()

    if q in d:
        return 0.2

    return 0.0

def retrieve(query):
    """Hierarchical retrieval: summary retrieval -> chapter-constrained chunk retrieval -> rerank -> filter."""

    # 1. Normalize query for retrieval
    retrieval_query = normalize_query(query)
    query_vector = model.encode(retrieval_query).tolist()
    keywords = extract_keywords(retrieval_query)

    # 2. First stage retrieval: getting the most relevant chapters
    top_chapters = get_top_chapters(query_vector)

    # 3. Secondary retrieval: search chunk candidates from the chunk collection
    results = chunk_collection.query(
        query_embeddings=[query_vector],
        n_results=20,
        where={"chapter_num": {"$in": list(top_chapters)}},
        include=["documents", "metadatas", "distances"]
    )


    candidates = []
    for doc, meta, dist in zip(results["documents"][0],
                               results["metadatas"][0],
                               results["distances"][0]):
        semantic_score = 1 - dist
        doc_tokens = set(re.findall(r"\b\w+\b", doc.lower()))
        matches = sum(1 for kw in keywords if kw in doc_tokens)
        coverage = matches / len(keywords) if keywords else 0
        phrase_bonus = get_phrase_bonus(retrieval_query, doc)
        final_score = round(0.7 * semantic_score + 0.2 * coverage + phrase_bonus, 3)
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
                filtered_docs.append(doc)
                filtered_metas.append((meta, score))

    return filtered_docs, filtered_metas, retrieval_query
