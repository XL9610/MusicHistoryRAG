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

#Retrieval strategy selection
RETRIEVAL_STRATEGY = "hybrid"
BM25_TOP_K = 20
VECTOR_TOP_K = 20
FINAL_TOP_K = 5
RRF_K = 60

#Load all chunks from ChromaDB for BM25 indexing
all_chunk_data = chunk_collection.get(include=["documents", "metadatas"])

ALL_DOCS = all_chunk_data["documents"]
ALL_METAS = all_chunk_data["metadatas"]
ALL_IDS = all_chunk_data["ids"]

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

def token_for_bm25(text):
    """ Tokenize a text into a list of tokens for BM25 retrieval."""
    return re.findall(r"\b\w+\b", text.lower())

#Build the BM25 index
#Does not need to be rebuilt every single time a query happens
BM25_TOKENIZED_DOCS = [token_for_bm25(doc) for doc in ALL_DOCS]
BM25_INDEX = BM25Okapi(BM25_TOKENIZED_DOCS)

def get_phrase_bonus(query, doc):
    """Give extra bonus if the exact query phrase appears in the document"""
    q=query.lower().strip()
    d = doc.lower()

    if q in d:
        return 0.2

    return 0.0

def retrieve_vector_candidates(retrieval_query, query_vector, keywords):
    """Retrieve candidates using the existing hierarchical vector search.

    Step 1: Use chapter summary embeddings to select top chapters.
    Step 2: Search chunk embeddings only within those selected chapters.
    Step 3: Add lightweight keyword and phrase bonuses.
    """
    #First stage: get the top relevant chapters according to the summary
    top_chapters = get_top_chapters(query_vector)

    #Second stage: search chunks within the top chapters
    results = chunk_collection.query(
        query_embeddings=[query_vector],
        n_results=VECTOR_TOP_K,
        where={"chapter_num": {"$in": list(top_chapters)}},
        include=["documents", "metadatas", "distances"]
    )

    candidates = []

    for doc, meta, dist in zip(results["documents"][0],results["metadatas"][0],results["distances"][0]):
        # ChromaDB returns cosine distance, so similarity is 1 - distance.
        semantic_score = 1 - dist
        #Count how many query keywords appear in the chunk
        doc_tokens = set(re.findall(r"\b\w+\b", doc.lower()))
        matches = sum(1 for kw in keywords if kw in doc_tokens)
        coverage = matches / len(keywords) if keywords else 0

        #Give a bonus if the full query appears in the chunk
        phrase_bonus = get_phrase_bonus(retrieval_query, doc)

        # Combine semantic score, keyword coverage, and phrase bonus.
        final_score = round(
            0.7 * semantic_score + 0.2 * coverage + phrase_bonus,
            3
        )

        candidates.append({
            "id": f"{meta['chapter_num']}_{meta['chunk_index']}",
            "doc": doc,
            "meta": meta,
            "score": final_score,
            "source": "vector"
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def retrieve_bm25_candidates(retrieval_query):
    """Retrieve candidates using BM25 lexical search.
        Strong for exact terms, names, genres, and historical labels.
        Unlike the vector path, this search runs over all chunks and is not
        limited by the chapter summary layer.
        """
    #Tokenize the user query in the same way as the documents
    query_tokens = token_for_bm25(retrieval_query)

    #Compute BM25 scores for all chunks
    scores = BM25_INDEX.get_scores(query_tokens)

    #Take the top BM25 candidates by raw BM25 score
    ranked_indexes = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:BM25_TOP_K]

    candidates = []

    #Normalize BM25 scores to roughly 0-1
    max_score = max([scores[i] for i in ranked_indexes], default=0)

    for i in ranked_indexes:
        raw_score = scores[i]

        #Ignore chunks with no BM25 matches
        if raw_score <=0:
            continue

        normalized_score = raw_score / max_score if max_score > 0 else 0

        meta = ALL_METAS[i]

        candidates.append({
            "id": f"{meta['chapter_num']}_{meta['chunk_index']}",
            "doc": ALL_DOCS[i],
            "meta": meta,
            "score": round(normalized_score, 3),
            "source": "bm25"
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF is useful because vector scores and BM25 scores are not directly
    comparable. Instead of adding raw scores, RRF uses ranking positions.
    """
    fused = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            item_id = item["id"]

            if item_id not in fused:
                fused[item_id] = {
                    "doc": item["doc"],
                    "meta": item["meta"],
                    "score": 0.0,
                    "sources": []
                }

            #Add RRF score based on rank
            fused[item_id]["score"] += 1.0 / (k + rank)
            fused[item_id]["sources"].append(item["source"])

    candidates = []

    for item_id, item in fused.items():
        candidates.append({
            "id": f"{item_id}",
            "doc": item["doc"],
            "meta": item["meta"],
            "score": round(item["score"], 4),
            "source": "+".join(sorted(set(item["sources"])))
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def filter_candidates(candidates):
    """Apply score threshold, chapter diversity, and final top-k filtering

    This function is shared by vector, BM25, and hybrid retrieval
    """
    if candidates:
        max_score = candidates[0]["score"]
        threshold = max_score*0.5

    filtered_docs = []
    filtered_metas = []
    chapter_count = {}

    for item in candidates:
        doc = item["doc"]
        meta = item["meta"]
        score = item["score"]

        #Drop weak candidates below the dynamic threshold
        if score < threshold:
            continue

        #Limit how many chunks can come from the same chapter
        ch = meta["chapter_num"]
        chapter_count[ch] = chapter_count.get(ch, 0) + 1

        if chapter_count[ch] > MAX_CHUNKS_PER_CHAPTER:
            continue

        #Copy metadata so we can attach retrieval source safely
        meta = dict(meta)
        meta["retrieval_source"] = item["source"]

        filtered_docs.append(doc)
        filtered_metas.append((meta, score))

        if len(filtered_docs) >= FINAL_TOP_K:
            break

    #Fallback, if filtering removes all the chapters, return the top candidates
    if not filtered_docs:
        for item in candidates[:FINAL_TOP_K]:
            meta = dict(item["meta"])
            meta["retrieval_source"] = item["source"]

            filtered_docs.append(item["doc"])
            filtered_metas.append((meta, item["score"]))

    return filtered_docs, filtered_metas


def retrieve(query, strategy=RETRIEVAL_STRATEGY):
    """Retrieve relevant chunks using vector, BM25, or both(hybrid search)

    Arguments:
        query {str} -- query to retrieve relevant chunks from
        strategy {str} -- strategy to use for retrieval
            - "vector": hierarchical semantic vector search only
            - "BM25": lexical BM25 search only
            - "hybrid": hybrid search merging vector and BM25 using RRF
    Returns:
        filtered_docs: List of retrieved chunk texts.
        filtered_metas: List of metadata-score pairs.
        retrieval_query: Normalized query used for retrieval.
    """
    # Normalize the query before retrieval.
    retrieval_query = normalize_query(query)

    # Compute embedding for vector retrieval.
    query_vector = model.encode(retrieval_query).tolist()

    #Extract meaningful keywords for lightweight reranking
    keywords = extract_keywords(retrieval_query)

    #Get vector candidates from the hierarchical path
    vector_candidates = retrieve_vector_candidates(
        retrieval_query,
        query_vector,
        keywords
    )

    #Get BM25 candidates from the entire chunk corpus
    bm25_candidates = retrieve_bm25_candidates(retrieval_query)

    if strategy == "vector":
        candidates = vector_candidates
    elif strategy == "BM25":
        candidates = bm25_candidates
    elif strategy == "hybrid":
        candidates = reciprocal_rank_fusion([vector_candidates, bm25_candidates])
    else:
        raise ValueError("Strategy not recognized, Use 'vector', 'bm25', or 'hybrid'.")

    filtered_docs, filtered_metas = filter_candidates(candidates)

    return filtered_docs, filtered_metas, retrieval_query
