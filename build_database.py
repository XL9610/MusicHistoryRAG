import fitz
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
import shutil
import anthropic
from config import (PDF_PATH, CHROMA_DB_PATH, COLLECTION_NAME, SUMMARY_COLLECTION_NAME,
                    EMBEDDING_MODEL, EMBEDDING_DEVICE, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
                    SUMMARY_API_KEY, SUMMARY_MODEL)
llm = anthropic.Anthropic(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

#OPENAI LLM for summary
summary_llm = OpenAI(api_key=SUMMARY_API_KEY)


SOURCE_LABEL = "Burkholder - A History of Western Music, 10th ed."

SUMMARY_CASHE = "chapter_summaries.json"

def make_chapter_summary(chapter_text: str, chapter_title: str) -> str:
    """
    Use LLM to generate a retrieval-optimized chapter summary.
    """
    # Truncate to avoid exceeding context window
    truncated = chapter_text[:8000]

    response =summary_llm.chat.completions.create(
        model=SUMMARY_MODEL,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""Summarize this music history chapter for a retrieval system.
        Include ALL composer names, work titles, musical forms, genres, and key terms mentioned.
        Be dense with searchable terms. No filler sentences.
        
    Chapter: {chapter_title}
    
    Text: {truncated}"""
        }]
    )


    return response.choices[0].message.content


def load_or_generate_summaries(chapter_titles, chapters):
    """Load from cache or generate from scratch summaries for chapters."""
    #If the caches document exists
    if os.path.exists(SUMMARY_CASHE):
        print(f"Loading cached summaries from {SUMMARY_CASHE}")
        with open(SUMMARY_CASHE, "r", encoding="utf-8") as f:
            return json.load(f)

    #If the cache does not exist, use the LLM to generate summaries chapter by chapter
    summaries = []
    total = len(chapters)

    for i, chapter in enumerate(chapters):
        chapter_text = "".join(chapter)
        print(f"\r  Creating summary [{i+1}/{total}] {chapter_titles[i]}...", end="", flush=True)

        try:
            summary = make_chapter_summary(chapter_text, chapter_titles[i])
        except Exception as e:
            #If the API summary fails, get the first 1200 words of that chapter instead
            print(f"\n Chapter{i+1} generation failed: {e}")
            clean = " ".join(chapter_text.split())
            summary = clean[:1200]

        summaries.append(summary)

    print(f"\n Generation complete. A total of {total} summaries are generated.")

    #Save the summaries
    with open(SUMMARY_CASHE, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return summaries



# ============================================================
# Step 1: Extract the contents of the PDF
# ============================================================
print(f"Reading from {PDF_PATH}...")
doc = fitz.open(PDF_PATH)
#Level1 -> Parts("PART ONE")
#Level2 -> Chapters("1. Music In Antiquity")
#Level3 -> subchapters("The Earliest Music")
toc = doc.get_toc()

# ============================================================
# Step 2: Get chapter and texts from the table of contents
# ============================================================
print("Getting chapter info from table of contents...")

#Get the 39 chapters
chapter_entries = []

for level, title, page in toc:
    if level == 2 and title[0].isdigit():
        chapter_entries.append((title, page))

#Generate chapter titles from the toc titles
# "1. Music In Antiquity" → "Chapter 1: Music In Antiquity"
CHAPTER_TITLES = []
for title, page in chapter_entries:
    dot_pos = title.index(".")
    num = title[:dot_pos]
    name = title[dot_pos + 1:].strip()
    CHAPTER_TITLES.append(f"Chapter {num}: {name}")

# ============================================================
# Step 3: Extract Text According to the TOC Page Numbers
# ============================================================
print("Extracting chapter texts from table of contents...")

#Default a glossary page for fail-safe
glossary_page = len(doc)

for level, title, page in toc:
    if "GLOSSARY" in title.upper():
        glossary_page = page
        break

chapters = []
for idx, (title,start_page) in enumerate(chapter_entries):
    #Set the ending page of the chapter
    if idx + 1 < len(chapter_entries):
        end_page = chapter_entries[idx + 1][1]
    else:
        end_page = glossary_page

    #Get texts page by page
    # -1 for switching from 1-indexed to 0-indexed
    chapter_lines = []
    for page_num in range(start_page-1, end_page-1):
        page = doc[page_num]
        text = page.get_text()
        for line in text.splitlines(keepends=True):
            if len(line.strip()) >= 3:
                chapter_lines.append(line)

    chapters.append(chapter_lines)

print(f"Extracted {len(chapters)} chapters.")



# ============================================================
# Step 4: Generate summaries and chunking
# ============================================================
print("Generating and chunking...")

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

all_chunks = []
all_metadata = []
all_summary_metadata = []

#Load or generate summaries for each chapter before chunking
chapter_summaries = load_or_generate_summaries(CHAPTER_TITLES, chapters)

#Chunking each chapter
for i, chapter in enumerate(chapters):
    chapter_text = "".join(chapter)

    #Metadata of the summaries
    all_summary_metadata.append({
        "chapter_title": CHAPTER_TITLES[i],
        "chapter_num": i + 1,
        "source": SOURCE_LABEL
    })
    #Chunking
    chunks = splitter.split_text(chapter_text)

    for j, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_metadata.append({
            "chapter_title": CHAPTER_TITLES[i],
            "chapter_num": i + 1,
            "chunk_index": j,
            "source": SOURCE_LABEL
        })

print(f"  Total chunks: {len(all_chunks)}")
print(f"  Total summaries: {len(chapter_summaries)}")

# ============================================================
# Step 5: Generate embeddings
# ============================================================
print("Generating embeddings (this may take ~30 seconds)...")

model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)

vectors = model.encode(all_chunks, show_progress_bar=True)
summary_vectors = model.encode(chapter_summaries, show_progress_bar=True)

print(f"  Embedding shape: {vectors.shape}")
print(f"  Summary embedding shape: {summary_vectors.shape}")


# ============================================================
# Step 6: Store in ChromaDB
# ============================================================
print("Storing in ChromaDB...")

shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

summary_collection = client.get_or_create_collection(
    name=SUMMARY_COLLECTION_NAME,
    metadata={"hnsw:space" : "cosine"}
)

chunk_collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

for i in range(len(all_chunks)):
    chunk_collection.add(
        ids=[f"chunk_{i}"],
        documents=[all_chunks[i]],
        embeddings=[vectors[i].tolist()],
        metadatas=[all_metadata[i]]
    )

for i in range(len(chapter_summaries)):
    summary_collection.add(
        ids = [f"chunk_{i}"],
        documents = [chapter_summaries[i]],
        embeddings = [summary_vectors[i].tolist()],
        metadatas=[all_summary_metadata[i]]
    )

print(f"  Stored {chunk_collection.count()} chunks")
print(f"  Stored {summary_collection.count()} summaries")
print("\nDatabase built successfully!")