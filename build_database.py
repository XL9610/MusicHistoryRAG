import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import shutil
from config import (PDF_PATH, CHROMA_DB_PATH, COLLECTION_NAME,SUMMARY_COLLECTION_NAME,
                    EMBEDDING_MODEL, EMBEDDING_DEVICE)

# ============================================================
# Chapter titles (in order, corresponding to chapters 1-39)
# ============================================================
CHAPTER_TITLES = [
    "Chapter 1: Music in Antiquity",
    "Chapter 2: The Christian Church in the First Millennium",
    "Chapter 3: Roman Liturgy and Chant",
    "Chapter 4: Song and Dance Music to 1300",
    "Chapter 5: Polyphony through the Thirteenth Century",
    "Chapter 6: New Developments in the Fourteenth Century",
    "Chapter 7: Music and the Renaissance",
    "Chapter 8: England and Burgundy in the Fifteenth Century",
    "Chapter 9: Franco-Flemish Composers, 1450-1520",
    "Chapter 10: Madrigal and Secular Song in the Sixteenth Century",
    "Chapter 11: Sacred Music in the Era of the Reformation",
    "Chapter 12: The Rise of Instrumental Music",
    "Chapter 13: New Styles in the Seventeenth Century",
    "Chapter 14: The Invention of Opera",
    "Chapter 15: Music for Chamber and Church in the Early Seventeenth Century",
    "Chapter 16: France, England, Spain, the New World, and Russia in the Seventeenth Century",
    "Chapter 17: Italy and Germany in the Late Seventeenth Century",
    "Chapter 18: The Early Eighteenth Century in Italy and France",
    "Chapter 19: German Composers of the Late Baroque",
    "Chapter 20: Musical Taste and Style in the Enlightenment",
    "Chapter 21: Opera and Vocal Music in the Early Classic Period",
    "Chapter 22: Instrumental Music: Sonata, Symphony, and Concerto",
    "Chapter 23: Classic Music in the Late Eighteenth Century",
    "Chapter 24: Revolution and Change",
    "Chapter 25: The Romantic Generation: Song and Piano Music",
    "Chapter 26: Romanticism in Classical Forms: Choral, Chamber, and Orchestral Music",
    "Chapter 27: Romantic Opera and Musical Theater to Midcentury",
    "Chapter 28: Opera and Musical Theater in the Later Nineteenth Century",
    "Chapter 29: Late Romanticism in German Musical Culture",
    "Chapter 30: Diverging Traditions in the Later Nineteenth Century",
    "Chapter 31: The Early Twentieth Century: Vernacular Music",
    "Chapter 32: The Early Twentieth Century: The Classical Tradition",
    "Chapter 33: Radical Modernists",
    "Chapter 34: Between the World Wars: Jazz and Popular Music",
    "Chapter 35: Between the World Wars: The Classical Tradition",
    "Chapter 36: Postwar Crosscurrents",
    "Chapter 37: Postwar Heirs to the Classical Tradition",
    "Chapter 38: The Late Twentieth Century",
    "Chapter 39: The Twenty-First Century",
]

SOURCE_LABEL = "Burkholder - A History of Western Music, 10th ed."

def make_chapter_summary(chapter_text: str, max_chars: int = 1200) -> str:
    """
    This is a pseudo-summary implemented for building the hierarchical retrieval pipeline
    Taking just the first max_chars characters from each chapter text
    """
    clean = " ".join(chapter_text.split())
    return clean[:max_chars]


# ============================================================
# Step 1: Extract text from PDF
# ============================================================
print(f"Extracting text from {PDF_PATH}...")

doc = fitz.open(PDF_PATH)
raw_lines = []
for page in doc:
    if page.number < 38:  # Skip front matter
        continue
    text = page.get_text()
    raw_lines.extend(text.splitlines(keepends=True))

print(f"  Extracted {len(raw_lines)} lines from {len(doc)} pages")

# ============================================================
# Step 2: Clean text
# ============================================================
print("Cleaning text...")

cleaned_lines = []

for line in raw_lines:
    # Stop at glossary/appendix
    if line.strip() == "GLOSSARY":
        break

    # Remove short garbage lines (score artifacts, single characters)
    if len(line.strip()) < 3:
        continue

    cleaned_lines.append(line)

print(f"  Cleaned: {len(cleaned_lines)} lines remaining")

# ============================================================
# Step 3: Split into chapters
# ============================================================
print("Splitting into chapters...")

chapters = []
current_lines = []

for line in cleaned_lines:
    if line.strip() == "C H A P T E R":
        if current_lines:
            chapters.append(current_lines)
        current_lines = []
    else:
        current_lines.append(line)

# The last chapter
if current_lines:
    chapters.append(current_lines)

# Remove pre-chapter content (Part intro, etc.)
chapters = chapters[1:]
chapters = chapters[:39]  # Keep only the 39 actual chapters
print(f"  Found {len(chapters)} chapters")

# ============================================================
# Step 4: Chunk each chapter into smaller segments with metadata
# ============================================================
print("Chunking chapters...")

splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

all_chunks = []
all_metadata = []

chapter_summaries = []
all_summary_metadata = []


for i, chapter in enumerate(chapters):
    chapter_text = "".join(chapter)

    #Chapter level summary text
    summary_text = make_chapter_summary(chapter_text)
    chapter_summaries.append(summary_text)
    all_summary_metadata.append({
        "chapter_title": CHAPTER_TITLES[i],
        "chapter_num": i + 1,
        "source": SOURCE_LABEL
    })


    #Chunks level texts
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
print(f"  Total chapter summaries: {len(chapter_summaries)}")

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