import os
from dotenv import load_dotenv

load_dotenv()

# Paths
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "music_history"
SUMMARY_COLLECTION_NAME = "music_history_chapter_summaries"
PDF_PATH = "A History of Western Music Tenth.pdf"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# LLM
LLM_API_KEY = os.getenv("MINIMAX_API_KEY")
LLM_BASE_URL = "https://api.minimax.io/anthropic"
LLM_MODEL = "MiniMax-M2.7"
LLM_MAX_TOKENS = 2000

# Retrieval
SCORE_THRESHOLD = 0.1
MAX_CHUNKS_PER_CHAPTER = 3
N_RESULTS = 10
TOP_CHAPTERS = 3

STOP_WORDS = {"what", "when", "where", "which", "who", "whom", "whose", "how",
              "does", "have", "will", "would", "could", "should", "about",
              "this", "that", "these", "those", "from", "with", "they", "them",
              "their", "there", "here", "were", "been", "being", "some", "than"}