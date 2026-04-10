# MusicRAG

An AI-powered study companion for Western music history, built on Burkholder's *A History of Western Music* (10th edition). Uses Retrieval-Augmented Generation (RAG) to answer questions grounded in textbook content.

## What It Does

Ask any question about Western music history — composers, styles, forms, historical developments — and get an accurate, source-cited answer based on the textbook.

```
Enter your question: What is sonata form?

📚 Sources:
  - [0.316] Chapter 23: Classic Music in the Late Eighteenth Century, chunk 14
  - [0.265] Chapter 23: Classic Music in the Late Eighteenth Century, chunk 18

Sonata form is a large-scale musical structure that grew out of binary form,
presenting an exposition, development, and recapitulation of thematic material.
It was first established in the 18th century... (Chapter 23)
```

## How It Works

```
User Question
      │
      ▼
┌──────────────────┐
│ Query Normalization │  Strip filler phrases ("expand on", "explain")
└──────┬───────────┘
       ▼
┌──────────────┐
│  Embedding   │  all-MiniLM-L6-v2 (384-dim, local CPU)
└──────┬───────┘
       ▼
┌──────────────┐
│  ChromaDB    │  2,240 chunks with chapter metadata
└──────┬───────┘
       ▼
┌──────────────────┐
│ Keyword Reranking │  Boost chunks containing query terms
│ Dynamic Threshold │  Adapt cutoff per query
│ Chapter Diversity │  Max 2 chunks per chapter
└──────┬───────────┘
       ▼
┌──────────────┐
│   LLM API    │  Generate answer with source citations
└──────┬───────┘
       ▼
   Cited Answer
```

## Setup

### Prerequisites

- Python 3.10+
- A PDF of Burkholder's *A History of Western Music* (10th edition)
- An LLM API key (default: [MiniMax](https://www.minimaxi.com/), but any Anthropic-compatible API works)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/music-rag.git
cd music-rag
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
MINIMAX_API_KEY=your_api_key_here
```

To use a different LLM provider, edit the API settings in `config.py`.

### Build the Database

Place your PDF in the project root (named `A History of Western Music Tenth.pdf`), then run:

```bash
python build_database.py
```

This takes about 1–2 minutes and will:
1. Extract text from the PDF (skipping front matter and appendices)
2. Clean noise (score artifacts, map labels, figure captions)
3. Split into 39 chapters, then chunk each chapter (~1,500 chars with 200-char overlap)
4. Generate embeddings using a local transformer model
5. Store 2,240 chunks with metadata in a local ChromaDB database

### Query

```bash
python query.py
```

## Project Structure

```
├── config.py           # All tunable parameters
├── retriever.py        # Search pipeline: normalize → retrieve → rerank → filter
├── generator.py        # LLM prompt and API call
├── query.py            # Interactive CLI (~20 lines)
├── build_database.py   # Full ingestion pipeline: PDF → ChromaDB
├── requirements.txt
└── .env                # API key (not committed)
```

## Retrieval Strategy

The retrieval pipeline went through three iterations:

**v1 — Keyword gate**: Used keyword filtering (`$contains`) as an entry point before vector search. Failed on natural language queries like "expand on galant style" because irrelevant keywords ("expand") narrowed the search space.

**v2 — Improved keyword gate**: Added stop words, case-variant matching, term sorting by length. Better, but the fundamental architecture was still fragile.

**v3 — Vector-first with keyword reranking (current)**: Always starts with broad vector search (top 20), then applies keyword-based reranking, dynamic thresholds, and chapter diversity filtering. Most stable across diverse query types.

Key features:
- **Query normalization** — strips filler phrases before embedding
- **Keyword reranking** — boosts chunks containing query terms without excluding those that don't
- **Dynamic threshold** — `max(0.08, top_score × 0.5)` adapts to each query's score distribution
- **Chapter diversity** — max 2 chunks per chapter to avoid redundant results
- **Fallback** — retries with relaxed threshold if primary filtering returns empty

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF extraction | PyMuPDF |
| Text chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| LLM | MiniMax-M2.7 (via Anthropic SDK) |

## Limitations

- Chunking is fixed-size, not section-aware — some chunks split mid-paragraph
- Single-turn only — no conversation memory
- Reranking is lexical keyword overlap, not a cross-encoder model
- No formal evaluation framework yet

## Planned Features

- **Quiz Mode** — auto-generate multiple-choice questions from textbook content
- **Composer Profile** — input a name, get a structured summary from all mentions
- **Concept Tracker** — trace a concept's evolution across chapters and eras
- **Web UI** — Streamlit/Gradio frontend with Markdown rendering

## License

MIT
