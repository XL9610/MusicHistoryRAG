import anthropic
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_MAX_TOKENS

# Initialize LLM client once
llm = anthropic.Anthropic(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL
)

SYSTEM_PROMPT = """You are a music history study assistant based on Burkholder's "A History of Western Music" (10th edition).

Rules:
- Answer ONLY using the provided textbook excerpts
- If the excerpts are insufficient, say so explicitly
- Cite the chapter title in parentheses when referencing specific information
- Give a direct answer first, then a brief explanation"""


def build_context(filtered_docs, filtered_metas):
    """Combine retrieved chunks with source labels."""
    context_parts = []
    for doc, (meta, score) in zip(filtered_docs, filtered_metas):
        context_parts.append(f"[Source: {meta['chapter_title']}]\n{doc}")
    return "\n\n".join(context_parts)


def generate_answer(query, filtered_docs, filtered_metas):
    """Send context + question to LLM and return answer text."""
    context = build_context(filtered_docs, filtered_metas)

    response = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": f"""{SYSTEM_PROMPT}

TEXTBOOK EXCERPTS:
{context}

QUESTION: {query}"""
            }
        ]
    )

    answer_parts = []
    for block in response.content:
        if block.type == "text":
            answer_parts.append(block.text)
    return "\n".join(answer_parts)