from retriever import retrieve
from generator import generate_answer

# Interactive query loop
while True:
    query = input("\nEnter your question (type 'quit' to exit): ")

    if query == "quit":
        break

    # 1. Retrieve relevant chunks
    filtered_docs, filtered_metas, matched_term = retrieve(query)

    if not filtered_docs:
        print("⚠️ No sufficiently relevant content found.")
        continue

    # 2. Display sources
    print("\n📚 Sources:")
    for meta, score in filtered_metas:
        print(f"  - [{score}] {meta['chapter_title']}, chunk {meta['chunk_index']}")

    # 3. Generate and display answer
    answer = generate_answer(query, filtered_docs, filtered_metas)
    print(answer)