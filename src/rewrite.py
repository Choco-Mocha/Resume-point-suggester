import os
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# --- Setup ---
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="resume_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
collection = chroma_client.get_collection(name="resume_bullets", embedding_function=ef)

# NOTE: change this if your embed_store.py used a different metadata key
# for the source resume filename (e.g. "filename", "resume_id", etc.)
SOURCE_KEY = "source"


def retrieve_similar_bullets(user_text, n_results=8):
    results = collection.query(
        query_texts=[user_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(documents, metadatas, distances))


def get_source_cluster(source_name, exclude_text=None, max_bullets=6):
    """
    Fetch other bullets belonging to the same source resume, so the LLM
    can see a real example of what bullets tend to cluster together
    (style/structure reference only).
    """
    result = collection.get(
        where={SOURCE_KEY: source_name},
        include=["documents"],
    )
    docs = result.get("documents", [])
    if exclude_text:
        docs = [d for d in docs if d.strip() != exclude_text.strip()]
    return docs[:max_bullets]


def rewrite_bullet(user_text):
    matches = retrieve_similar_bullets(user_text, n_results=8)

    examples_block = "\n".join(f"- {doc}" for doc, _meta, _dist in matches)

    # Top 3 sources by similarity (lowest distance = most similar)
    top_sources = matches[:3]

    # Pull real bullet clusters from the top matching sources — shows
    # what OTHER bullets appeared alongside a similar point in real
    # resumes, purely as structural inspiration.
    seen_sources = set()
    cluster_blocks = []
    for doc, meta, _dist in top_sources:
        source_name = meta.get(SOURCE_KEY, "unknown") if meta else "unknown"
        if source_name in seen_sources or source_name == "unknown":
            continue
        seen_sources.add(source_name)
        cluster = get_source_cluster(source_name, exclude_text=doc)
        if cluster:
            cluster_text = "\n".join(f"  - {c}" for c in cluster)
            cluster_blocks.append(f"From a resume with a similar point:\n{cluster_text}")

    clusters_block = "\n\n".join(cluster_blocks) if cluster_blocks else "(no additional cluster examples found)"

    prompt = f"""You are a resume writing coach.

The user wrote this raw bullet point:
"{user_text}"

Here are examples of strong, well-written resume bullets from similar contexts (for STYLE and STRUCTURE reference only — do not copy their content, facts, numbers, or achievements):
{examples_block}

Here are examples of OTHER bullets that appeared alongside similar points in real resumes (for STRUCTURE/PATTERN reference only — shows what kinds of complementary points people typically pair with a bullet like this):
{clusters_block}

TASK 1 — Rewrite the user's bullet:
Generate 3 different rewritten versions of the user's bullet point. Each version should:
- Start with a strong action verb
- Be concise (roughly 1 line, ~20-25 words)
- If a metric or impact is implied but missing, flag it clearly as [ADD METRIC] rather than inventing one
- Stay strictly truthful to what the user actually wrote — do not add achievements, numbers, or scope they didn't state
Vary the 3 versions in approach — e.g. one emphasizes leadership, one emphasizes outcome/impact, one is the most concise version possible.

TASK 2 — Suggest companion bullets:
Resume bullets for a single project/role/initiative usually come in a cluster of 3-4 covering different angles (e.g. context/scope, specific actions or skills used, collaboration/leadership, quantified impact/result). Based on the user's bullet and the pattern shown in the cluster examples above, suggest 2-3 ADDITIONAL bullet points the user could write to round out this same achievement.
- These must be FILL-IN-THE-BLANK templates, not invented facts — use bracketed placeholders like [describe X] or [ADD NUMBER] for anything you don't actually know
- Each suggestion should target a DIFFERENT angle than the user's original bullet and than each other
- Keep each suggestion to one line

Format your response EXACTLY like this:

Option 1: [rewritten bullet]
Option 2: [rewritten bullet]
Option 3: [rewritten bullet]

Why: [one sentence explaining the overall changes made across all versions]

Companion bullets you could add:
1. [template bullet with placeholders]
2. [template bullet with placeholders]
3. [template bullet with placeholders]"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    result_text = response.text

    # Append source attribution for the top 3 most similar bullets used
    sources_block = "\n\nTop matching resumes referenced for style:\n"
    for i, (doc, meta, dist) in enumerate(top_sources, 1):
        source_name = meta.get(SOURCE_KEY, "unknown") if meta else "unknown"
        similarity = 1 - dist  # rough similarity score for display only
        sources_block += f"  {i}. {source_name}  (similarity: {similarity:.2f})\n"

    return result_text + sources_block


def topic_stats(query_text, distance_threshold=0.5, n_results=None):
    """
    Estimate what % of resumes in the DB have a bullet semantically
    similar to the given topic/query.

    This is an approximation based on embedding distance, not an exact
    keyword count — phrasing, OCR quality, etc. all affect results.
    """
    total_count = collection.count()
    if n_results is None:
        n_results = total_count  # search the whole collection

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, total_count),
        include=["metadatas", "distances"],
    )
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matching_sources = set()
    for meta, dist in zip(metadatas, distances):
        if dist <= distance_threshold:
            source_name = meta.get(SOURCE_KEY, "unknown") if meta else "unknown"
            matching_sources.add(source_name)

    # Total unique resumes in the whole DB (not just matched ones)
    all_metadatas = collection.get(include=["metadatas"])["metadatas"]
    all_sources = set(
        meta.get(SOURCE_KEY, "unknown") for meta in all_metadatas if meta
    )
    total_resumes = len(all_sources) if all_sources else 1

    pct = (len(matching_sources) / total_resumes) * 100

    print(f"\nTopic: \"{query_text}\"")
    print(f"Matched resumes: {len(matching_sources)} / {total_resumes} ({pct:.1f}%)")
    print("(Based on semantic similarity, distance threshold = "
          f"{distance_threshold}. Lower threshold = stricter match.)")
    if matching_sources:
        print("\nSample matching resumes:")
        for name in list(matching_sources)[:10]:
            print(f"  - {name}")
    print()


if __name__ == "__main__":
    print("Commands:")
    print("  Paste a bullet point to get rewrite suggestions")
    print("  Type 'stats: <topic>' to see what % of resumes mention that topic")
    print("  Type 'quit' to exit\n")

    while True:
        user_input = input("> ")
        stripped = user_input.strip()

        if stripped.lower() == "quit":
            break
        if not stripped:
            continue

        if stripped.lower().startswith("stats:"):
            topic = stripped[len("stats:"):].strip()
            if not topic:
                print("Usage: stats: <topic>  e.g. stats: participated in an olympiad")
                continue
            topic_stats(topic)
            continue

        print("\nRetrieving similar examples and generating suggestion...\n")
        result = rewrite_bullet(user_input)
        print(result)
        print("\n" + "-" * 60 + "\n")