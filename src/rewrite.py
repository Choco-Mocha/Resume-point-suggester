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


def retrieve_similar_bullets(user_text, n_results=6):
    results = collection.query(query_texts=[user_text], n_results=n_results)
    return results["documents"][0]


def rewrite_bullet(user_text):
    examples = retrieve_similar_bullets(user_text)
    examples_block = "\n".join(f"- {ex}" for ex in examples)

    prompt = f"""You are a resume writing coach.

The user wrote this raw bullet point:
"{user_text}"

Here are examples of strong, well-written resume bullets from similar contexts (for STYLE and STRUCTURE reference only — do not copy their content, facts, numbers, or achievements):
{examples_block}

Generate 3 different rewritten versions of the user's bullet point. Each version should:
- Start with a strong action verb
- Be concise (roughly 1 line, ~20-25 words)
- If a metric or impact is implied but missing, flag it clearly as [ADD METRIC] rather than inventing one
- Stay strictly truthful to what the user actually wrote — do not add achievements, numbers, or scope they didn't state

Vary the 3 versions in approach — for example, one could emphasize leadership, one could emphasize the outcome/impact, and one could be the most concise version possible.

Format your response exactly like this:

Option 1: [rewritten bullet]
Option 2: [rewritten bullet]
Option 3: [rewritten bullet]

Why: [one sentence explaining the overall changes made across all versions]"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    print("Paste your raw resume bullet (or 'quit' to exit):\n")
    while True:
        user_input = input("> ")
        if user_input.strip().lower() == "quit":
            break
        if not user_input.strip():
            continue

        print("\nRetrieving similar examples and generating suggestion...\n")
        result = rewrite_bullet(user_input)
        print(result)
        print("\n" + "-" * 60 + "\n")