import json
from collections import Counter

INPUT_FILE = "data/processed/chunks.json"
OUTPUT_FILE = "data/processed/chunks_clean.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Count how many times each exact text appears across all resumes
text_counts = Counter(c["text"] for c in chunks)

# Count how many DISTINCT source files each text appears in
# (this matters more than raw count, in case one resume repeats a line)
source_map = {}
for c in chunks:
    source_map.setdefault(c["text"], set()).add(c["source"])

# If a line appears in more than 5 different resumes, it's boilerplate
BOILERPLATE_THRESHOLD = 5

clean_chunks = []
removed = []

for c in chunks:
    if len(source_map[c["text"]]) > BOILERPLATE_THRESHOLD:
        removed.append(c["text"])
    else:
        clean_chunks.append(c)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clean_chunks, f, indent=2)

print(f"Original chunks: {len(chunks)}")
print(f"Removed as boilerplate: {len(chunks) - len(clean_chunks)}")
print(f"Remaining clean chunks: {len(clean_chunks)}")
print("\nSample of removed boilerplate:")
for text in sorted(set(removed))[:15]:
    print(f"  - {text}")