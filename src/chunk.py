import os
import json

INPUT_DIR = "data/cleaned"
OUTPUT_FILE = "data/processed/chunks.json"

os.makedirs("data/processed", exist_ok=True)

def is_valid_bullet(line):
    line = line.strip()
    if len(line) < 20:          # too short to be a real bullet
        return False
    if not any(c.islower() for c in line):  # likely a header/label
        return False
    if len(line.split()) < 4:   # too few words to be a real sentence
        return False
    return True

def chunk_resume(text, source_file):
    lines = text.split("\n")
    chunks = []
    for line in lines:
        line = line.strip()
        if is_valid_bullet(line):
            chunks.append({
                "source": source_file,
                "text": line
            })
    return chunks

if __name__ == "__main__":
    all_chunks = []
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]

    for f in files:
        path = os.path.join(INPUT_DIR, f)
        with open(path, "r", encoding="utf-8") as infile:
            text = infile.read()
        chunks = chunk_resume(text, f)
        all_chunks.extend(chunks)
        print(f"{f}: {len(chunks)} bullets found")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        json.dump(all_chunks, outfile, indent=2)

    print(f"\nTotal bullets collected: {len(all_chunks)}")
    print(f"Saved to {OUTPUT_FILE}")