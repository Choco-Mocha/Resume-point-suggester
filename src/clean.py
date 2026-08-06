import os
import re

INPUT_DIR = "data/extracted"
CLEANED_DIR = "data/cleaned"
os.makedirs(CLEANED_DIR, exist_ok=True)

def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)          # collapse multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)        # collapse excessive blank lines
    text = re.sub(r'Page \d+ of \d+', '', text)   # remove page numbers (common pattern)
    text = text.strip()
    return text

if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    for f in files:
        with open(os.path.join(INPUT_DIR, f), "r", encoding="utf-8") as infile:
            raw = infile.read()
        cleaned = clean_text(raw)
        with open(os.path.join(CLEANED_DIR, f), "w", encoding="utf-8") as outfile:
            outfile.write(cleaned)
    print(f"Cleaned {len(files)} files → {CLEANED_DIR}")