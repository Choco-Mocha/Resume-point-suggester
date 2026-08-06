import os

OUTPUT_DIR = "data/extracted"

files = os.listdir(OUTPUT_DIR)
print(f"Total extracted files: {len(files)}")

for f in files:
    path = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(path)
    if size < 500:  # suspiciously short — likely OCR failure
        print(f"⚠️  LOW CONTENT: {f} ({size} bytes)")