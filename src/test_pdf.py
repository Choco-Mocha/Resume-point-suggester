import fitz  # PyMuPDF

pdf = r"data\resumes\Tanmay_lodha_Petronet_LNG_Limited_1pg.pdf"

doc = fitz.open(pdf)

print("Pages:", len(doc))

for i, page in enumerate(doc):
    text = page.get_text()
    print(f"Page {i+1}: {len(text)} characters")
    print("-" * 40)
    print(text[:500])
    print()