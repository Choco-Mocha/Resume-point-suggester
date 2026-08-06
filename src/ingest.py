import os
import fitz
from PIL import Image
import pytesseract


# Tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


RESUME_DIR = "data/resumes"
OUTPUT_DIR = "data/extracted"


os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_text(pdf_path):

    doc = fitz.open(pdf_path)

    full_text = ""

    for page in doc:

        # Try normal PDF extraction first
        text = page.get_text()

        if text.strip():

            full_text += text

        else:

            # OCR fallback
            pix = page.get_pixmap(
                dpi=300
            )

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            text = pytesseract.image_to_string(
                img,
                lang="eng"
            )

            full_text += text

    return full_text



if __name__ == "__main__":

    pdf_files = [
        f for f in os.listdir(RESUME_DIR)
        if f.lower().endswith(".pdf")
    ]


    print(f"Found {len(pdf_files)} resumes")


    for i, file in enumerate(pdf_files, 1):

        try:

            pdf_path = os.path.join(
                RESUME_DIR,
                file
            )


            txt_name = (
                os.path.splitext(file)[0]
                + ".txt"
            )


            out_path = os.path.join(
                OUTPUT_DIR,
                txt_name
            )


            # Skip already processed resumes
            if os.path.exists(out_path):

                print(
                    f"[{i}/{len(pdf_files)}] Skipping: {file}"
                )

                continue


            print(
                f"[{i}/{len(pdf_files)}] Processing: {file}"
            )


            text = extract_text(pdf_path)


            with open(
                out_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(text)


            print(
                f"   Saved {len(text)} characters"
            )


        except Exception as e:

            print(
                f"ERROR processing {file}: {e}"
            )