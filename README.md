# Resume-point-suggester

RAG-based tool that rewrites resume bullets using similar bullets from a pool of real resumes as context.

## Features
- Rewrite a single bullet or a block of bullets from one role
- Topic search across resumes (retrieval only, no API call)
- Retrieved-chunk preview and copy-paste prompt for Gemini web
- Gemini API with automatic fallback to local Ollama
- On-disk response cache

## Pipeline
PDF → PyMuPDF + Tesseract OCR → clean → chunk into bullets → boilerplate dedupe → `BAAI/bge-small-en-v1.5` embeddings → ChromaDB → top-k retrieval → Gemini / Ollama (`llama3.1:8b`)

## Numbers
- Resumes indexed: <N>
- Bullets indexed: <N>
- Time to find a comparable bullet: <X min> manual vs <Y sec> with tool

## Demo
<screenshots or video>

## Setup
Requires Python, [Tesseract](https://github.com/tesseract-ocr/tesseract), [Ollama](https://ollama.com) (`ollama pull llama3.1:8b`).

```
git clone https://github.com/Choco-Mocha/Resume-point-suggester.git
cd Resume-point-suggester
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`.env`:
```
GOOGLE_API_KEY=your_key
```

Tesseract path is set in `src/ingest.py`.

## Usage
Put PDFs in `data/resumes/`, then:
```
python src\ingest.py
python src\clean.py
python src\chunk.py
python src\dedupe_boilerplate.py
python src\embed_store.py
streamlit run src\app.py
```
To re-index after adding resumes, delete `resume_db` and rerun `embed_store.py`.

## Files
- `ingest.py`: PDF to text
- `clean.py`, `chunk.py`, `dedupe_boilerplate.py`: preprocessing
- `embed_store.py`: embeddings, ChromaDB
- `rewrite.py`: retrieval, prompts, fallback, cache
- `app.py`: Streamlit UI

## Limitations
- Fallback triggers only on Gemini failure; Ollama must be running locally
- Resume data not included (private)
