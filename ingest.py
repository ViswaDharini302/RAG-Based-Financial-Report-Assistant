# Read PDFs, split into chunks, convert to vectors,and save into FAISS vector database.

import os
import pickle
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATABASE_DIR = "database"
INDEX_FILE = os.path.join(DATABASE_DIR, "financial.index")
CHUNKS_FILE = os.path.join(DATABASE_DIR, "chunks.pkl")
META_FILE = os.path.join(DATABASE_DIR, "metadata.pkl")

# Reads a PDF file and returns all text as one big string
def read_pdf(pdf_path: str) -> str:
    print(f"Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    all_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            all_text += text + "\n"
    print(f"Read {len(reader.pages)} pages, got {len(all_text)} characters")
    return all_text

# Splits a large text into many small overlapping pieces (chunks).
def chunk_text(text: str, source_name: str) -> tuple[list, list]:
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_text(text)
    # For every chunk, record which file it came from
    metadata = [{"source": source_name} for _ in chunks]
    print(f"Created {len(chunks)} chunks")
    return chunks, metadata

# Converts text to numbers (vectors)
def load_embedding_model():
    print("Loading embedding model (downloads once, ~90 MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model ready!")
    return model

def embed_chunks(chunks: list, model) -> np.ndarray:
    print(f"Converting {len(chunks)} chunks to vectors...")
    vectors = model.encode(
        chunks,
        show_progress_bar=True,
        batch_size=32)  # Process 32 chunks at a time (faster) 
    print(f"Vectors shape: {vectors.shape}")
    return vectors

# Creates a FAISS index and adds all vectors to it.
def build_faiss_index(vectors: np.ndarray):
    print("Building FAISS vector index...")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors.astype("float32"))
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

# Saves the FAISS index, chunks, and metadata to disk
def save_to_disk(index, chunks: list, metadata: list):
    os.makedirs(DATABASE_DIR, exist_ok=True)
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Merging with existing database...")
        existing_index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            existing_chunks = pickle.load(f)
        with open(META_FILE, "rb") as f:
            existing_meta = pickle.load(f)
        existing_index.add(
            index.reconstruct_n(0, index.ntotal).astype("float32"))
        chunks = existing_chunks + chunks
        metadata = existing_meta + metadata
        index = existing_index
    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)
    # Save chunks (the raw text pieces)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    # Save metadata (which chunk came from which file)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    print(f"💾 Saved {index.ntotal} total vectors to database/")

def ingest_pdf(pdf_path: str):
    filename = os.path.basename(pdf_path)
    print(f"\n{'=' * 55}")
    print(f"Ingesting: {filename}")
    print(f"{'=' * 55}")
    text = read_pdf(pdf_path)
    if not text.strip():
        print("⚠️ No text found in PDF (might be scanned image). Skipping.")
        return
    chunks, metadata = chunk_text(text, source_name=filename)
    model = load_embedding_model()
    vectors = embed_chunks(chunks, model)
    index = build_faiss_index(vectors)
    save_to_disk(index, chunks, metadata)
    print(f"\nDone! '{filename}' is ready to be queried.\n")

if __name__ == "__main__":
    REPORTS_DIR = "reports"
    pdf_files = [
        os.path.join(REPORTS_DIR, f)
        for f in os.listdir(REPORTS_DIR)
        if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in 'reports/' folder.")
        print("Please copy your PDF reports into the 'reports/' folder first!")
    else:
        print(
            f"Found {len(pdf_files)} PDF file(s): "
            f"{[os.path.basename(f) for f in pdf_files]}")
        for pdf in pdf_files:
            ingest_pdf(pdf)
        print("\nAll PDFs ingested! Now run: streamlit run app.py")

