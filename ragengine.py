# PURPOSE: The "brain" of the project.
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
# Load environment variables from .env file (reads your API key)
load_dotenv()
#  File paths
DATABASE_DIR = "database"
INDEX_FILE   = os.path.join(DATABASE_DIR, "financial.index")
CHUNKS_FILE  = os.path.join(DATABASE_DIR, "chunks.pkl")
META_FILE    = os.path.join(DATABASE_DIR, "metadata.pkl")
class RAGEngine:
    def __init__(self):
        self.index    = None
        self.chunks   = None
        self.metadata = None
        self.embed_model = None
        self.gemini_model = None
        self.is_ready = False
        self._load_resources()

    def _load_resources(self): 
        if not os.path.exists(INDEX_FILE):
            print("  No database found. Please run: python ingest.py")
            return
        print(" Loading FAISS index...")
        self.index = faiss.read_index(INDEX_FILE)
        print(" Loading text chunks...")
        with open(CHUNKS_FILE, 'rb') as f:
            self.chunks = pickle.load(f)
        with open(META_FILE, 'rb') as f:
            self.metadata = pickle.load(f)
        print(" Loading embedding model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # ── Connect to Gemini API ──
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            print("  No Gemini API key found. Please update your .env file.")
            return
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        self.is_ready = True
        print(f" RAG Engine ready! {self.index.ntotal} chunks indexed.\n")

    def retrieve(self, question: str, top_k: int = 5, source_filter: str = None):
        # Convert the question into a vector
        question_vector = self.embed_model.encode([question])          
        question_vector = question_vector.astype('float32')
        # Search FAISS for the closest matching chunk vectors
        # D = distances (how similar), I = indices (which chunks)
        distances, indices = self.index.search(question_vector, top_k * 2)
        results = []
        for idx in indices[0]:                        
            if idx == -1:                             
                continue
            chunk_text = self.chunks[idx]
            source     = self.metadata[idx]["source"]
            # If user selected a specific report, only use chunks from that report
            if source_filter and source_filter != "All Reports":
                if source_filter not in source:
                    continue
            results.append((chunk_text, source))
            if len(results) >= top_k:                 # Stop once we have enough
                break
        return results

#Send retrieved chunks + question to Gemini and get an answer
    def generate_answer(self, question: str, context_chunks: list) -> str:
        if not context_chunks:
            return "❌ I couldn't find any relevant information in the uploaded reports for your question."
        # Build a formatted context string from all the chunks
        context_text = ""
        for i, (chunk, source) in enumerate(context_chunks, 1):
            context_text += f"\n--- Excerpt {i} (from {source}) ---\n{chunk}\n"
        # This is the "prompt" — the instructions we send to Gemini
        prompt = f"""You are a financial analyst assistant helping users understand annual reports.
CONTEXT FROM FINANCIAL REPORTS:
{context_text}
QUESTION: {question}
INSTRUCTIONS:
- Answer ONLY using the context provided above
- Give detailed insights in layman's terms where the context don't change
- If the answer is not in the context, say "I couldn't find this information in the uploaded reports"
- Be specific with numbers, dates, and percentages when available
- Format your answer clearly with bullet points or paragraphs
- At the end, mention which report(s) you found this information in
ANSWER:"""
        # Send the prompt to Gemini and get the response
        response = self.gemini_model.generate_content(prompt)
        return response.text

    def ask(self, question: str, source_filter: str = None) -> dict:
        if not self.is_ready:
            return {
                "answer": "⚠️ The system is not ready. Please check that you have:\n1. Uploaded PDF files\n2. Run python ingest.py\n3. Added your Gemini API key to .env",
                "sources": [],
                "chunks": []
            }
        # Step 1: Retrieve relevant chunks
        retrieved = self.retrieve(question, top_k=5, source_filter=source_filter)
        # Step 2: Generate answer using Gemini
        answer = self.generate_answer(question, retrieved)
        # Collect unique source filenames
        sources = list(set(source for _, source in retrieved))
        return {
            "answer":  answer,
            "sources": sources,
            "chunks":  retrieved       # Raw chunks — used to show "Sources" tab in UI
        }
#Returns a list of all PDF filenames that have been indexed.
    def get_available_sources(self) -> list:
        if not self.metadata:
            return []
        sources = list(set(m["source"] for m in self.metadata))
        return sorted(sources)
