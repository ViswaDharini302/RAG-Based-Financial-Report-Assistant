RAG-Based Financial Report Assistant

RAG-Based Financial Report Assistant is an AI-powered application designed to simplify the analysis of financial reports by enabling users to interact with PDF documents using natural language. Instead of manually searching through lengthy annual reports, users can ask questions in plain English and receive accurate, context-aware answers generated through a Retrieval-Augmented Generation (RAG) pipeline.

## Features
- Natural language question answering over financial PDF reports.
- Retrieval-Augmented Generation (RAG) pipeline for context-aware responses.
- Semantic document retrieval using FAISS vector search.
- Dense vector embeddings generated with Sentence Transformers.
- AI-powered answer generation using Google Gemini.
- Automatic PDF text extraction, preprocessing, and chunking.
- Fast and scalable retrieval from large financial documents.
- Interactive web interface built with Streamlit.
- Modular architecture that supports multiple financial reports.
- Grounded responses based on the retrieved document context.


## Working

The application follows a Retrieval-Augmented Generation (RAG) workflow to provide reliable answers from financial reports.

1. One or more financial PDF reports are uploaded to the application.
2. The documents are parsed, and the extracted text is divided into smaller semantic chunks.
3. Each text chunk is converted into a dense vector embedding using a Sentence Transformers embedding model.
4. The generated embeddings are stored in a FAISS vector database to enable efficient similarity-based retrieval.
5. When a user submits a question, the query is transformed into an embedding using the same embedding model.
6. FAISS retrieves the document chunks that are most semantically relevant to the user's query.
7. The retrieved context, along with the user's question, is provided to Google Gemini.
8. Google Gemini generates a context-aware response based exclusively on the retrieved information, delivering accurate and relevant insights from the financial reports.


