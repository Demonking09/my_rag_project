# PDF-Based RAG Assistant

A polished local Retrieval-Augmented Generation (RAG) application that answers questions from a university PDF syllabus using LangChain, Chroma, and Google Gemini.

## Overview

This project demonstrates a practical end-to-end RAG pipeline:

1. Load a PDF document
2. Split it into meaningful chunks
3. Store embeddings in a local Chroma vector database
4. Retrieve the most relevant context for a user question
5. Generate a concise answer using Google Gemini
6. Fall back gracefully if the live model is unavailable

## Architecture

```text
User Question
    │
    ▼
PDF Loader (PyPDF)
    │
    ▼
Text Splitter
    │
    ▼
Embeddings (Google Gemini)
    │
    ▼
Chroma Vector Store
    │
    ▼
Retriever + Prompt Template
    │
    ▼
Google Gemini Chat Model
    │
    └── Fallback excerpt if model is unavailable
```

## Project Structure

```text
.
├── rag_app.py              # Main RAG application
├── tests/
│   └── test_rag_app.py    # Fallback and retrieval behavior tests
├── chroma_db/             # Local Chroma persistence directory
├── SET0101_B-Tech_CSE_2023_27.pdf
├── .env.example           # Example environment configuration
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## Features

- PDF ingestion from a local syllabus file
- Chunk-based document splitting
- Persistent vector search with Chroma
- Local fallback answer generation if the API is unavailable
- Clean Python implementation suitable for experimentation and learning

## Requirements

- Python 3.10+
- A Google API key with access to Gemini models

## Setup

1. Create a virtual environment
   ```bash
   python -m venv .venv
   ```

2. Activate it
   ```bash
   .venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a .env file
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. Run the app
   ```bash
   python rag_app.py
   ```

## Usage

When prompted, enter a question such as:

- What topics are in the soft computing syllabus?
- What are the course outcomes?
- Explain the program structure briefly.

## Notes

The app is intentionally simple and educational. It can be extended with:

- better chunking strategies
- hybrid retrieval
- source citation support
- a web interface
- evaluation metrics for RAG quality

## License

This project is intended for educational and demonstration purposes.
