# re-ranker-rag-agent
# PDF RAG System with FastAPI, Jina AI, and Groq

A high-performance Retrieval-Augmented Generation (RAG) backend that allows users to upload PDF documents, index them into a Supabase vector store, and perform hybrid searches with AI-powered reranking.

## 🚀 Features
- **PDF Processing:** Extracts text and chunks documents using `RecursiveCharacterTextSplitter`.
- **Hybrid Search:** Combines vector embeddings and keyword matching via Supabase RPC.
- **Advanced Embeddings:** Uses Jina AI's `jina-embeddings-v3`.
- **Reranking:** Refines search results using `jina-reranker-v3` for higher accuracy.
- **LLM Integration:** Powered by Llama 3.3 (via Groq) for fast and accurate responses.

## 🛠️ Setup

### Prerequisites
1. **Supabase:** You must have a Supabase project with the `pgvector` extension enabled and a table named `documents` containing an `embedding` column (vector size 1024).
2. **Hybrid Search Function:** You need a PostgreSQL function named `hybrid_search` defined in your Supabase database.

### Environment Variables
Create a `.env` file in the root directory:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
