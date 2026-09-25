# Demo deployment

The default configuration now uses:

```text
Cohere Embed API -> Pinecone -> Gemini 3.6 Flash
```

## Environment variables

Copy `.env.example` to `.env` and set:

```text
COHERE_API_KEY=...
GEMINI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=rag-nong-nghiep-cohere
PINECONE_NAMESPACE=rag_collection_cohere
CORS_ORIGINS=https://your-frontend-domain.example
```

The API keys must stay on the backend. Do not put them in the React frontend.

## Install and run

```powershell
pip install -r requirements.txt
uvicorn api.app.main:app --host 0.0.0.0 --port 8000
```

Run the frontend separately:

```powershell
cd frontend
npm install
npm run dev
```

Upload PDFs, then press `Index`. The first indexing run creates the Pinecone
index if it does not exist and embeds all chunks with Cohere `embed-v4.0` at
1024 dimensions. The current demo configuration uses `force_reindex: true`,
so pressing `Index` rebuilds the configured namespace from the current PDF
set. The embedding cache is stored at `.cache/cohere_embeddings.json`.

The Cohere Embed v4 API accepts up to 96 text inputs per request. The configured
batch size is 96 to reduce the number of API calls; the account's trial quota
still applies.

## Important deployment note

The current hybrid retriever keeps BM25 documents in
`storage/pinecone/documents.jsonl`. On a VPS this directory must be on a
persistent disk. On an ephemeral platform, move this artifact to object
storage or replace the local BM25 stage with a managed sparse/full-text index
before running multiple API replicas.
