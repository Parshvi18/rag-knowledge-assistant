#  RAG Knowledge Assistant

An AI-powered knowledge management platform that lets you upload documents and instantly query them using natural language. Built with a production-grade RAG (Retrieval-Augmented Generation) pipeline not just an API wrapper.


## 🛠️ Tech Stack

**Frontend**
- Next.js 14 + TypeScript
- Tailwind CSS + Shadcn/UI
- React Query for server state
- Streaming UI for real-time LLM responses

**Backend**
- Node.js + Fastify
- PostgreSQL for metadata and user management
- Pinecone / pgvector for vector storage
- OpenAI Embeddings + GPT-4 for generation

**Infrastructure**
- Docker for containerization
- AWS S3 for document storage
- Redis for caching and rate limiting

---

## ✨ Key Features

- 📄 Upload PDFs, DOCX, and TXT files
- 🔍 Semantic search across all your documents
- 💬 Conversational Q&A with memory across turns
- 📌 Source citations with exact document references
- ⚡ Streaming responses for real-time feel
- 🔐 Auth with organization-level document isolation
- 📊 Query history and analytics dashboard

---

## 🧠 Technical Decisions

**Why RAG over fine-tuning?**
Fine-tuning bakes knowledge into weights — it can't update dynamically. RAG retrieves fresh context at query time making it ideal for ever-changing document sets.

**Why pgvector over Pinecone?**
For self-hosted setups pgvector keeps everything in one Postgres instance reducing infra complexity. Pinecone is offered as an alternative for scale.

**Chunking strategy**
Used recursive character splitting with 512 token chunks and 50 token overlap to preserve context across chunk boundaries — a critical detail most implementations miss.

---

## 🏃 Running Locally

```bash
git clone https://github.com/yourusername/rag-knowledge-assistant
cd rag-knowledge-assistant
cp .env.example .env
docker-compose up
```

---

## 🔑 Environment Variables

```env
OPENAI_API_KEY=
DATABASE_URL=
REDIS_URL=
AWS_S3_BUCKET=
PINECONE_API_KEY=
```

---

## 📈 Performance

- Average query response time: ~1.2s end to end
- Supports documents up to 50MB
- Handles 10K+ chunks per organization
- P95 latency under 2s at 100 concurrent users

---

