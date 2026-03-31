# AMAO — Adaptive Multi-Agent Orchestration

A production-ready, **config-driven** multi-agent AI backend that routes natural language queries to specialized agents — SQL, NoSQL, and RAG — using a LangGraph orchestration pipeline, served via FastAPI with a Streamlit frontend.

The key principle: **every client gets exactly the agents they need, nothing more.** Agent availability, database connections, LLM models, and vector store backends are all defined per-client in a single `config.yaml` file.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Core Pipeline](#core-pipeline)
- [Vector Store & RAG](#vector-store--rag)
- [Database Layer](#database-layer)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Roles & Access Control](#roles--access-control)
- [Client Configuration](#client-configuration)
- [Environment Setup](#environment-setup)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)

---

## Overview

AMAO accepts natural language queries and intelligently routes them across three agent types:

- **SQL Agent** — Generates and executes `SELECT` queries against relational databases (MySQL, PostgreSQL, SQLite, MariaDB, MSSQL)
- **NoSQL Agent** — Queries document databases (MongoDB)
- **RAG Agent** — Retrieves answers from uploaded PDFs and text files using vector similarity search

Each client organisation has its own isolated configuration: which agents are active, which databases they connect to, which LLM model they use, and which vector store backend holds their documents. Two clients can run entirely different agent combinations on the same deployment.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│         (Login · Chat · Super Admin Dashboard)       │
└───────────────────────┬──────────────────────────────┘
                        │ HTTP (JWT Bearer)
                        ▼
┌──────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│                  POST /chat                          │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│               GraphManager (per-client cache)        │
│   Reads config.yaml → builds Orchestrator once       │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│            Orchestrator  (LangGraph graph)           │
│                                                      │
│  router_node                                         │
│    └─► LLM decides which agents to run + sub-queries │
│                                                      │
│  agent nodes  (only enabled agents exist as nodes)   │
│    ├─► sql_agent   → async parallel SQL execution    │
│    ├─► nosql_agent → MongoDB query execution         │
│    └─► rag_agent   → vector similarity retrieval     │
│                                                      │
│  final_node                                          │
│    └─► LLM synthesises all results → final_response  │
└──────────────────────────────────────────────────────┘
```

---

## How It Works

1. **User sends a query** via the Streamlit chat or directly to `POST /chat`.
2. **JWT is validated** — user's `client_id` is extracted from the token.
3. **GraphManager** checks its cache. If no orchestrator exists for this client, it reads `config.yaml`, builds the agent graph, and caches it.
4. **Router node** — an LLM receives the user query and the list of available agents (from config) and returns a JSON execution plan: which agents to run and with what sub-query.
5. **Agent nodes execute sequentially** in the order specified by the plan. Each agent appends its results to the shared state (`sql_agent_results`, `rag_agent_results`, etc.).
6. **Final node** — another LLM call synthesises all agent results into a single coherent answer.
7. **Response** is returned as `{ "final_response": "..." }`.

If files are uploaded alongside the query, they are chunked and indexed into the client's vector store before the query is processed.

---

## Project Structure

```
.
├── app.py                          # Streamlit frontend
├── main.py                         # FastAPI app: routers, middleware, DB init, seeding
├── pyproject.toml                  # Dependencies (uv)
│
├── src/
│   ├── agents/
│   │   ├── base.py                 # Abstract BaseAgent (name, config, run())
│   │   ├── sql_agent.py            # SQL agent with schema cache + parallel execution
│   │   ├── nosql_agent.py          # NoSQL / MongoDB agent
│   │   └── rag_agent.py            # RAG agent with multi-intent decomposition
│   │
│   ├── api/routes/
│   │   ├── chat/__init__.py        # POST /chat — main query endpoint
│   │   ├── auth/__init__.py        # /login /register /refresh /logout /get-current-user
│   │   ├── clients/__init__.py     # Client CRUD + DB connection test
│   │   ├── configs/__init__.py     # Per-client config.yaml CRUD
│   │   ├── agents/__init__.py      # Agent management
│   │   ├── models/__init__.py      # LLM model management
│   │   ├── user/__init__.py        # User management
│   │   ├── feedback/__init__.py    # User feedback
│   │   └── logs/__init__.py        # Log access
│   │
│   ├── core/
│   │   ├── orchestrator.py         # LangGraph graph: router → agents → final
│   │   ├── graph_manager.py        # Per-client orchestrator cache (async-safe)
│   │   ├── agent_factory.py        # Instantiates agents from config
│   │   ├── llm_factory.py          # Creates & caches LLM clients (OpenAI / Groq)
│   │   ├── registry.py             # Maps agent names → agent classes
│   │   └── state_manager.py        # AgentState TypedDict definition
│   │
│   ├── configs/                    # Per-client YAML files (one folder per client UUID)
│   │   └── client_id_<uuid>/
│   │       └── config.yaml
│   │
│   ├── Database/
│   │   ├── base_db.py              # Async SQLAlchemy engine + session wrapper
│   │   ├── connection_factory.py   # Registry-pattern factory for SQL + MongoDB
│   │   ├── connection_manager.py   # Per-client connection cache + RBAC enforcement
│   │   ├── models/                 # SQLModel ORM models (system DB tables)
│   │   └── schema_extractor/
│   │       ├── sql_extractor.py    # Introspects SQL DB schema for LLM context
│   │       └── nosql_extractor.py  # Introspects MongoDB collections
│   │
│   ├── prompts/
│   │   ├── router_prompt.py        # Builds JSON execution plan from query + agents
│   │   ├── sql_prompt.py           # Generates SELECT queries from schema + question
│   │   ├── nosql_prompt.py         # Generates MongoDB queries
│   │   ├── rag_prompt.py           # Decomposes query into semantic sub-intents
│   │   └── final_prompt.py         # Synthesises multi-agent results into one answer
│   │
│   ├── repositories/               # Data access layer (one per domain)
│   ├── schema/                     # Pydantic / SQLModel schemas
│   ├── security/
│   │   ├── o_auth.py               # JWT creation, validation, role enforcement
│   │   └── dependencies.py         # FastAPI OAuth2 scheme dependency
│   ├── services/                   # Business logic layer
│   ├── settings/config.py          # Env-var config (DB URL, JWT, LLM, embeddings)
│   ├── tools/
│   │   ├── sql_search.py           # Executes SQL queries
│   │   ├── nosql_search.py         # Executes NoSQL queries
│   │   ├── rag_search.py           # Calls vector store retrieve()
│   │   └── nosql_executors/
│   │       └── mongo_executor.py   # MongoDB-specific execution
│   ├── utils/
│   │   ├── document_processor.py   # PDF/TXT chunker (PyMuPDF, heading detection)
│   │   ├── db_seeder.py            # Seeds initial roles, users, models on startup
│   │   ├── hash_util.py            # JWT / password hashing
│   │   └── logger.py               # Structured logging
│   ├── vector_db/
│   │   ├── base.py                 # BaseVectorStore (embedding loader, path helpers)
│   │   ├── faiss_store.py          # FAISS: incremental diff + dedup ingestion
│   │   └── chroma_store.py         # ChromaDB: incremental diff + dedup ingestion
│   └── vector_stores/              # Persisted vector data (one folder per client UUID)
│       └── client_id_<uuid>/
│           ├── faiss/              # index.faiss, index.pkl, hashes/
│           └── chroma/             # chroma.sqlite3, hashes/
│
├── logs/app.log
└── test/                           # Sample PDFs and test scripts
```

---

## Agents

### SQL Agent
- Extracts the full schema of all configured SQL connections (table names, column names, types) and injects it into the LLM prompt.
- Schema is cached in-memory per connection with a 10-minute TTL to avoid repeated introspection.
- LLM returns a JSON array of `{ connection_alias, query }` pairs.
- All queries are executed in **parallel** via `asyncio.gather`.
- Only `SELECT` queries are permitted — write operations are silently dropped.

### NoSQL Agent
- Introspects MongoDB collection structure via `nosql_extractor.py`.
- LLM generates the appropriate query for the target collection.
- Executed via `mongo_executor.py` using Motor (async MongoDB driver).

### RAG Agent
- On each query, the LLM first **decomposes the question** into atomic semantic sub-intents (e.g. "What is X?" + "How does Y work?").
- Each sub-query is run against the client's vector store independently.
- Results are aggregated and passed to the final node.
- Supports FAISS and ChromaDB backends (configured per client).

### Base Agent
All agents extend `BaseAgent`, which enforces a single interface:
```python
async def run(self, state: dict) -> dict
```
The `state` dict carries `user_query`, `client_id`, `user_id`, `connection_manager`, and accumulated results from prior agents.

---

## Core Pipeline

### Router Prompt Logic
The router receives the **exact list of enabled agents** for the client (not all possible agents). Its rules:
- Always include every available agent in the execution plan.
- Preserve the full original query per agent — never split or modify it.
- Append `user_id` / `client_id` context to SQL and NoSQL queries.
- Return a strictly valid JSON array — no markdown, no explanation.

Example plan for a client with all three agents enabled:
```json
[
  { "agent": "sql_agent",   "query": "How many users signed up last month? (user_id=..., client_id=...)" },
  { "agent": "nosql_agent", "query": "How many users signed up last month? (user_id=..., client_id=...)" },
  { "agent": "rag_agent",   "query": "How many users signed up last month?" }
]
```

### LangGraph Execution
The graph is built dynamically based on the enabled agent set. Conditional edges route through each agent in plan order, then to the `final` node. The graph is compiled once and cached per client in `GraphManager`.

### Final Node
Collects `sql_agent_results`, `nosql_agent_results`, and `rag_agent_results` from state, formats them as structured context, and calls the LLM to produce a single coherent answer. Returns `"No relevant data found."` if all results are empty.

---

## Vector Store & RAG

### Document Ingestion
Upload PDFs or `.txt` files via the chat endpoint. The `DocumentProcessor` (PyMuPDF):
- Sorts blocks by vertical position per page.
- Detects and prefixes headings to their associated paragraphs.
- Detects reference sections (`References`, `Bibliography`, etc.) and stops ingestion there.
- Filters copyright notices and standalone page numbers.
- SHA-256 hashes each chunk for deduplication.

### Incremental Indexing
Both FAISS and ChromaDB stores use the same diff strategy:
- **Same filename re-uploaded** → diff old vs new hashes, delete removed chunks, add new chunks.
- **New filename** → check against all existing IDs in the index, skip duplicates, add only truly new chunks.

Re-uploading an unchanged document is a no-op.

### Storage Layout
```
src/vector_stores/
└── client_id_<uuid>/
    ├── faiss/
    │   ├── index.faiss
    │   ├── index.pkl
    │   └── hashes/
    │       └── <document_name>.hashes   # JSON: { sha256_hash: paragraph_text }
    └── chroma/
        ├── chroma.sqlite3
        └── hashes/
            └── <document_name>.hashes
```

---

## Database Layer

### System Database (MySQL)
Used internally by the platform to store users, clients, roles, agent configs, model configs, feedback, and logs. Managed by SQLModel + async SQLAlchemy. Initialised and seeded automatically on application startup via FastAPI's `lifespan` hook.

### Client Databases
Configured per-client in `config.yaml`. Connected on demand via `ConnectionManager`, which caches connections per `client_id` for the lifetime of the process.

Supported SQL databases:

| Database   | Async Driver         |
|------------|----------------------|
| MySQL      | `mysql+aiomysql`     |
| PostgreSQL | `postgresql+asyncpg` |
| MariaDB    | `mariadb+aiomysql`   |
| SQLite     | `sqlite+aiosqlite`   |
| MSSQL      | `mssql+aioodbc`      |

Supported NoSQL: **MongoDB** via Motor (async).

SQL connection pool settings: `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`, `pool_recycle=3600`.

---

## API Reference

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/register-user` | Register a new user |
| `POST` | `/login` | JSON body login → returns `access_token` + `refresh_token` |
| `POST` | `/refresh` | Refresh access token |
| `DELETE` | `/logout` | Invalidate current token |
| `GET` | `/get-current-user` | Get authenticated user's profile |

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Submit a query, optionally upload files for RAG indexing |

Request: `multipart/form-data` with `query: str` and optional `files: List[UploadFile]`.
Response: `{ "final_response": "...", "sql_agent_results": [...], ... }`

### Clients *(SuperAdmin only)*
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/clients/add-client` | Register a new client organisation |
| `PUT` | `/clients/update-client/{client_id}` | Update client details |
| `DELETE` | `/clients/remove-client/{client_id}` | Delete a client |
| `GET` | `/clients/list-clients` | Paginated client list |
| `GET` | `/clients/get-client/{client_id}` | Get a single client |
| `GET` | `/clients/connect/{client_id}` | Test client DB connections |

### Configs *(SuperAdmin only)*
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/configs/create-config-file/{client_id}` | Upload a config.yaml for a client |
| `PUT` | `/configs/update-config-file/{client_id}` | Update existing config |
| `DELETE` | `/configs/remove-config-file/{client_id}` | Remove config |
| `GET` | `/configs/read-config/{client_id}` | Read current config as JSON |

Additional route groups: `/agents`, `/models`, `/user`, `/feedback`, `/logs`.

---

## Frontend

The Streamlit app (`app.py`) connects to the FastAPI backend at `http://localhost:8000`.

### Login
JWT-based login. Token is stored in `st.session_state` for the duration of the session.

### Assistant *(all roles)*
- Full chat interface with message history.
- **Upload & Index Knowledge** expander — drag-and-drop PDFs or `.txt` files and click **Build Index** to ingest them into the RAG vector store before querying.

### Super Admin Dashboard *(SuperAdmin only)*
- **Clients tab** — Register new client organisations, view the active client list.
- **Configs tab** — Select a client, upload a `config.yaml`, or view the current config rendered as syntax-highlighted YAML.

---

## Roles & Access Control

| Role | Permissions |
|------|-------------|
| `SuperAdmin` | Full access to all routes and all client data |
| `Admin` | Chat + access to own client's data only |
| `User` | Chat only, scoped to own client |

Enforcement happens at two levels:
- **Route level** via `auth_dependency.require_roles([...])` FastAPI dependency.
- **Connection level** — `ConnectionManager` blocks `Admin`/`User` from accessing any other client's databases at query time, regardless of what they pass.

---

## Client Configuration

Each client has a `config.yaml` under `src/configs/client_id_<uuid>/`. Include only the agents the client requires — the orchestrator graph is built exclusively from agents with `enabled: true`.

```yaml
client_name: Acme Corp

allowed_agents:

  sql_agent:
    agent_id: <uuid>
    version: v0
    enabled: true
    model_name: llama-3.3-70b-versatile   # gpt-4o also supported
    temperature: 0
    description: Handles sales and user activity queries.
    database:
      connection1:
        db_type: mysql          # mysql | postgres | sqlite | mssql | mariadb
        host: localhost
        port: 3306
        username: dbuser
        password: dbpass
        db_name: sales_db
      connection2:
        db_type: postgres
        host: pg-host
        port: 5432
        username: pguser
        password: pgpass
        db_name: analytics_db

  nosql_agent:
    agent_id: <uuid>
    version: v0
    enabled: true
    model_name: llama-3.3-70b-versatile
    temperature: 0
    description: Handles product catalogue queries.
    database:
      connection1:
        db_type: mongodb
        host: localhost
        port: 27017
        username: admin
        password: admin123
        db_name: catalogue

  rag_agent:
    agent_id: <uuid>
    version: v0
    enabled: true
    model_name: llama-3.3-70b-versatile
    top_k: 3
    description: Answers questions from uploaded company documents.
    vector_db: faiss            # faiss | chroma
```

A client with only `sql_agent` and `rag_agent` (no `nosql_agent`) will have a two-node graph — the NoSQL node simply does not exist for that client.

---

## Environment Setup

Copy `.env.example` to `.env` and fill in your values:

```env
# System Database (MySQL)
MY_SQL_USER=
MY_SQL_PASSWORD=
MY_SQL_HOST=
MY_SQL_PORT=
MY_SQL_DB=

# JWT
HASH_SECRET_KEY=
HASH_ALGORITHM=HS256
TOKEN_EXPIRY_TIME=3600

# LLM API key (Groq or OpenAI)
LLM_API_KEY=

# HuggingFace embedding model name
EMBEDDING_MODEL=

# Vector store root directory
VECTOR_DB_PATH=src/vector_stores
```

**LLM provider routing** is automatic based on `model_name` in the client config:
- Names starting with `gpt` → OpenAI (`ChatOpenAI`)
- Names containing `llama` → Groq (`ChatGroq`)

The API key is read from the agent's `api_key` field first, then falls back to `LLM_API_KEY`.

---

## Getting Started

### Prerequisites
- Python 3.10+
- A running MySQL instance (system database)
- `uv` package manager (recommended) or `pip`

### Install

```bash
git clone https://github.com/your-org/amao.git
cd amao

# With uv
uv sync

# Or with pip
pip install -e .
```

### Run the Backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On first startup the application will create all system DB tables and seed initial roles and data automatically.

### Run the Frontend

```bash
streamlit run app.py
```

### Onboard a Client

1. Log in as `SuperAdmin`.
2. Go to **Management → Clients** and register the client.
3. Go to **Management → Configs**, select the client, and upload their `config.yaml`.
4. Switch to **Assistant** mode and start querying.

### Index Documents for RAG

In the Assistant view, expand **Upload & Index Knowledge**, upload PDFs or `.txt` files, then click **Build Index**. Documents are chunked, deduplicated, embedded, and stored in the client's vector store automatically.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend framework | FastAPI + Uvicorn |
| Agent orchestration | LangGraph |
| LLM providers | OpenAI (`gpt-*`), Groq (`llama-*`) |
| Embeddings | HuggingFace Sentence Transformers |
| Vector stores | FAISS, ChromaDB |
| SQL databases | MySQL, PostgreSQL, SQLite, MariaDB, MSSQL (all async) |
| NoSQL databases | MongoDB (Motor — async) |
| System ORM | SQLModel + async SQLAlchemy |
| PDF processing | PyMuPDF |
| Auth | JWT (python-jose) + passlib (argon2) |
| Frontend | Streamlit |
| Package manager | uv |
| Logging | Structured file + console logger |
