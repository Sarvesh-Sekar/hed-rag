# 🎓 Higher Education Department (HED) – Hybrid RAG Pipeline

## 📌 Overview

The **HED RAG Pipeline** is an AI-powered system designed to enable intelligent retrieval, analysis, and decision support over large-scale government data sources such as policies, schemes, regulations, and institutional datasets.

The system combines:

* **Semantic Retrieval (Vector RAG)** for understanding unstructured documents
* **Knowledge Graph (Graph RAG)** for relationship-aware reasoning
* **Structured Querying (SQL)** for precise analytical data

This hybrid approach ensures accurate, explainable, and context-aware responses for administrators, policymakers, and stakeholders.

---

## 🚀 Problem Statement

Higher Education Departments manage vast and fragmented data across:

* Policies & regulations (PDFs, circulars)
* Schemes & programs (web portals)
* Institutional & statistical data (CSV, APIs)

Current systems:

* Depend on manual search
* Lack cross-source insights
* Are slow and error-prone

👉 This system addresses these issues through **AI-driven retrieval and reasoning**.

---

## 🧠 Key Features

### 🔍 Hybrid Retrieval System

* **Vector Search** → semantic understanding of documents
* **Graph Retrieval** → relationship traversal (multi-hop queries)
* **SQL Queries** → structured analytics

---

### 🧩 Knowledge Graph Integration

* Models entities like:

  * Schemes
  * Institutions
  * States
  * Policies
  * Metrics (GER, funding, rankings)
* Enables:

  * Cross-entity reasoning
  * Policy impact analysis
  * Multi-hop queries

---

### 📂 Multi-Source Ingestion

Supports ingestion from:

* File uploads (PDF, DOCX, CSV, XLSX)
* Cloud storage (S3, Drive)
* Web scraping (government portals)

---

### ⚙️ Asynchronous Processing

* Distributed ingestion using background workers
* Scalable ETL pipelines for different data types
* Job tracking and fault tolerance

---

### 🧠 Session-Aware Querying

* Redis-based session memory (hot storage)
* PostgreSQL for persistent chat history
* Context-aware follow-up queries

---

## 🏗️ System Architecture

```text
                ┌────────────────────┐
                │   Data Sources     │
                │ PDFs | TXT | DOCX  │
                └────────┬───────────┘
                         ↓
                ┌────────────────────┐
                │ Ingestion Layer    │
                │ ETL Pipelines      │
                └────────┬───────────┘
                         ↓
            ┌────────────────────────────────┐
            │ Document Processing Layer      │
            │ - Text Extraction              │
            │ - Chunking                     │
            │ - Embedding Generation         │
            │ - Chunk Scoring                │
            └────────┬───────────────────────┘
                     ↓
         ┌────────────────────────────────┐
         │ Knowledge Graph Construction   │
         │ - Entity Extraction (LLM)      │
         │ - Relationship Extraction      │
         │ - Entity Resolution            │
         │ - Ontology Mapping             │
         │ - Graph Merging                │
         └────────┬───────────────────────┘
                  ↓
        ┌────────────────────────────────┐
        │ Storage Layer                  │
        │ - Neo4j Graph DB               │
        │ - Chunk Embeddings             │
        │ - PostgreSQL                   │
        │ - Redis Cache                  │
        └────────┬───────────────────────┘
                 ↓
        ┌────────────────────────────────┐
        │ Hybrid Retrieval Engine        │
        │ Vector + Graph + SQL           │
        └────────┬───────────────────────┘
                 ↓
        ┌────────────────────────────────┐
        │ LLM Response Generation        │
        └────────┬───────────────────────┘
                 ↓
        ┌────────────────────────────────┐
        │ User Interface / APIs          │
        └────────────────────────────────┘
```

---

## 🔄 Data Processing Pipelines

### 📄 Unstructured Data (PDFs, Web)

* Text extraction
* Chunking
* Entity & relationship extraction (LLM)
* Graph construction (Neo4j)
* Embedding generation → Vector DB

---

### 📊 Structured Data (CSV, XLSX)

* Schema normalization
* Direct relationship mapping
* SQL storage (analytics)
* Graph enrichment (entity relationships)

---

## 🔗 Graph + Embedding Integration

Each graph node stores:

* Entity metadata
* Embedding vector

This enables:

1. **Semantic Node Retrieval** (via embeddings)
2. **Graph Expansion** (via relationships)

---

## 🔍 Query Flow

```text
User Query
   ↓
Session Context (Redis)
   ↓
Query Router
   ↓
 ┌───────────────┬───────────────┬───────────────┐
 │ Vector Search │ Graph Query   │ SQL Query     │
 └───────────────┴───────────────┴───────────────┘
          ↓ Combined Context
               ↓
              LLM
               ↓
            Response
```

---

## 🧠 Hybrid RAG Strategy

* **Vector RAG** → finds relevant documents
* **Graph RAG** → discovers relationships
* **SQL** → provides exact numerical insights

👉 Combined for **accurate + explainable answers**

---

## ⚡ Tech Stack

* **Backend:** FastAPI
* **LLM Framework:** LangChain / LangGraph
* **Graph DB:** Neo4j


---

## 🔐 Scalability & Reliability

* Asynchronous ingestion pipeline
* Horizontal worker scaling
* Redis-based caching
* Periodic sync to persistent storage

---

## 💡 Use Cases

* Policy analysis and comparison
* Scheme eligibility and impact queries
* Institutional performance insights
* Decision support for administrators

---

## 🚀 Future Enhancements

* Agentic workflow for complex reasoning
* Real-time data ingestion
* Advanced analytics dashboards
* Multi-lingual support

---

## 🧠 Key Insight

> “Vector RAG finds relevant information,
> Graph RAG connects it,
> SQL validates it — together they enable intelligent decision-making.”

---
