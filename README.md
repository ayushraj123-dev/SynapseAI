# SynapseAI

> **Interactive Knowledge Base & Topic-Augmented Knowledge Graph Builder**

SynapseAI is an intelligent knowledge retrieval and reasoning platform designed to transform unstructured documents into an explorable, topic-augmented knowledge graph. It enables grounded question answering by combining dense vector similarity with graph traversal and multi-hop reasoning.

---

## 📌 Project Overview

SynapseAI's end-to-end vision encompasses:
1. **Document Ingestion**: Parsing PDF and TXT documents safely into normalized text with page metadata.
2. **Boundary-Aware Chunking**: Splitting text into provenance-tracked semantic chunks with stable IDs.
3. **Entity & Theme Extraction**: Extracting named entities, relations (triplets), and concept themes.
4. **Knowledge Graph Construction**: Building a topological network representing interconnected ideas.
5. **Vector Store Integration**: Indexing embeddings for semantic search and factual retrieval.
6. **Adaptive Query Routing**: Directing user questions to thematic, relationship, or factual retrieval paths.
7. **Grounded Answer Synthesis**: Generating verifiable answers with citations referencing exact source documents and graph nodes.

> **Current Status:** **Complete Hackathon Architecture** is active and fully functional. It features persistent **Neo4j** graph database storage alongside local **ChromaDB** vector persistence, cross-document topic community retrieval, a split-screen Streamlit interface, and dynamic auto-focus / auto-zoom graph evidence highlighting. Zero required paid API keys or mandatory cloud dependencies — fully operational with local fallback.

---

## 🏗️ Project Structure

```text
SynapseAI/
├── app.py                                   # Split-screen Streamlit application with Document Processing, Grounded Q&A, and Interactive Graph
├── requirements.txt                         # Runtime & test dependencies (streamlit, networkx, pyvis, chromadb, neo4j, pytest)
├── README.md                                # Comprehensive setup, architecture, Neo4j configuration, and verification guide
├── .gitignore                               # Git ignore rules for virtual environments, chroma_db, and caches
├── .env.example                             # Template for Neo4j and optional LLM environment configurations
├── chroma_db/                               # Local persistent ChromaDB vector store (git-ignored)
├── data/
│   └── .gitkeep                             # Directory for user-provided documents (tracked in git)
├── src/                                     # Core application modules (clean separation of concerns)
│   ├── __init__.py                          # Package initialization and module exports
│   ├── document_ingestion.py                # PDF & TXT extraction, validation, and chunking (Task 2 ✅)
│   ├── entity_extraction.py                 # Entity, relationship, and theme extraction with multi-doc tracking (Task 3 ✅)
│   ├── graph_database.py                    # Persistent Neo4j driver, Cypher schema constraints, and repeat-safe upsert (Priority 1 ✅)
│   ├── knowledge_graph.py                   # Knowledge graph construction, PyVis canvas, and auto-zoom highlighting (Task 4 & Priority 5 ✅)
│   ├── vector_store.py                      # Persistent ChromaDB vector store & semantic search (Task 4 ✅)
│   ├── query_router.py                      # Hybrid retrieval routing between topic communities and graph paths (Task 5 & Priority 3 ✅)
│   └── answer_generator.py                  # Grounded answer synthesis with source citations & abstention (Task 5 & Priority 4 ✅)
└── tests/
    ├── test_project_structure.py            # Verification suite for repository files, imports, and stubs
    ├── test_document_ingestion.py           # Verification suite for PDF/TXT extraction, chunking, and boundaries
    ├── test_entity_extraction.py            # Verification suite for entities, relations, themes, and provenance
    ├── test_knowledge_graph_and_vector_store.py # Automated test suite for graph topology, ChromaDB, & vector retrieval
    ├── test_query_routing_and_grounded_qa.py # Test suite for query routing, hybrid retrieval, citations, & 4-case evaluation fixture
    └── test_neo4j_and_cross_doc_graph.py    # Test suite for Neo4j persistence, Cypher constraints, cross-doc themes & UI highlighting (Priority 7 ✅)
```

---

## 🚦 Feature Status Matrix

| Component / Feature | Current Status | Details |
| :--- | :--- | :--- |
| **Split-Screen Web UI** | ✅ **Active** | Two-panel experience: Left panel handles document ingestion, chat input, and answers; Right panel renders the interactive knowledge graph |
| **Document Upload & Validation** | ✅ **Active** | Enforces `.pdf` and `.txt` extensions, multi-file ingestion, and a **20 MB per-file size limit** |
| **PDF Text Extraction** | ✅ **Active** | In-memory page-by-page extraction via PyMuPDF; tracks 1-based page numbers |
| **TXT Decoding & Fallback** | ✅ **Active** | UTF-8 with automatic BOM stripping (`utf-8-sig`) and CP1252 / Latin-1 fallback |
| **Boundary-Aware Chunking** | ✅ **Active** | Target ~800 chars, ~120 chars overlap; respects word and punctuation boundaries |
| **Provenance Tracking** | ✅ **Active** | Every chunk tracks `chunk_id`, `source_filename`, `page_number`, `chunk_index`, and `char_count` |
| **Entity Extraction** | ✅ **Active** | Normalization, deduplication, categorization (`CONCEPT`, `TECHNOLOGY`, `ORGANIZATION`, `PERSON`, etc.) |
| **Relationship Extraction** | ✅ **Active** | Subject-Predicate-Object triples with strict endpoint validation and exact source quotes |
| **Cross-Document Themes** | ✅ **Active** | Topic communities detecting cross-document concept overlaps (`TopicTheme.is_cross_document`) |
| **Persistent Graph Database (Neo4j)** | ✅ **Active** | Official `neo4j` Python driver, schema uniqueness constraints, indexes, parameterized Cypher upserts, and clean purge |
| **Local Graph Fallback** | ✅ **Active** | Transparent, honest NetworkX fallback when Neo4j is unconfigured or offline |
| **Interactive Graph Canvas & Auto-Zoom** | ✅ **Active** | PyVis/VisJS canvas that automatically zooms to and highlights evidence nodes and relationship paths for user queries |
| **Vector Database** | ✅ **Active** | Persistent local `ChromaDB` index with repeat-safe upsert and document purge |
| **Semantic Retrieval** | ✅ **Active** | Dense similarity search returning ranked passages, relevance scores, and citations |
| **Adaptive Query Routing** | ✅ **Active** | Classifies user queries into `FACTUAL`, `RELATIONSHIP`, `THEMATIC`, or `UNANSWERABLE` |
| **Hybrid Retrieval Engine** | ✅ **Active** | Dynamically routes between ChromaDB dense passages, Neo4j/NetworkX paths, and topic communities |
| **Grounded Answer Synthesis** | ✅ **Active** | Generates direct answers backed by verifiable `[1]`, `[2]` citations with page and chunk provenance |
| **Evidence Sufficiency & Abstention** | ✅ **Active** | Conservative checks; safely abstains on out-of-scope queries without hallucinating |

---

## 🧭 Query Routing, Hybrid Retrieval & Grounded Q&A (Task 5)

### 1. Query Classification Architecture ([`src/query_router.py`](file:///c:/Users/ayush/OneDrive/Desktop/Hackathons/Hypnexis/SynapseAI/src/query_router.py))
SynapseAI dynamically analyzes user queries using deterministic heuristics and entity/theme awareness:
- **`FACTUAL`**: Questions seeking a specific fact, definition, or property (e.g., *"What vector database does SynapseAI use?"*). Routes to dense semantic passage retrieval over ChromaDB and checks graph node attributes.
- **`RELATIONSHIP`**: Questions asking how entities are connected, interact, or compare (e.g., *"How is SynapseAI connected to ChromaDB?"*). Inspects the NetworkX knowledge graph for direct edges and multi-hop shortest paths (up to 3 hops), verifying links against underlying passage evidence.
- **`THEMATIC`**: Questions seeking broad themes, summaries, or patterns across the corpus (e.g., *"What are the main themes across these documents?"*). Retrieves topic theme clusters, component summaries, and representative passages.
- **`UNANSWERABLE`**: Out-of-scope trivia, chit-chat, or queries without textual grounding (e.g., *"What is the average battery life of an iPhone 16 in minutes?"*). Bypasses speculative retrieval and triggers transparent abstention.

### 2. Hybrid Retrieval & Provenance Deduplication
- **Multi-Source Evidence**: Retrieval aggregates vector passages, graph relationship triples, node attributes, and theme summaries into typed `RetrievedEvidence` items.
- **Deduplication with Provenance Preservation**: When a vector passage and a structured graph relationship originate from the exact same source chunk, the deduplicator prioritizes the structured relationship (`graph_relationship > vector_passage > theme_summary > graph_node`) while retaining distinct passages across the document collection.
- **Document Integrity**: Graph connectivity alone is never treated as proof of a relationship unless the edge carries concrete supporting quotation provenance (`source_filename`, `page_number`, `source_chunk_id`).

### 3. Answer Generation & Citation Validation ([`src/answer_generator.py`](file:///c:/Users/ayush/OneDrive/Desktop/Hackathons/Hypnexis/SynapseAI/src/answer_generator.py))
- **Dual Synthesis Modes**:
  1. **Zero-API Offline Grounded Synthesizer**: Extracts and synthesizes statements directly from retrieved evidence and verified graph triples, attaching strict citation tags `[1]`, `[2]`. Runs 100% locally with zero external dependencies.
  2. **Configurable LLM Provider**: Supports optional API integration (Google Gemini / OpenAI). If unconfigured or encountering rate limits/timeouts, seamlessly falls back to the deterministic offline synthesizer without breaking the user experience.
- **Citation Validation**: Every returned citation is cross-checked against `retrieval.evidence_items`. If a citation cannot be resolved to an actual document chunk or verified graph edge, it is rejected.
- **Stable Citation Identifiers**: Citations resolve to:
  - Source filename (e.g., `synapse_tech_report.pdf`)
  - 1-based PDF page number (`Page 1`) or document-level marker for `.txt`
  - Exact source `chunk_id`
  - Supporting quotation snippet

### 4. Conservative Evidence Sufficiency & Abstention
Before returning an answer, SynapseAI evaluates evidence sufficiency:
- **Keyword Coverage & Relevance**: Verifies that substantive query terms actually appear in retrieved passages.
- **Relationship Presence**: For relationship questions, confirms that connecting graph paths or mutual co-occurrence evidence exists.
- **Three-Tier Status**:
  - `SUPPORTED`: High topical coverage (>=50%) or verified graph triples.
  - `PARTIALLY_SUPPORTED`: Moderate coverage (25%-50%); warns the user that context may be incomplete.
  - `INSUFFICIENT_EVIDENCE`: Minimal/zero keyword overlap or out-of-scope; refuses to answer and explains what information is missing without fabricating citations.

---

## 🧪 Controlled Evaluation Fixture

SynapseAI includes an automated evaluation fixture in [`tests/test_query_routing_and_grounded_qa.py`](file:///c:/Users/ayush/OneDrive/Desktop/Hackathons/Hypnexis/SynapseAI/tests/test_query_routing_and_grounded_qa.py) evaluating four controlled cases:

| Case | Test Query | Expected Intent | Retrieval Strategy | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Case A: Factual** | *"What graph library does SynapseAI implement?"* | `FACTUAL` | Dense Vector Search (Top-4) | Grounded answer citing `synapse_spec.pdf` (Page 1, `eval_chunk_1`), identifying NetworkX. |
| **Case B: Relationship** | *"How is SynapseAI connected to NetworkX?"* | `RELATIONSHIP` | Graph Traversal + Vector Verification | Traverses edge `SynapseAI ——[IMPLEMENTS]——> NetworkX`, cites exact quotation. |
| **Case C: Thematic** | *"What are the main themes across these documents?"* | `THEMATIC` | Thematic Cluster Retrieval | Identifies `Hybrid Retrieval Architecture` theme cluster without hallucination. |
| **Case D: Unanswerable** | *"What is the average battery life of an iPhone 16 in minutes?"* | `UNANSWERABLE` / `FACTUAL` | Conservative Sufficiency Check | **Abstained safely**: reports insufficient evidence, generates **0** false citations. |

---

## 🚀 Running Tests & Launching SynapseAI

### Step 1: Run All Automated Tests
```powershell
pytest -v
```

**Actual Test Execution Results:**
```text
tests/test_document_ingestion.py ............             [ 23%]
tests/test_entity_extraction.py ...........               [ 44%]
tests/test_knowledge_graph_and_vector_store.py ............ [ 67%]
tests/test_project_structure.py .....                    [ 76%]
tests/test_query_routing_and_grounded_qa.py ............. [100%]

======================== 52 passed, 1 warning in 2.51s ========================
```
All **52 automated unit and integration tests** pass locally with zero paid API keys or cloud dependencies.

### Step 2: Run the Controlled Evaluation Fixture Only
```powershell
pytest tests/test_query_routing_and_grounded_qa.py -k "TestControlledEvaluationFixture" -vv
```

### Step 3: Launch the Streamlit Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501`:
1. Use **Overview & Ingestion** to process documents and inspect the knowledge graph.
2. Navigate to **Ask SynapseAI (Grounded Q&A)** (or use the Q&A panel on the Overview page).
3. Click any of the quick example buttons or type your own question.
4. Inspect the classified intent badge, retrieval strategy, synthesized answer, verifiable citations, and expandable evidence accordion.

---

## ⚠️ Known Limitations & Troubleshooting

1. **OCR on Scanned PDFs**: Text extraction relies on PyMuPDF's direct text layer extraction. Non-searchable scanned image PDFs will display an advisory warning to upload digital PDFs or plain text files.
2. **Deterministic Classification Heuristics**: Intent classification uses deterministic patterns and entity matching. Unusually phrased or highly ambiguous queries may default to `FACTUAL`, but the conservative sufficiency checker safely prevents hallucinated answers by abstaining if evidence coverage is low.
3. **Local Embedding Memory**: The default ONNX embedding model (`all-MiniLM-L6-v2`) runs on CPU and requires ~79 MB download on first run. If offline, SynapseAI automatically uses the `LocalDeterministicEmbeddingFunction` fallback.
4. **Citation Scoping**: Citations reference the exact chunks and pages indexed during document ingestion. Changes to documents require re-running document processing and vector indexing to maintain synchronized provenance.

---

## 📄 License
MIT License. Developed for hackathons and open-source exploration.

---

## 🕸️ Part A: Knowledge Graph Architecture (Task 4)

### 1. Graph Representation ([`src/knowledge_graph.py`](file:///c:/Users/ayush/OneDrive/Desktop/Hackathons/Hypnexis/SynapseAI/src/knowledge_graph.py))
- **Core Engine**: Implemented using `networkx.DiGraph` to support directed relationships, path analysis, density metrics, and component isolation.
- **Node Structure**:
  - `entity_id`: Stable canonical slug from Task 3 (e.g. `ent_tech_knowledge_graph`).
  - `name`: Display label.
  - `entity_type`: Categorical type (`CONCEPT`, `TECHNOLOGY`, `ORGANIZATION`, `PERSON`, `LOCATION`, `METHOD`).
  - `aliases`: Variant surface forms.
  - `description`: Contextual explanation.
  - `mention_count`: Aggregated mention frequency across chunks.
  - `source_references`: Complete provenance references (`source_filename`, `page_number`, `chunk_id`, `text_snippet`).
  - `themes`: List of topic themes to which the entity belongs.
- **Directed Edge Structure**:
  - `subject_id` ➔ `object_id`.
  - `relationship_id`: Stable triple identifier.
  - `predicate`: Normalized predicate verb (e.g. `USES`, `IMPLEMENTS`, `EXTENDS`, `ENABLES`).
  - `confidence`: Extraction confidence score (0.0 to 1.0).
  - `evidence`: Supporting quotation records with filename, page, and chunk ID.

### 2. Strict Validation & Graph Statistics
- **Endpoint Integrity**: Every relationship is verified before edge insertion. If either `subject_id` or `object_id` does not exist in the entity node set, the edge is rejected and logged in `missing_endpoints`.
- **Theme Segregation**: Themes are stored as node attributes, ensuring artificial edges are **never** synthesized simply because two entities share a theme.
- **Topological Metrics**: The builder reports node count, edge count, number of weakly connected components, graph density, and distributions by entity type and theme.

### 3. Interactive Visualization with PyVis
- Rendered using `pyvis.network.Network` and embedded safely in Streamlit via `st.components.v1.html`.
- **Security & Safety**: All user-supplied document strings and entity descriptions are sanitized using `html.escape` to prevent XSS / script injection.
- **Controls**:
  - **Filter by Entity Type**: Narrow down canvas to specific categories.
  - **Filter by Topic Theme**: Visualize clusters belonging to a common theme.
  - **Focus Node (Ego Network)**: Isolate a selected entity and its immediate 1-hop in/out neighbors.
- **Node & Edge Tooltips**: Hover over nodes to inspect description, mentions, and source references. Hover over edges to view predicate, confidence, and exact quotation snippets.
- **Decoupled Persistence**: Graph data is exportable via `export_to_json()` and `export_to_dict()` so the representation can be stored locally or migrated to an external graph database (e.g., Neo4j) without rewriting extraction logic.

---

## 🗄️ Part B: Persistent Vector Database (Task 4)

### 1. Vector Store Architecture ([`src/vector_store.py`](file:///c:/Users/ayush/OneDrive/Desktop/Hackathons/Hypnexis/SynapseAI/src/vector_store.py))
- **Database Engine**: Uses **ChromaDB** (`chromadb.PersistentClient`) targeting a local directory (`./chroma_db`).
- **Repeat-Safe Upsert**: Uses `collection.upsert(ids=..., documents=..., metadatas=...)`. Re-indexing identical chunks updates existing records rather than producing duplicates.
- **Stale Record Management**: Supports `delete_document(source_filename)`, enabling clean removal of old chunks when documents are deleted or replaced.
- **Stored Metadata**: Each chunk record stores:
  - `source_filename`: Original document name.
  - `page_number`: 1-based page number (or `-1` for text files).
  - `chunk_index`: 0-based chunk index within document.
  - `char_count`: Character length of the passage.

### 2. Embedding Providers & Local Fallback
SynapseAI provides a dual-mode embedding configuration:
1. **Primary Local ONNX Embeddings (`DefaultEmbeddingFunction`)**:
   - Uses `all-MiniLM-L6-v2` via `onnxruntime`.
   - Runs **100% locally** on CPU without external API keys or cloud services.
   - Initial download (~79 MB) is cached automatically in `~/.cache/chroma/onnx_models`.
2. **Offline Deterministic Fallback (`LocalDeterministicEmbeddingFunction`)**:
   - Zero-dependency 256-dimensional term-frequency normalized hash vector.
   - Guaranteed to execute in `<1 ms` without internet access or model downloads.
   - Enables fast, offline unit testing.

### 3. Semantic Retrieval Test Panel
- Located directly in the Streamlit application under `Overview & Ingestion` and `Vector Search Explorer`.
- Allows users to enter conceptual queries (e.g. *"how convolutional filters detect edges"*).
- Adjust `top_k` results (1 to 10) and optionally filter by source document.
- Returns ranked passages with cosine relevance score (`Relevance: XX.X%`), distance metric, chunk ID, filename, and page number.

---

## 🌐 Persistent Graph Database Setup (Neo4j)

SynapseAI integrates an official **Neo4j** persistence layer (`src/graph_database.py`) using parameterized Cypher queries and formal schema constraints.

### 1. Schema Design
- **Node Labels**:
  - `(d:Document {filename: $filename, total_pages: $pages, total_chunks: $chunks})`
  - `(c:Chunk {chunk_id: $id, source_filename: $fn, page_number: $page, text: $text})`
  - `(e:Entity {entity_id: $id, name: $name, entity_type: $type, description: $desc})`
  - `(t:Theme {theme_id: $id, name: $name, description: $desc, document_filenames: [...]})`
- **Relationship Types**:
  - Structural: `(d)-[:CONTAINS]->(c)`
  - Mention: `(c)-[:MENTIONS {text_snippet: $snip}]->(e)`
  - Factual: `(s:Entity)-[:RELATED_TO {relationship_id: $id, predicate: $pred, confidence: $conf, evidence_text: $ev, source_filename: $fn, page_number: $pg}]->(o:Entity)`
  - Thematic Community: `(e:Entity)-[:BELONGS_TO]->(t:Theme)`
  - Document Representation: `(t:Theme)-[:REPRESENTS_DOCUMENT]->(d:Document)`
- **Uniqueness Constraints**: Enforced on `d.filename`, `c.chunk_id`, `e.entity_id`, and `t.theme_id`.

### 2. Running Neo4j Locally (Optional but Recommended)
SynapseAI runs out-of-the-box with an automatic in-memory NetworkX fallback. To enable persistent Neo4j storage, spin up a local instance via Docker:

```powershell
docker run --name synapse-neo4j -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/synapsepassword123 neo4j:latest
```

Configure `.env` (or set environment variables):
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=synapsepassword123
NEO4J_DATABASE=neo4j
```

If Neo4j is offline or unconfigured, SynapseAI clearly labels the database status as `Local Fallback (NetworkX)` in the UI banner and operates seamlessly without crashing.

---

## 🖥️ Split-Screen User Interface

The application features a responsive split-screen layout:
- **Left Panel (Uploader & Grounded Q&A)**:
  - Drag-and-drop document uploader supporting multi-document batches.
  - Chunk viewer, entity/relationship extraction controls, and Neo4j sync.
  - Chat input for factual, relational, thematic, and unanswerable queries.
  - Formatted answer card with verifiable `[1]`, `[2]` citations, source quotes, and page references.
- **Right Panel (Interactive Knowledge Graph Canvas)**:
  - Interactive VisJS physics graph with pan, zoom, and node dragging.
  - **Automatic Query Focus**: When a query executes, the graph automatically zooms to and highlights the relevant evidence nodes (bright cyan, enlarged) and relationship paths, while dimming unrelated nodes.
  - **Reset Focus Button**: Easily restores the global graph view.
  - **Node & Edge Inspection**: Click on any node or edge in the inspector to examine provenance, confidence, and source quotations.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (tested and verified on Python 3.10, 3.11, 3.12, and 3.14 on Windows)
- **Git**

---

### Step 1: Navigate to the Repository

```powershell
cd c:\Users\ayush\OneDrive\Desktop\Hackathons\Hypnexis\SynapseAI
```

---

### Step 2: Activate Your Virtual Environment

```powershell
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS / Linux (bash/zsh):
source .venv/bin/activate
```

---

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

Key packages included in `requirements.txt`:
- `streamlit` - Frontend user interface.
- `pymupdf` - Fast, in-memory PDF extraction.
- `networkx>=3.0.0` - Directed graph data structures and topological algorithms.
- `pyvis>=0.3.2` - Interactive VisJS physics graph visualization.
- `chromadb>=1.0.0` - Persistent local vector database.
- `neo4j>=5.0.0` - Official Neo4j Python driver.
- `pytest` - Automated testing framework.

---

### Step 4: Run the Complete Automated Test Suite

```powershell
# Run all tests with pytest
pytest -v
```

**Test Execution Results:**
```text
tests/test_document_ingestion.py ............             [ 19%]
tests/test_entity_extraction.py ...........               [ 34%]
tests/test_knowledge_graph_and_vector_store.py ............ [ 53%]
tests/test_neo4j_and_cross_doc_graph.py ...........       [ 71%]
tests/test_project_structure.py .....                    [ 79%]
tests/test_query_routing_and_grounded_qa.py ............. [100%]

======================== 63 passed, 1 warning in 2.93s ========================
```
All **63 automated tests** execute and pass with 100% success rate without requiring paid APIs or cloud infrastructure.

---

### Step 5: Launch the Streamlit Application

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in your browser:
1. **Upload Documents**: Upload `.pdf` or `.txt` files in the Left Panel and click **Process & Chunk Documents**.
2. **Extract Entities**: Click **Run Entity Extraction** to derive entities, relations, themes, and automatically persist records to Neo4j.
3. **Ask Questions**: In the chat input, ask questions:
   - *Factual*: *"What vector database does SynapseAI use?"*
   - *Relational*: *"How is SynapseAI connected to ChromaDB?"*
   - *Cross-Document Thematic*: *"What are the main themes across all uploaded documents?"*
   - *Unanswerable*: *"What is the top flight speed of a peregrine falcon?"*
4. **Observe Automatic Graph Highlighting**: Notice the Right Panel graph immediately zooms in to highlight the exact matched entities, themes, and relationship path behind the answer!

---

## ⚠️ Known Limitations
- **OCR**: Image-only scanned PDFs require external OCR tools (e.g. Tesseract) prior to ingestion. SynapseAI safely detects and warns when a PDF has zero extractable text.
- **Neo4j Offline Fallback**: If Neo4j credentials are not provided in `.env`, SynapseAI seamlessly falls back to NetworkX in-memory storage, clearly notifying the user in the UI.
- **Offline LLM Grounding**: When no external LLM API key (`GEMINI_API_KEY` / `OPENAI_API_KEY`) is configured, answer generation runs deterministically offline using exact source evidence sentences and structured path quotations.

---

## 📄 License
MIT License. Developed for the Hypnexis Hackathon.
