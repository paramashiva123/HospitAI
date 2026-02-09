# HospitAI: Agentic Retrieval-Augmented Generation (RAG) System 🏥

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B.svg)](https://streamlit.io)
[![CrewAI](https://img.shields.io/badge/CrewAI-1.9.3-orange.svg)](https://crewai.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Key Features](#-key-features--business-impact)
- [System Architecture](#-system-architecture)
- [Technical Stack](#-technical-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation--setup)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Safety & Reliability](#-safety--reliability)
- [Performance Optimization](#-performance-optimization)
- [License](#-license)

---

## Executive Summary

**HospitAI** is an advanced, multi-domain intelligence system designed to unify fragmented hospital data into a single, conversational interface. By leveraging **Agentic RAG** (Retrieval-Augmented Generation), the platform enables healthcare professionals and administrators to query complex clinical guides, patient records, and financial statements using natural language.

The system doesn't just "search" for keywords—it understands user intent, navigates a high-performance vector database, and uses a "crew" of AI agents to synthesize grounded, factual answers.

---

## 🌟 Key Features & Business Impact

### 1. **Multi-Domain Intelligence**
The system is partitioned into three specialized knowledge silos to ensure high precision and data isolation:

- **Doctor's Guide:** Instant access to clinical medicine guides and treatment protocols
- **Patient Data:** Secure retrieval of patient history, medical reports, and discharge summaries
- **Financial Data:** Analysis of hospital financial statements and private sector hospital reports

### 2. **Intelligent Query Routing**
Unlike standard AI, HospitAI uses an **Intent Classifier** that automatically identifies whether a question is about:
- Clinical medicine
- Financial analysis
- Patient-specific information

The classifier directs searches to the relevant secure partition for optimal precision.

### 3. **Agentic Workflow**
Utilizing the **CrewAI** framework, the system employs specialized agents with defined roles:

- **The Answerer:** Expert in crafting concise, grounded answers using only retrieved context to prevent AI "hallucinations"

### 4. **Smart Ingestion & Deduplication**
To maintain a "clean" knowledge base, the system includes:
- SHA-256 content hashing for deduplication
- SQLite-based tracking to ensure documents are only indexed once
- Automatic detection and skipping of duplicate PDFs

### 5. **Interactive Web Interface**
Beautiful Streamlit-powered UI with:
- Real-time chat interface
- System monitoring dashboard
- Query history tracking
- Status indicators

---

## 🏗️ System Architecture

### Data Flow

```
┌─────────────┐
│   PDF Data  │
│  (3 Domains)│
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Deduplication  │
│  (SHA-256 Hash) │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Text Splitting  │
│  (600 char)     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Embeddings    │
│ (MiniLM-L12-v2) │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Milvus Storage  │
│  (3 Partitions) │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  User Query     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│Intent Classifier│
│  (Llama 3.3)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Vector Search   │
│  (Top-K: 5)     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  CrewAI Agent   │
│ (Qwen3-VL-30B)  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Grounded Answer │
└─────────────────┘
```

### Key Components

1. **Ingestion Pipeline:** PDFs → Deduplication → Text Splitting → Embedding → Storage
2. **Query Pipeline:** User Input → Intent Classification → Vector Search → Agent Processing → Response
3. **Storage Layer:** Milvus (Vector DB) + SQLite (Deduplication Tracking)

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit 1.54.0 | Interactive web interface |
| **Orchestration** | CrewAI 1.9.3 | Multi-agent system coordination |
| **Vector Database** | Milvus (Local) | High-performance similarity search |
| **Primary LLM** | Qwen3-VL-30B | Answer generation |
| **Intent Classifier** | Llama-3.3-70B (Groq) | Query routing |
| **Embeddings** | all-MiniLM-L12-v2 | 384-dim sentence embeddings |
| **Deduplication** | SQLite3 + SHA-256 | Document hash tracking |
| **PDF Processing** | PyPDF, PDFPlumber | Document loading |
| **Text Splitting** | LangChain | Recursive character splitting |

---

## 📂 Project Structure

```
HospitAI/
├── frontend.py                    # Streamlit web application
├── medical.db                     # SQLite deduplication database
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (API keys)
├── README.md                     # This file
│
├── rag_crew/
│   ├── src/rag_crew/
│   │   ├── crew.py               # Core RAG logic, Milvus setup, agents
│   │   ├── main.py               # CLI entry point for testing
│   │   └── config/
│   │       ├── agents.yaml       # Agent role definitions
│   │       └── tasks.yaml        # Task descriptions
│   │
│   └── data/                     # Knowledge base
│       ├── doctor_guide/         # Medical PDFs
│       ├── financial_data/       # Hospital financial PDFs
│       └── patient_data/         # Patient record PDFs
│
└── milvus_data/                  # Milvus vector storage (auto-created)
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python:** 3.10 or higher
- **Milvus:** Standalone instance running on `127.0.0.1:19530`
- **HuggingFace Account:** For model access
- **Groq API Key:** For Llama-3.3-70B intent classification

### Step 1: Clone the Repository

```bash
git clone https://github.com/paramashiva123/HospitAI.git
cd HospitAI
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Milvus

Install and start Milvus standalone:

```bash
# Using Docker
docker pull milvusdb/milvus:latest
docker run -d --name milvus_standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest
```

Or follow the [official Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md).

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# HuggingFace
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Groq (for intent classification)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Adjust logging
LITELLM_DISABLE_LOGGING=true
LITELLM_LOG=ERROR
OTEL_SDK_DISABLED=true
```

### Step 5: Prepare Your Data

Place your PDF documents in the appropriate directories:

```bash
mkdir -p rag_crew/data/doctor_guide
mkdir -p rag_crew/data/financial_data
mkdir -p rag_crew/data/patient_data

# Copy your PDFs
cp /path/to/medical/guides/*.pdf rag_crew/data/doctor_guide/
cp /path/to/financial/reports/*.pdf rag_crew/data/financial_data/
cp /path/to/patient/records/*.pdf rag_crew/data/patient_data/
```

### Step 6: Initialize the System

The system will automatically ingest documents on first run. This may take several minutes depending on the number of PDFs.

---

## 🎯 Usage

### Web Interface (Recommended)

Launch the Streamlit application:

```bash
streamlit run frontend.py
```

The interface will open at `http://localhost:8501`. Features include:
- Natural language query input
- Real-time agent status updates
- Query history
- System monitoring dashboard

### Command Line Interface

For testing or scripting:

```bash
cd rag_crew/src
python -m rag_crew.main
```

Example interaction:
```
Hospital AGENTIC RAG

Question: What is the medical history of patient rogers?
[Agent searches patient_data partition...]
[Response with grounded medical information]

Question: List out some drug related problems.
[Agent searches doctor_guide partition...] if the intent of the query is 'others' then it searches all the partitions. 

```

---

## ⚙️ Configuration

### Adjusting System Parameters

Edit `crew.py` to modify:

```python
# Vector Database Settings
MILVUS_URI = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "hospital_rag"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

# Text Splitting
CHUNK_SIZE = 600        # Characters per chunk
CHUNK_OVERLAP = 150     # Overlap between chunks

# Retrieval
TOP_K = 5              # Number of similar chunks to retrieve

# LLM Settings
llm = LLM(
    model="huggingface/Qwen/Qwen3-VL-30B-A3B-Instruct",
    temperature=0.2,    # Lower = more deterministic
    max_tokens=512      # Maximum response length
)
```

### Adding New Domains

To add a new knowledge domain:

1. Add the partition name to `PARTITIONS` in `crew.py`:
```python
PARTITIONS = {
    "doctor_guide",
    "financial_data",
    "patient_data",
    "your_new_domain",  # Add here
}
```

2. Update the intent classifier prompt in the `intent_classifier()` function

3. Create the data directory: `mkdir -p rag_crew/data/your_new_domain`

4. Add PDFs and restart the system

---

## 🛡️ Safety & Reliability

### Grounded Responses Only

The system is designed with a **"Grounded-only"** policy:

- Agents are explicitly instructed to answer queries **only** using retrieved context
- No hallucination or speculation
- All responses are verifiable against source documents
- Sources are cited in responses

### Data Security

- **Partitioned Architecture:** Medical, financial, and patient data are stored in separate partitions
- **Hash-based Deduplication:** Prevents data redundancy while maintaining privacy
- **Local Deployment:** All data remains on-premises (Milvus runs locally)

### Error Handling

The system includes comprehensive error handling:
- Graceful degradation if specific partitions are empty
- Fallback to cross-partition search if no results found
- User-friendly error messages in the UI

---

## 📊 Performance Optimization

### Indexing Strategy

The system uses **HNSW (Hierarchical Navigable Small World)** indexing:
- Fast approximate nearest neighbor search
- Optimized for 384-dimensional vectors
- Inner Product (IP) metric for similarity

```python
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200},
    },
)
```

### Deduplication Benefits

- **Storage Efficiency:** Duplicate documents are detected and skipped
- **Faster Queries:** Smaller index means faster searches
- **Cost Savings:** Reduced embedding API calls during ingestion

### Recommended Hardware

- **CPU:** 4+ cores
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** SSD for Milvus data directory
- **GPU:** Optional, but recommended for faster embedding generation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CrewAI** for the multi-agent framework
- **Milvus** for high-performance vector storage
- **LangChain** for RAG utilities
- **Streamlit** for the beautiful UI
- **HuggingFace** for open-source models

---

## 📧 Contact & Support

- **Email:** kambalaparamashiva123@gmail.com

---

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Advanced citation tracking
- [ ] Export conversation history
- [ ] Voice input support
- [ ] Mobile application
- [ ] Integration with hospital EHR systems

---

**Built with ❤️ for healthcare professionals**
