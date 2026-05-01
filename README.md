# 🏗️ BIS Standards Recommendation Engine

> **BIS × Sigma Squad AI Hackathon · IIT Tirupati**  
> *Accelerating MSE Compliance – Automating BIS Standard Discovery*

An AI-powered Retrieval-Augmented Generation (RAG) system that converts product descriptions into accurate BIS standard recommendations in seconds — helping Indian Micro and Small Enterprises navigate regulatory compliance effortlessly.

---

## 🎯 Problem Statement

Indian MSEs spend **weeks** manually identifying which Bureau of Indian Standards (BIS) regulations apply to their products. Our system reduces this to **seconds** using semantic search over BIS SP-21 (Building Materials).

---

## 🏛️ System Architecture

```
Product Description
        │
        ▼
┌─────────────────────┐
│   Text Embedder      │  all-MiniLM-L6-v2 (sentence-transformers)
│   (Dense Retrieval)  │
└─────────┬───────────┘
          │  query vector
          ▼
┌─────────────────────┐
│   FAISS Vector Store │  IndexFlatIP (cosine similarity)
│   BIS SP-21 Chunks   │  ~40 chunks from 20 standards
└─────────┬───────────┘
          │  top-10 candidates
          ▼
┌─────────────────────┐
│  Keyword Boost +     │  Domain-aware scoring boost
│  MMR Deduplication   │  Maximal Marginal Relevance
└─────────┬───────────┘
          │  top-5 unique standards
          ▼
┌─────────────────────┐
│   Claude (Anthropic) │  Rationale generation per standard
│   claude-sonnet-4    │
└─────────┬───────────┘
          │
          ▼
    Ranked Results (JSON)
```

---

## 📁 Repository Structure

```
bis-rag/
├── inference.py          ← Judge entry-point (mandatory)
├── eval_script.py        ← Evaluation script (mandatory)
├── app.py                ← Streamlit demo UI
├── requirements.txt
├── README.md
├── src/
│   └── rag_pipeline.py   ← Core RAG pipeline
└── data/
    ├── public_test_set.json   ← 10 sample queries + ground truth
    └── results/               ← Output files
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-team/bis-rag.git
cd bis-rag
pip install -r requirements.txt
```

### 2. Set API Key (optional – for LLM rationale)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run Inference (Judge Command)

```bash
python inference.py --input data/public_test_set.json --output data/results/team_results.json
```

### 4. Evaluate

```bash
python eval_script.py \
  --predictions data/results/team_results.json \
  --ground_truth data/public_test_set.json \
  --output data/results/metrics.json
```

### 5. Launch Demo UI

```bash
streamlit run app.py
```

---

## 📊 Chunking & Retrieval Strategy

### Chunking
Each BIS standard is chunked into **2 overlapping semantic windows**:

| Chunk Type | Content | Purpose |
|---|---|---|
| `summary` | Full description + applicability | Semantic similarity |
| `keywords` | Standard ID + usage keywords | Lexical coverage |

This dual-chunk approach ensures both semantic and keyword-level matches are captured.

### Retrieval Pipeline
1. **Dense retrieval** – `all-MiniLM-L6-v2` embeds the query; FAISS `IndexFlatIP` retrieves top-10 candidates by cosine similarity
2. **Keyword boost** – Additive score boost for domain keyword overlap (+0.02 per matched word)
3. **MMR deduplication** – Ensures top-5 results cover unique standards (no duplicate IS numbers)
4. **LLM rationale** – Claude generates one-sentence rationale per standard explaining relevance

### Why `all-MiniLM-L6-v2`?
- **Fast**: ~14,000 sentences/sec on CPU
- **Accurate**: Strong performance on domain-specific retrieval benchmarks
- **Lightweight**: 22M parameters, runs on standard hardware

---

## 🗂️ BIS Standards Covered

| Category | Standards |
|---|---|
| **Cement** | IS 269, IS 8112, IS 12269, IS 455, IS 1489, IS 4031, IS 3812 |
| **Concrete** | IS 516, IS 456, IS 10262, IS 9103, IS 1343, IS 13920, IS 2204 |
| **Steel** | IS 1786, IS 2062, IS 432 |
| **Aggregates** | IS 383, IS 2386, IS 2430 |
| **Structural Loads** | IS 875 |

---

## 📈 Expected Metrics

| Metric | Target | Expected |
|---|---|---|
| Hit Rate @3 | >80% | ~90% |
| MRR @5 | >0.7 | ~0.82 |
| Avg Latency | <5s | ~1.5s (CPU) |

---

## 🛠️ Environment

- **Python**: 3.10+
- **Hardware**: CPU sufficient (consumer GPU optional for faster embedding)
- **OS**: Linux / macOS / Windows
- **Key libraries**: `sentence-transformers`, `faiss-cpu`, `anthropic`, `streamlit`, `pdfplumber`

---

## 💡 Innovation Highlights

- **Dual-chunk strategy** captures both semantic meaning and keyword signals
- **Keyword boost** domain-tunes retrieval without fine-tuning
- **MMR deduplication** prevents redundant standards in top-k results
- **Graceful degradation** – works without FAISS (numpy fallback) and without API key (no rationale)
- **Zero re-training** – fully works with pre-trained embeddings out of the box

---

## 👥 Team

Built with ❤️ for the BIS × Sigma Squad AI Hackathon, IIT Tirupati.

---

*For support, reach out via the BIS X SS Hackathon WhatsApp Group.*
