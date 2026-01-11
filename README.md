# MProof

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Node.js](https://img.shields.io/badge/node.js-18+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4.1-38bdf8.svg?logo=tailwindcss)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript)
![MCP](https://img.shields.io/badge/MCP-2024--11--05-00d4aa.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

AI-powered document analysis platform with classification, fraud detection, and MCP integration.

## Overview

MProof analyzes documents using multiple AI methods:

- **Naive Bayes Classifier** - Fast word-frequency based classification (~1ms)
- **BERT Embeddings** - Semantic understanding with deep learning (~100ms)
- **LLM Classification** - Ollama or vLLM-powered analysis as fallback (vLLM supports parallel processing)
- **Fraud Detection** - PDF metadata, image forensics (ELA), text anomaly detection
- **MCP Server** - Integration with AI assistants (Claude, Cursor, etc.)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.9+, FastAPI, SQLAlchemy, SQLite |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| AI/ML | BERT (sentence-transformers), Naive Bayes, Ollama/vLLM |
| OCR | Tesseract with rotation detection |
| Protocol | MCP (Model Context Protocol) |

## Features

### Document Processing
- Upload PDF, images (JPG/PNG), Office documents (DOCX/XLSX)
- Real-time processing with Server-Sent Events (SSE)
- Inline PDF preview with zoom

### Smart Text Extraction
| Feature | Description |
|---------|-------------|
| Multi-method PDF extraction | PyMuPDF → pypdf → pdfminer → OCR (automatic fallback) |
| Garbage text detection | Detects corrupted font encoding and triggers OCR |
| Rotation detection | OCR tries 0°, 90°, 180°, 270° and picks best result |
| Quality assessment | Automatic low/medium/high quality scoring |

**Garbage Detection Checks:**
- High ratio of special characters (`&'/!%.&'1&+$`) vs letters
- Too few recognizable words (3+ letter sequences)
- Control characters or unicode artifacts
- Lack of common Dutch/English stop words

### Classification
| Priority | Method | Description | Speed |
|----------|--------|-------------|-------|
| 1 | Naive Bayes | Word frequency classifier | ~1ms |
| 1 | BERT | Semantic embeddings (runs in parallel) | ~100ms |
| 2 | Deterministic | Keywords/regex matching | <1ms |
| 3 | LLM | Ollama/vLLM AI classification | ~2-5s |

**Multi-Model Support:**
- Train separate NB and BERT models per use case (e.g., `backoffice`, `mdoc`)
- Automatic fallback to default model if named model doesn't exist
- Confidence scores from both NB and BERT shown in UI

### Fraud Detection

**Risk Scoring:** 0-100% per document with signal-based analysis.

| Category | Signals |
|----------|---------|
| **PDF Metadata** | Suspicious generators (FPDF, TCPDF, wkhtmltopdf), timestamp mismatches, missing producer |
| **Image Forensics** | Error Level Analysis (ELA) for JPEG manipulation detection |
| **Text Anomalies** | Unicode manipulation, invisible characters, repeating patterns |
| **EXIF Analysis** | Photo editing software detection (Photoshop, GIMP) |

**ELA Detection:**
- Detects JPEG compression inconsistencies indicating manipulation
- Shows confidence percentage based on bright pixel ratio
- Visual heatmap highlighting suspicious regions
- Threshold: >20% bright pixels in error image = potential manipulation

### LLM Integration

**Dual Provider Support:**
| Feature | Ollama | vLLM |
|---------|--------|------|
| API | `/api/chat` | `/v1/chat/completions` (OpenAI-compatible) |
| Processing | Sequential | Parallel (faster for batches) |
| Port | 11434 | 8000 |
| Best For | Development | Production |

**Robust JSON Handling:**
- Automatic repair of truncated/malformed LLM responses
- Merges separate `{"data": ...}` and `{"evidence": ...}` objects
- Strips echoed instructions from responses
- Converts string `"null"` to proper `null` values
- Response time tracking (shown in UI)

**Provider Switching:**
- Active provider stored in database (persists across restarts)
- Switch via Settings page or API: `POST /api/llm/switch`
- Health check only queries active provider (not both)

### Skip Markers

Skip markers are text patterns that indicate where document processing should stop. When a skip marker is found, all text after it is ignored. This is useful for:

- **Disclaimers**: Legal text at the end of documents
- **Footers**: Repeated page footers with irrelevant info
- **Terms & Conditions**: Standard contract appendices
- **Signatures**: "Getekend te" or signature blocks

When a skip marker is applied, the document shows which pattern matched and at what position the text was truncated.

### Signal-Based Classification Policy

Documents are evaluated against configurable signals and policies:

**Built-in Signals:**
| Signal | Type | Description |
|--------|------|-------------|
| `iban_present` | boolean | Document contains IBAN |
| `date_count` | count | Number of dates (DD-MM-YYYY) |
| `amount_count` | count | Number of amounts (€X.XXX,XX) |
| `date_amount_row_count` | count | Lines with both date and amount |
| `line_count` | count | Non-empty lines |
| `token_count` | count | Word count |

**User-Defined Signals:** Create custom keyword or regex-based signals.

**Policy Example:**
```json
{
  "requirements": [
    {"signal": "iban_present", "op": "==", "value": true},
    {"signal": "date_amount_row_count", "op": ">=", "value": 5}
  ],
  "exclusions": [
    {"signal": "has_contract_terms", "op": "==", "value": true}
  ],
  "acceptance": {
    "trained_model": {"enabled": true, "min_confidence": 0.85},
    "deterministic": {"enabled": true},
    "llm": {"enabled": true}
  }
}
```

## Requirements

- macOS or Linux
- Python 3.9+
- Node.js 18+ (20 recommended)
- Ollama or vLLM (for LLM features - vLLM recommended for parallel processing)
- Tesseract OCR
- ~500MB RAM for BERT model (optional)

## Installation

### 1. Clone Repository

```bash
git clone <repository>
cd MProof
```

### 2. Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Install LLM Provider

Choose one or both:

**Option A: Ollama (Development)**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Download model
ollama pull mistral
```

**Option B: vLLM (Production - Parallel Processing)**
```bash
# Install vLLM
pip install vllm

# Run vLLM server (example)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --served-model-name llama3.2:3b \
  --port 8000
```

> **Note:** vLLM supports parallel requests, making it faster for batch processing. Ollama processes requests sequentially. You can switch between providers in the Settings page.

### 5. Install Tesseract

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Add Dutch language pack (optional but recommended)
brew install tesseract-lang  # macOS
sudo apt install tesseract-ocr-nld  # Ubuntu
```

## Configuration

### Backend (`backend/.env`)

MProof supports two LLM providers: **Ollama** and **vLLM**. Configure both in `.env`, then switch between them via the Settings page (the choice is stored in the database).

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT=180.0
OLLAMA_MAX_RETRIES=3

# vLLM Configuration (OpenAI-compatible)
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=llama3.2:3b
VLLM_TIMEOUT=180.0
VLLM_MAX_RETRIES=3

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# Storage
DATA_DIR=./data
```

**Starting vLLM:**

```bash
# Example: Run vLLM with a model
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --served-model-name llama3.2:3b \
  --port 8000
```

### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## Running

### Quick Start

```bash
./start.sh
```

### Manual Start

Terminal 1 - Backend:
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

### Access

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **API**: http://localhost:8000/api

## Multi-Model Architecture

Organize training data by model:

```
data/
├── backoffice/           # Model "backoffice"
│   ├── bankafschrift/    # Document type with sample PDFs
│   ├── factuur/
│   └── loonstrook/
└── mdoc/                 # Model "mdoc"
    ├── contract/
    └── rapport/
```

Train models via Settings page or API:

```bash
# Train Naive Bayes
curl -X POST "http://localhost:8000/api/classifier/train?model_name=backoffice"

# Train BERT
curl -X POST "http://localhost:8000/api/classifier/bert/train?model_name=backoffice"
```

**Model Fallback:**
- If a named NB model doesn't exist, falls back to default model
- BERT tries all available models and picks best confidence

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check (includes active LLM provider) |
| GET | `/api/documents` | List documents |
| GET | `/api/documents/{id}` | Get document |
| POST | `/api/upload` | Upload document |
| POST | `/api/documents/{id}/analyze` | Re-analyze document |
| GET | `/api/documents/{id}/events` | SSE stream |
| GET | `/api/documents/{id}/fraud-analysis` | Fraud analysis |

### LLM Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/llm/health` | Check LLM provider health |
| POST | `/api/llm/switch` | Switch active provider |

### Subjects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/subjects` | Search subjects |
| POST | `/api/subjects` | Create subject |
| GET | `/api/subjects/{id}` | Get subject |

### Document Types

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/document-types` | List types |
| POST | `/api/document-types` | Create type |
| PUT | `/api/document-types/{slug}` | Update type |
| DELETE | `/api/document-types/{slug}` | Delete type |

### Signals

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/signals` | List all signals |
| POST | `/api/signals` | Create user signal |
| PUT | `/api/signals/{key}` | Update signal |
| DELETE | `/api/signals/{key}` | Delete user signal |
| POST | `/api/signals/test` | Test signals on text |

### Classification Policy

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/document-types/{slug}/policy` | Get policy |
| PUT | `/api/document-types/{slug}/policy` | Update policy |
| DELETE | `/api/document-types/{slug}/policy` | Reset to default |
| POST | `/api/document-types/{slug}/policy/preview` | Preview eligibility |

### Classifier

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/classifier/status` | Get NB status |
| POST | `/api/classifier/train` | Train Naive Bayes |
| GET | `/api/classifier/bert/status` | Get BERT status |
| POST | `/api/classifier/bert/train` | Train BERT |

### API Keys

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/api-keys` | List keys |
| POST | `/api/api-keys` | Create key |
| DELETE | `/api/api-keys/{id}` | Delete key |

### Skip Markers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/skip-markers` | List all skip markers |
| POST | `/api/skip-markers` | Create skip marker |
| PUT | `/api/skip-markers/{id}` | Update skip marker |
| DELETE | `/api/skip-markers/{id}` | Delete skip marker |

## MCP Integration

MProof provides an HTTP-based Model Context Protocol server for AI assistants.

### Configuration

Add to your MCP client config (e.g., `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "mproof": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "X-Client-ID": "<your_client_id>",
        "X-Client-Secret": "<your_client_secret>"
      }
    }
  }
}
```

**Scopes:**
- `documents:read` - Read documents, text, metadata
- `documents:write` - Upload and analyze documents
- `subjects:read` - Search and read subjects
- `classification:read` - Read document types, signals, policies
- `fraud:read` - Read fraud analysis results

### Available Tools

**Document Management:**
- `list_documents` - List with filters
- `get_document` - Get details
- `get_document_text` - Get extracted text
- `get_document_metadata` - Get metadata
- `analyze_document` - Queue for processing
- `search_documents` - Search by text/type/risk

**Fraud Detection:**
- `get_fraud_analysis` - Full fraud analysis
- `list_high_risk_documents` - Filter by risk score/level

**Classification:**
- `list_signals` - List all signals
- `get_signal` - Get signal details
- `list_document_types` - List types with policies
- `get_document_type_policy` - Get policy
- `preview_eligibility` - Test text against policy
- `compute_signals` - Compute all signals for text

**Training:**
- `train_classifier` - Train Naive Bayes
- `train_bert_classifier` - Train BERT
- `get_classifier_status` - Naive Bayes status
- `get_bert_classifier_status` - BERT status

**Subjects:**
- `list_subjects` - Search subjects

## Document Processing Pipeline

| Stage | Progress | Description |
|-------|----------|-------------|
| Sniffing | 0-10% | MIME detection, SHA256 hash |
| Text Extraction | 10-45% | Multi-method with OCR fallback |
| Classification | 45-60% | NB + BERT → Deterministic → LLM |
| Metadata Extraction | 60-85% | Schema-driven field extraction |
| Risk Analysis | 85-100% | Fraud detection and scoring |

### Artifact Structure

```
data/subjects/{subject_id}/documents/{document_id}/
├── original/{filename}
├── text/
│   ├── extracted.json      # Per-page extraction info
│   └── extracted.txt       # Combined text
├── llm/
│   ├── classification_*.txt  # LLM classification artifacts
│   └── extraction_*.txt      # LLM extraction with timing
├── metadata/
│   ├── result.json         # Extracted fields
│   ├── validation.json     # Field validation results
│   └── evidence.json       # Quote evidence for fields
└── risk/
    └── result.json         # Fraud analysis results
```

## Troubleshooting

### Text Extraction Issues

**Garbage text (corrupted fonts):**
- System automatically detects and switches to OCR
- Look for log: `"Text extraction garbage text detected, using OCR"`

**Low OCR quality:**
- Ensure Tesseract language packs are installed
- Check image resolution (min 150 DPI recommended)
- System tries multiple rotations automatically

### Ollama Issues

```bash
# Check if running
curl http://localhost:11434/api/tags

# List models
ollama list

# Restart
pkill ollama && ollama serve
```

### vLLM Issues

```bash
# Check if running (OpenAI-compatible endpoint)
curl http://localhost:8000/v1/models

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Check GPU memory
nvidia-smi
```

### LLM Response Issues

**Truncated responses:**
- System automatically repairs malformed JSON
- Check logs for `"LLM response appears truncated"`

**Wrong JSON structure:**
- System merges separate data/evidence objects
- Converts string "null" to proper null values

### Database Reset

```bash
cd backend
rm -f data/app.db
python3 -m alembic upgrade head
```

## Development

### Run Tests

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

### Lint Frontend

```bash
cd frontend
npm run lint
```

## License

MIT License - See LICENSE file for details.
