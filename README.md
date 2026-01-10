# MProof

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14+-black?style=flat-square&logo=next.js&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLM-FF6600?style=flat-square&logo=llama&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=flat-square&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

Een document analyse platform met AI-powered classificatie en metadata extractie.

## Features

- **Document Upload**: PDF, afbeeldingen (JPG/PNG), en Office documenten (DOCX/XLSX)
- **Real-time Processing**: Server-Sent Events (SSE) voor live voortgang
- **OCR Integratie**: Automatische tekstextractie met kwaliteitsbeoordeling
- **LLM Analyse**: Ollama integratie voor document classificatie en metadata extractie
- **Risk Analyse**: Deterministische risicoscoring met uitlegbare signalen
- **Subject Organisatie**: Groepeer documenten per persoon, bedrijf, dossier of andere context
- **Configuratie UI**: Dynamisch document type beheer met aanpasbare velden
- **Moderne UI**: Glassmorphism design met responsive layout

## Architectuur

### Backend (Python FastAPI)
- **Database**: SQLite met SQLAlchemy 2.0 + Alembic migraties
- **Processing**: Asynchrone document pipeline met background job queue
- **Real-time**: SSE voor live voortgang updates
- **LLM**: Ollama HTTP API client

### Frontend (Next.js)
- **UI**: Glassmorphism design met Tailwind CSS
- **State**: TanStack Query voor caching en real-time updates
- **Icons**: Font Awesome

---

## Vereisten

### Systeem
- macOS / Linux
- Python 3.9+
- Node.js 18+ (20 aanbevolen)
- Ollama (voor lokale LLM)

---

## Ollama Installatie & Setup

MProof gebruikt [Ollama](https://ollama.ai) voor lokale LLM-powered document analyse. Volg deze stappen om Ollama te installeren en configureren.

### 1. Installeer Ollama

**macOS (Homebrew):**
```bash
brew install ollama
```

**macOS (Installer):**
Download van https://ollama.ai/download

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama Server

```bash
# Start de Ollama server (blijft draaien in terminal)
ollama serve
```

Of start als achtergrond service:
```bash
# macOS - Ollama draait automatisch na installatie via installer
# Linux - systemd service
sudo systemctl start ollama
```

### 3. Download het Mistral Model

```bash
# Download Mistral (aanbevolen, ~4GB)
ollama pull mistral

# Of een ander model naar keuze
ollama pull llama2
ollama pull codellama
```

### 4. Verifieer Installatie

```bash
# Check of Ollama draait
curl http://localhost:11434/api/tags

# Test het model
ollama run mistral "Hallo, werkt dit?"
```

### 5. Configureer MProof

Maak een `.env` bestand in de `backend/` directory:

```bash
cd backend
cp .env.example .env
```

Pas de waardes aan indien nodig:

```env
# Ollama server URL (standaard localhost)
OLLAMA_BASE_URL=http://localhost:11434

# Model naam (moet gedownload zijn)
OLLAMA_MODEL=mistral:latest

# Timeout in seconden (verhoog voor trage machines)
OLLAMA_TIMEOUT=60.0
```

### Ollama op een andere machine

Als Ollama op een andere server draait:

```env
# Voorbeeld: Ollama op een GPU server
OLLAMA_BASE_URL=http://192.168.1.100:11434
OLLAMA_MODEL=mistral:latest
```

Zorg dat de Ollama server toegankelijk is:
```bash
# Op de Ollama server, bind aan alle interfaces
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone <repository>
cd MProof

# Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Pas aan indien nodig

# Frontend setup
cd ../frontend
npm install
cp .env.example .env.local  # Pas aan indien nodig
```

### 2. Database Initialiseren

```bash
cd backend
source venv/bin/activate
alembic upgrade head
```

### 3. Start Applicatie

**Optie A: Automatisch script**
```bash
./start.sh
```

**Optie B: Handmatig starten**

Terminal 1 - Backend:
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

### 4. Open Applicatie

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Backend API**: http://localhost:8000/api

---

## Configuratie

### Backend (.env)

Zie `backend/.env.example` voor alle opties:

```env
# LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:latest
OLLAMA_TIMEOUT=60.0
OLLAMA_MAX_RETRIES=3

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# Storage
DATA_DIR=./data

# OCR
TESSERACT_CONFIG=--psm 6
```

### Frontend (.env.local)

```env
# Backend URL (zonder /api suffix)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## Document Processing Pipeline

### Stadia & Voortgang

1. **Sniffing (0-10%)**: MIME type detectie, SHA256 berekening
2. **Text Extraction (10-45%)**: OCR voor afbeeldingen/PDFs
3. **Classification (45-60%)**: LLM document type classificatie
4. **Metadata Extraction (60-85%)**: Schema-gedreven veld extractie
5. **Risk Analysis (85-100%)**: Deterministische risicoscoring

### Artifact Structuur

```
./data/subjects/{subject_id}/documents/{document_id}/
├── original/{filename}          # Origineel bestand
├── text/
│   ├── extracted.json          # OCR resultaten met pagina metadata
│   └── extracted.txt           # Gecombineerde tekst
├── llm/
│   ├── classification_*.txt    # Classificatie prompts/responses
│   └── extraction_*.txt        # Extractie prompts/responses
├── metadata/
│   ├── result.json             # Geëxtraheerde metadata
│   ├── validation.json         # Validatie fouten
│   └── evidence.json           # Evidence spans per veld
└── risk/
    └── result.json             # Risk analyse resultaten
```

---

## API Voorbeelden

### Health Check
```bash
curl http://localhost:8000/api/health
```

### Subject Aanmaken
```bash
curl -X POST http://localhost:8000/api/subjects \
  -H "Content-Type: application/json" \
  -d '{"name": "Jan Jansen", "context": "person"}'
```

### Document Uploaden
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "subject_id=1" \
  -F "file=@document.pdf"
```

### Real-time Updates (SSE)
```bash
curl -N http://localhost:8000/api/documents/1/events
```

---

## Troubleshooting

### Ollama Connectie Problemen

```bash
# Check of Ollama draait
curl http://localhost:11434/api/tags

# Check of model beschikbaar is
ollama list

# Herstart Ollama
pkill ollama
ollama serve
```

### OCR Problemen

```bash
# Check Tesseract installatie
tesseract --version

# Installeer op macOS
brew install tesseract

# Installeer op Ubuntu/Debian
sudo apt install tesseract-ocr
```

### Database Reset

```bash
cd backend
rm -f data/app.db
alembic upgrade head
```

---

## Development

### Backend Tests
```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

### Frontend Linting
```bash
cd frontend
npm run lint
```

---

## Licentie

Dit project is beschikbaar voor educatieve en demonstratie doeleinden.
