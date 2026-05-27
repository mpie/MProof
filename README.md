# MProof — Document Analyse Platform

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Node.js](https://img.shields.io/badge/node.js-18+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4.1-38bdf8.svg?logo=tailwindcss)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript)
![MCP](https://img.shields.io/badge/MCP-2024--11--05-00d4aa.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

MProof analyseert documenten met AI: classificatie, metadata-extractie, fraudedetectie en MCP-integratie voor AI-assistenten.

---

## Wat doet MProof?

| Functie | Beschrijving |
|---------|-------------|
| **Documentclassificatie** | Herkent het documenttype via Naive Bayes, BERT en/of LLM |
| **Metadata-extractie** | Haalt gestructureerde velden op (bedragen, namen, datums, …) via LLM |
| **Fraudedetectie** | Beoordeelt risico op basis van PDF-metadata, afbeeldingsanalyse en tekstafwijkingen |
| **MCP-server** | Koppeling met AI-assistenten zoals Claude of Cursor |
| **Meerdere modellen** | Train aparte classificatiemodellen per gebruik (bijv. `backoffice`, `mdoc`) |

---

## Technologie

| Onderdeel | Technologie |
|-----------|------------|
| Backend | Python 3.9+, FastAPI, SQLAlchemy, SQLite |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| AI/ML | BERT (sentence-transformers), Naive Bayes, Ollama / vLLM |
| OCR | Tesseract met automatische rotatiedetectie |
| Protocol | MCP (Model Context Protocol) |

---

## Functies

### Documentverwerking

- Upload PDF, afbeeldingen (JPG/PNG) en Office-bestanden (DOCX/XLSX)
- Verwerking in real-time via Server-Sent Events (SSE)
- Inline PDF-viewer met zoom en blauw gemarkeerde evidence

### Tekstextractie

MProof probeert meerdere methoden achtereenvolgens:

| Methode | Wanneer |
|---------|---------|
| PyMuPDF | Eerste keuze voor PDFs |
| pypdf | Terugval als PyMuPDF faalt |
| pdfminer | Tweede terugval |
| Tesseract OCR | Als de tekst onleesbaar of gecorrumpeerd is |

**Automatische kwaliteitscontrole:**
- Detecteert "rommel-tekst" door gecorrumpeerde lettertypen (hoge ratio bijzondere tekens, weinig herkenbare woorden)
- Probeert automatisch OCR als de tekst onbruikbaar is
- OCR test vier rotaties (0°, 90°, 180°, 270°) en kiest de beste

### Classificatie

| Prioriteit | Methode | Snelheid | Beschrijving |
|-----------|---------|----------|-------------|
| 1 | Deterministisch (STERK) | <1ms | Trefwoordregels — álle regels moeten matchen |
| 2 | Naive Bayes | ~1ms | Woordfrequentieclassificator |
| 2 | BERT | ~100ms | Semantische embeddings (parallel met NB, wint bij significant hogere score) |
| 3 | Deterministisch (terugval) | <1ms | Trefwoordregels — bij lage modelbetrouwbaarheid |
| 4 | LLM | ~2–5s | Ollama of vLLM als alle andere methoden falen |

**Meerdere modellen:**
- Trainbare NB- en BERT-modellen per toepassing (bijv. `backoffice`, `mdoc`)
- Automatische terugval naar standaardmodel als het gevraagde model niet bestaat
- Betrouwbaarheidsscores van zowel NB als BERT zichtbaar in de UI

### Fraudedetectie

Elk document krijgt een risicoscore van 0–100%.

| Categorie | Signalen |
|-----------|---------|
| **PDF-metadata** | Verdachte generatoren (FPDF, TCPDF, wkhtmltopdf), tijdstempelafwijkingen, ontbrekende producent |
| **Afbeeldingsforensics** | Error Level Analysis (ELA) — detecteert JPEG-manipulatie |
| **Tekstafwijkingen** | Unicode-manipulatie, onzichtbare tekens, herhalende patronen |
| **EXIF-analyse** | Detecteert bewerkingssoftware (Photoshop, GIMP) |

ELA werkt door het document opnieuw op te slaan als JPEG en het verschil te meten. Pixels die sterk afwijken duiden op bewerking. Bij meer dan 20% afwijkende pixels geeft MProof een waarschuwing.

### LLM-integratie

**Twee providers:**

| | Ollama | vLLM |
|-|--------|------|
| API-formaat | `/api/chat` | `/v1/chat/completions` (OpenAI-compatibel) |
| Verwerking | Sequentieel | Parallel (sneller bij meerdere verzoeken) |
| Standaardpoort | 11434 | 8000 |
| Aanbevolen voor | Ontwikkeling | Productie |

De actieve provider wordt opgeslagen in de database (blijft ingesteld na herstart). Wisselen via Instellingen of `POST /api/llm/switch`.

**Robuuste JSON-verwerking:**
- Herstelt automatisch afgekapte of misvormde LLM-antwoorden
- Voegt losse `{"data": …}` en `{"evidence": …}` objecten samen
- Verwijdert per ongeluk mee-geëchode instructies
- Converteert string `"null"` naar echte `null`

### Sla-over-markeringen (Skip Markers)

Definieer tekstpatronen waarná de documentverwerking stopt. Handig voor:
- Disclaimers en juridische teksten onderaan documenten
- Herhalende paginavoetteksten
- Standaard algemene voorwaarden
- Handtekeningblokken ("Getekend te …")

### Signalenbeleid

Documenten worden getoetst aan configureerbare signalen en beleidsregels.

**Ingebouwde signalen:**

| Signaal | Type | Beschrijving |
|---------|------|-------------|
| `iban_present` | boolean | Document bevat een IBAN |
| `date_count` | getal | Aantal datums (DD-MM-YYYY) |
| `amount_count` | getal | Aantal bedragen (€X.XXX,XX) |
| `date_amount_row_count` | getal | Regels met zowel datum als bedrag |
| `line_count` | getal | Niet-lege regels |
| `token_count` | getal | Aantal woorden |

Naast ingebouwde signalen kunt u eigen trefwoord- of regex-signalen aanmaken.

**Voorbeeld beleid:**
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

---

## Installatie

### Vereisten

- macOS of Linux
- Python 3.9+
- Node.js 18+ (20 aanbevolen)
- Ollama of vLLM
- Tesseract OCR
- ~500 MB RAM voor het BERT-model (optioneel)

### 1. Repository klonen

```bash
git clone <repository>
cd MProof
```

### 2. Backend installeren

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

# Installeer PyTorch zonder CUDA (aanbevolen op servers zonder GPU)
# Sla deze stap over als u een GPU heeft
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Overige afhankelijkheden
pip install -r requirements.txt
cp .env.example .env
```

> **GPU vs. CPU:** Zonder GPU is de CPU-versie van PyTorch ~2 GB kleiner en voldoende snel voor BERT-classificatie. Met een NVIDIA GPU kunt u de bovenste stap overslaan en installeert `sentence-transformers` automatisch de GPU-versie.

### 3. Frontend installeren

```bash
cd frontend
npm install
```

### 4. LLM-provider installeren

**Optie A: Ollama (ontwikkeling)**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Model downloaden
ollama pull mistral
```

**Optie B: vLLM (productie — parallelle verwerking)**
```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --served-model-name llama3.2:3b \
  --port 8000
```

### 5. Systeemsoftware (Ubuntu/Debian)

```bash
# Tesseract OCR
sudo apt install tesseract-ocr

# Nederlands taalpakket (aanbevolen)
sudo apt install tesseract-ocr-nld

# Afbeeldingsbibliotheken voor Pillow
sudo apt install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev \
    libfribidi-dev libxcb1-dev

# Bouwhulpmiddelen
sudo apt install build-essential python3-dev
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

---

## Configuratie

### Backend (`backend/.env`)

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT=180.0
OLLAMA_MAX_RETRIES=3

# vLLM (OpenAI-compatibel)
VLLM_BASE_URL=http://localhost:8000
VLLM_MODEL=llama3.2:3b
VLLM_TIMEOUT=180.0
VLLM_MAX_RETRIES=3

# Database
DATABASE_URL=sqlite+aiosqlite:///./data/app.db

# Opslag
DATA_DIR=./data
```

> De actieve provider en alle LLM-instellingen (URL, model, max tokens, contextvenster) worden opgeslagen in de database en zijn te wijzigen via **Instellingen → LLM** in de UI.

### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## Starten

### Snel starten

```bash
./start.sh
```

### Handmatig starten

**Terminal 1 — Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

### Adressen

| | Adres |
|-|-------|
| Frontend | http://localhost:3000 |
| API-documentatie | http://localhost:8000/docs |
| API | http://localhost:8000/api |

---

## Meerdere modellen trainen

Organiseer trainingsdata per model:

```
data/
├── backoffice/           # Model "backoffice"
│   ├── bankafschrift/    # Documenttype met voorbeeld-PDFs
│   ├── factuur/
│   └── loonstrook/
└── mdoc/                 # Model "mdoc"
    ├── contract/
    └── rapport/
```

Trainen via de UI (Instellingen) of API:

```bash
# Naive Bayes trainen
curl -X POST "http://localhost:8000/api/classifier/train?model_name=backoffice"

# BERT trainen
curl -X POST "http://localhost:8000/api/classifier/bert/train?model_name=backoffice"
```

Als een gevraagd model niet bestaat, valt het systeem automatisch terug op het standaardmodel.

---

## API-overzicht

### Documenten

| Methode | Endpoint | Beschrijving |
|---------|----------|-------------|
| GET | `/api/health` | Statuscontrole (incl. actieve LLM-provider) |
| GET | `/api/documents` | Documenten opvragen |
| GET | `/api/documents/{id}` | Één document opvragen |
| POST | `/api/upload` | Document uploaden |
| POST | `/api/documents/{id}/analyze` | Document opnieuw analyseren |
| GET | `/api/documents/{id}/events` | SSE-stream (live voortgang) |
| GET | `/api/documents/{id}/fraud-analysis` | Fraudeanalyse opvragen |

### LLM-instellingen

| Methode | Endpoint | Beschrijving |
|---------|----------|-------------|
| GET | `/api/llm/health` | Controleer beschikbaarheid van de provider |
| GET | `/api/llm/settings` | Huidige instellingen opvragen |
| PUT | `/api/llm/settings/{provider}` | Instellingen bijwerken |
| POST | `/api/llm/switch` | Wissel van actieve provider |

### Subjecten

| Methode | Endpoint | Beschrijving |
|---------|----------|-------------|
| GET | `/api/subjects` | Subjecten zoeken |
| POST | `/api/subjects` | Subject aanmaken |
| GET | `/api/subjects/{id}` | Één subject opvragen |

### Documenttypen

| Methode | Endpoint | Beschrijving |
|---------|----------|-------------|
| GET | `/api/document-types` | Lijst van typen |
| POST | `/api/document-types` | Type aanmaken |
| PUT | `/api/document-types/{slug}` | Type bijwerken |
| DELETE | `/api/document-types/{slug}` | Type verwijderen |

### Signalen

| Methode | Endpoint | Beschrijving |
|---------|----------|-------------|
| GET | `/api/signals` | Alle signalen opvragen |
| POST | `/api/signals` | Eigen signaal aanmaken |
| PUT | `/api/signals/{key}` | Signaal bijwerken |
| DELETE | `/api/signals/{key}` | Eigen signaal verwijderen |
| POST | `/api/signals/test` | Signalen testen op tekst |

### Classificatiebeleid

| Methode | Endpoint | Beschrijving |
|---------|----------|-------------|
| GET | `/api/document-types/{slug}/policy` | Beleid opvragen |
| PUT | `/api/document-types/{slug}/policy` | Beleid bijwerken |
| DELETE | `/api/document-types/{slug}/policy` | Beleid resetten naar standaard |
| POST | `/api/document-types/{slug}/policy/preview` | Beleid testen op tekst |

### Classificator

| Methode | Endpoint | Beschrijving |
|---------|----------|-------------|
| GET | `/api/classifier/status` | Status Naive Bayes |
| POST | `/api/classifier/train` | Naive Bayes trainen |
| GET | `/api/classifier/bert/status` | Status BERT |
| POST | `/api/classifier/bert/train` | BERT trainen |

### API-sleutels & Sla-over-markeringen

| Methode | Endpoint | Beschrijving |
|---------|----------|-------------|
| GET/POST | `/api/api-keys` | Sleutels beheren |
| DELETE | `/api/api-keys/{id}` | Sleutel verwijderen |
| GET/POST | `/api/skip-markers` | Markeringen beheren |
| PUT/DELETE | `/api/skip-markers/{id}` | Markering bijwerken of verwijderen |

---

## MCP-integratie

MProof biedt een HTTP-gebaseerde MCP-server voor gebruik met AI-assistenten zoals Claude of Cursor.

### Configuratie

Voeg toe aan uw MCP-clientconfiguratie (bijv. `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "mproof": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "X-Client-ID": "<uw_client_id>",
        "X-Client-Secret": "<uw_client_secret>"
      }
    }
  }
}
```

### Rechten (scopes)

| Scope | Toegang |
|-------|---------|
| `documents:read` | Documenten, tekst en metadata lezen |
| `documents:write` | Documenten uploaden en analyseren |
| `subjects:read` | Subjecten zoeken en lezen |
| `classification:read` | Documenttypen, signalen en beleid lezen |
| `fraud:read` | Fraudeanalyseresultaten lezen |

### Beschikbare tools

**Documentbeheer:**
- `list_documents` — Documenten opvragen met filters
- `get_document` — Details van één document opvragen
- `get_document_text` — Geëxtraheerde tekst opvragen
- `get_document_metadata` — Geëxtraheerde metadatavelden opvragen
- `analyze_document` — Document in de wachtrij zetten voor analyse (optioneel: `model_name` om een specifiek classificatiemodel te kiezen)
- `search_documents` — Zoeken op tekst, type of risico

**Fraudedetectie:**
- `get_fraud_analysis` — Volledige fraudeanalyse opvragen
- `list_high_risk_documents` — Filteren op risicoscore of -niveau

**Classificatie:**
- `list_signals` — Alle signalen opvragen
- `get_signal` — Details van één signaal opvragen
- `list_document_types` — Documenttypen met bijbehorend beleid opvragen
- `get_document_type_policy` — Beleid van een documenttype opvragen
- `preview_eligibility` — Tekst toetsen aan beleid
- `compute_signals` — Alle signalen berekenen voor een tekst

**Trainen:**
- `train_classifier` — Naive Bayes trainen
- `train_bert_classifier` — BERT trainen
- `get_classifier_status` — Status van Naive Bayes opvragen
- `get_bert_classifier_status` — Status van BERT opvragen
- `list_classifier_models` — Beschikbare classificatiemodellen opvragen (met bijbehorende documenttypen)

**Subjecten:**
- `list_subjects` — Subjecten zoeken

---

## Verwerkingspijplijn

| Fase | Voortgang | Beschrijving |
|------|-----------|-------------|
| Inspectie | 0–10% | MIME-detectie, SHA256-hash |
| Tekstextractie | 10–45% | Multi-methode met OCR-terugval |
| Classificatie | 45–60% | NB + BERT → Deterministisch → LLM |
| Metadata-extractie | 60–85% | Veldextractie via LLM |
| Risicoanalyse | 85–100% | Fraudedetectie en scoring |

### Hoe werkt de metadata-extractie?

#### Stap 1: Classificatie
Het systeem bepaalt eerst het documenttype (bijv. `factuur`, `loonstrook`, `taxatierapport`). Hiervoor worden Naive Bayes, BERT en eventueel een LLM ingezet.

#### Stap 2: Veldextractie via LLM
Op basis van het documenttype vraagt het systeem de LLM om specifieke velden te extraheren:
```
Voorbeeld "taxatierapport":
→ adres, bouwjaar, WOZ-waarde, marktwaarde, energielabel, oppervlakte
```
Relevante tekstfragmenten worden geselecteerd op basis van trefwoorden en budget, en in één LLM-aanroep verwerkt. De UI toont hoeveel fragmenten verwerkt worden (bijv. `Extractie (3/9)`).

#### Stap 3: Evidence zoeken
Voor elke geëxtraheerde waarde zoekt het systeem in het volledige document naar de exacte tekstlocatie:

| Matchtype | Beschrijving | Voorbeeld |
|-----------|-------------|---------|
| Exact | Letterlijke tekst | "P.C.M. Vastgoed Holding B.V." |
| Genormaliseerd | Witruimteverschillen (OCR) | "Calle Aloe 2A" vs. "Calle  Aloe  2A" |
| Numeriek | Opmaakvarianten | 100000 → "100.000" → "€ 100.000,-" |
| Hoofdletterverschil | Andere kapitalisatie | "AMSTERDAM" = "Amsterdam" |

#### Stap 4: Markering in PDF
Alle gevonden locaties worden blauw gemarkeerd in de PDF-viewer. Met navigatieknoppen springt u direct naar pagina's met evidence.

```
┌─────────────────────────────────────────────┐
│ 📄 document.pdf            • Met highlights │
├─────────────────────────────────────────────┤
│ ◀ 1/7 ▶           🔍- 100% 🔍+              │
├─────────────────────────────────────────────┤
│    ┌─────────────────────────────────────┐  │
│    │     PDF PAGINA-INHOUD               │  │
│    │                                     │  │
│    │  ████ € 100.000,- gemarkeerd ████   │  │
│    │                                     │  │
│    └─────────────────────────────────────┘  │
│  ┌─ Gevonden evidence: ──────────────────┐  │
│  │ "€ 100.000,-" p1 │ "P.C.M. Vast..." p1│  │
│  └───────────────────────────────────────┘  │
├─────────────────────────────────────────────┤
│ Pagina's met evidence: [p1] [p3] [p7]       │
└─────────────────────────────────────────────┘
```

### Bestandsstructuur per document

```
data/subjects/{subject_id}/documents/{document_id}/
├── original/{bestandsnaam}
├── text/
│   ├── extracted.json      # Extractie-info per pagina
│   └── extracted.txt       # Gecombineerde tekst
├── llm/
│   ├── classification_*.txt  # LLM-classificatieartifacten
│   └── extraction_*.txt      # LLM-extractie met tijdsduur
├── metadata/
│   ├── result.json         # Geëxtraheerde velden
│   ├── validation.json     # Validatieresultaten
│   └── evidence.json       # Bronverwijzingen per veld
└── risk/
    └── result.json         # Fraudeanalyseresultaten
```

---

## Problemen oplossen

### Tekstextractie

**Rommel-tekst (gecorrumpeerde lettertypen):**
- Het systeem schakelt automatisch over naar OCR
- Log: `"Text extraction garbage text detected, using OCR"`

**Slechte OCR-kwaliteit:**
- Controleer of de Tesseract-taalpakketten geïnstalleerd zijn
- Minimale beeldresolutie: 150 DPI
- Het systeem probeert automatisch vier rotaties

### Ollama

```bash
# Controleer of Ollama draait
curl http://localhost:11434/api/tags

# Beschikbare modellen
ollama list

# Herstarten
pkill ollama && ollama serve
```

### vLLM

```bash
# Controleer of vLLM draait
curl http://localhost:8000/v1/models

# Test een aanroep
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hallo"}],
    "max_tokens": 50
  }'

# GPU-geheugen controleren
nvidia-smi
```

### LLM-antwoorden afgekapt of te kort

- Stel **Contextvenster** in via **Instellingen → LLM** op de werkelijke contextlengte van uw model (bijv. `4096`).
- Het beschikbare aantal outputtokens = contextvenster − inputtokens − marge.
- Bij een verkeerde instelling corrigeert het systeem zichzelf eenmalig per sessie (log: `"Auto-corrected context_length to …"`).

### Verkeerde JSON-structuur van LLM

- Het systeem voegt losse `data`- en `evidence`-objecten automatisch samen.
- String `"null"` wordt automatisch omgezet naar `null`.

### Database resetten

```bash
cd backend
rm -f data/app.db
python3 -m alembic upgrade head
```

---

## Ontwikkeling

### Tests uitvoeren

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

### Frontend linten

```bash
cd frontend
npm run lint
```

---

## Licentie

MIT — zie het `LICENSE`-bestand voor details.
