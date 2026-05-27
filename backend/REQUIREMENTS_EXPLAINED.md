# Requirements.txt Uitleg

Dit document legt uit wat elk package in `requirements.txt` doet en welke systeemdependencies nodig zijn.

## Core Web Framework
- **fastapi==0.104.1** - Web framework (geen systeemdependencies)
- **uvicorn[standard]==0.24.0** - ASGI server (geen systeemdependencies)
- **pydantic==2.5.0** - Data validation (geen systeemdependencies)
- **pydantic-settings==2.1.0** - Settings management (geen systeemdependencies)

## Database
- **sqlalchemy==2.0.23** - ORM (geen systeemdependencies)
- **alembic==1.12.1** - Database migrations (geen systeemdependencies)
- **aiosqlite==0.19.0** - Async SQLite driver (geen systeemdependencies)
- **greenlet==3.0.3** - Coroutine support (geen systeemdependencies)

## HTTP & Networking
- **httpx==0.25.2** - HTTP client (geen systeemdependencies)
- **urllib3<2.0** - HTTP library (geen systeemdependencies)
- **python-multipart==0.0.6** - File upload support (geen systeemdependencies)

## Document Processing

### PDF Processing
- **pypdf==6.6.0** - Pure Python PDF library (geen systeemdependencies)
- **pdfminer.six==20221105** - PDF text extraction (geen systeemdependencies)
- **PyMuPDF==1.23.6** - Fast PDF rendering (geen systeemdependencies, maar kan sneller zijn met system libraries)

### OCR & Images
- **pytesseract==0.3.10** - Python wrapper voor Tesseract OCR
  - **SYSTEEMDEPENDENCY:** `tesseract-ocr` (en optioneel `tesseract-ocr-nld`)
- **Pillow>=10.0.0** - Image processing library
  - **SYSTEEMDEPENDENCIES:** 
    ```bash
    sudo apt install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
        libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev \
        libfribidi-dev libxcb1-dev
    ```

### Office Documents
- **python-docx==1.1.0** - DOCX processing (geen systeemdependencies)
- **openpyxl==3.1.2** - Excel processing (geen systeemdependencies)

## Machine Learning / AI

### BERT Classifier
- **huggingface-hub>=0.34.0,<1.0** - Model downloads (geen systeemdependencies)
- **sentence-transformers>=2.2.0** - BERT embeddings (geen systeemdependencies, maar downloadt models)
- **scikit-learn>=1.3.0** - ML algorithms (geen systeemdependencies)
- **numpy>=1.24.0** - Numerical computing (geen systeemdependencies)

> **Let op:** BERT models worden automatisch gedownload bij eerste gebruik (~500MB)

## Testing (Development Only)
- **pytest==7.4.3** - Testing framework
- **pytest-asyncio==0.21.1** - Async test support
- **pytest-mock==3.12.0** - Mocking support

## Wat wordt NIET geïnstalleerd via requirements.txt?

### System Packages (moet je handmatig installeren):
1. **Tesseract OCR** - `sudo apt install tesseract-ocr tesseract-ocr-nld`
2. **Pillow dependencies** - Zie hierboven
3. **Build tools** - `sudo apt install build-essential python3-dev` (voor sommige packages)

### Optioneel:
- **python-magic** - MIME type detection (optioneel, code valt terug op extension-based)
- **vLLM** - LLM server (optioneel, alleen als je vLLM gebruikt i.p.v. Ollama)
- **Ollama** - LLM server (optioneel, alleen als je LLM features gebruikt)

## Installatie Volgorde

1. **Installeer eerst systeemdependencies:**
   ```bash
   sudo apt install tesseract-ocr tesseract-ocr-nld \
       libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
       libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev \
       libfribidi-dev libxcb1-dev build-essential python3-dev
   ```

2. **Dan Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Samenvatting

**Geen systeemdependencies nodig voor:**
- FastAPI, Uvicorn, Pydantic
- SQLAlchemy, Alembic, SQLite
- HTTP clients
- Pure Python PDF libraries (pypdf, pdfminer)
- Office document libraries
- ML libraries (numpy, scikit-learn, sentence-transformers)

**Systeemdependencies WEL nodig voor:**
- **pytesseract** → `tesseract-ocr`
- **Pillow** → libjpeg, zlib, freetype, etc.

**Optioneel (niet in requirements.txt):**
- vLLM of Ollama (voor LLM features)
- python-magic (voor MIME detection)
