import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from sqlalchemy import text

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document as DocxDocument

from app.config import settings
from app.services.feature_extractor import preprocess_text_for_classification
from app.services.policy_loader import load_global_config

logger = logging.getLogger(__name__)

# Set environment variables to prevent oversubscription on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Dutch stopwords - common words that don't add classification value
DUTCH_STOPWORDS = frozenset([
    # Articles & pronouns
    'de', 'het', 'een', 'die', 'dat', 'deze', 'dit', 'hij', 'zij', 'wij', 'jullie',
    'hun', 'haar', 'hem', 'mij', 'jou', 'ons', 'hen', 'wie', 'wat', 'welke', 'welk',
    # Prepositions
    'van', 'in', 'op', 'te', 'aan', 'met', 'voor', 'door', 'over', 'bij', 'naar',
    'uit', 'tot', 'om', 'onder', 'tegen', 'tussen', 'zonder', 'binnen', 'buiten',
    # Conjunctions
    'en', 'of', 'maar', 'want', 'dus', 'omdat', 'als', 'dan', 'toen', 'terwijl',
    'hoewel', 'indien', 'tenzij', 'zodra', 'voordat', 'nadat', 'zodat',
    # Common verbs
    'is', 'zijn', 'was', 'waren', 'ben', 'bent', 'heeft', 'hebben', 'had', 'hadden',
    'wordt', 'worden', 'werd', 'werden', 'kan', 'kunnen', 'kon', 'konden', 'zal',
    'zullen', 'zou', 'zouden', 'moet', 'moeten', 'moest', 'moesten', 'mag', 'mogen',
    'wil', 'willen', 'wilde', 'wilden', 'gaat', 'gaan', 'ging', 'gingen', 'komt',
    'komen', 'kwam', 'kwamen', 'doet', 'doen', 'deed', 'deden', 'zegt', 'zeggen',
    # Adverbs & misc
    'niet', 'ook', 'nog', 'wel', 'al', 'er', 'hier', 'daar', 'waar', 'hoe', 'nu',
    'dan', 'toen', 'zo', 'toch', 'heel', 'erg', 'zeer', 'meer', 'veel', 'weinig',
    'alle', 'alles', 'iets', 'niets', 'iemand', 'niemand', 'elke', 'elk', 'ander',
    'andere', 'eigen', 'zelf', 'alleen', 'samen', 'verder', 'eerst', 'laatste',
    # Numbers (as words)
    'een', 'twee', 'drie', 'vier', 'vijf', 'zes', 'zeven', 'acht', 'negen', 'tien',
    # Common document words to filter
    'pagina', 'bladzijde', 'datum', 'naam', 'adres', 'www', 'http', 'https', 'com', 'org', 'net',
])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> Path:
    # backend/app/services/doc_type_classifier.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _default_dataset_dir() -> Path:
    return _project_root() / "data"


def _safe_slug_from_folder(name: str) -> Optional[str]:
    raw = unicodedata.normalize("NFKC", (name or "").strip().lower())
    raw = raw.replace("_", "-").replace(" ", "-")
    raw = re.sub(r"[^a-z0-9\-]+", "-", raw)
    raw = re.sub(r"-{2,}", "-", raw).strip("-")
    if not raw:
        return None
    if not re.match(r"^[a-z][a-z0-9\-]*$", raw):
        return None
    return raw


def _load_label_map(dataset_dir: Path) -> Dict[str, str]:
    # Optional mapping file to map folder names -> document_type_slug
    # Example: { "bank_statement": "bankafschrift" }
    path = dataset_dir / "label_map.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Failed to read label_map.json: {e}")
    return {}


def _compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _is_mostly_empty(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text or "")
    if not cleaned:
        return True
    alpha_numeric = len(re.findall(r"[a-zA-Z0-9]", cleaned))
    return (alpha_numeric / len(cleaned)) < 0.1


def _read_text_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        # Some datasets may contain latin-1-ish files
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


def _ocr_with_rotation_detection(img: Image.Image, set_rotation_callback: Optional[callable] = None) -> str:
    """Perform OCR on image, trying different rotations (0, 90, 180, 270) and return best result.
    
    Optimized: First tries 0°, only tries other rotations if 0° doesn't produce good results.
    
    Args:
        img: PIL Image to perform OCR on
        set_rotation_callback: Optional callback to set the rotation angle (for status updates)
        
    Returns:
        Best OCR text result from all rotations
    """
    results = []
    
    # First try 0° (no rotation) - most common case
    try:
        text_0 = pytesseract.image_to_string(img, config=settings.tesseract_config)
        alnum_count_0 = sum(1 for c in text_0 if c.isalnum())
        word_count_0 = len(text_0.split())
        score_0 = alnum_count_0 * 2 + word_count_0
        
        results.append({
            'angle': 0,
            'text': text_0,
            'score': score_0,
            'alnum_count': alnum_count_0,
            'word_count': word_count_0
        })
        
        logger.debug(f"OCR rotation 0°: {alnum_count_0} alnum chars, {word_count_0} words, score: {score_0}")
        
        # If 0° produces good results (reasonable amount of text), skip other rotations
        # Threshold: at least 50 alphanumeric characters and 10 words indicates readable text
        if alnum_count_0 >= 50 and word_count_0 >= 10:
            logger.debug(f"OCR 0° produces good results ({alnum_count_0} alnum, {word_count_0} words), skipping other rotations")
            if set_rotation_callback:
                set_rotation_callback(0)
            return text_0
        
    except Exception as e:
        logger.warning(f"OCR failed for rotation 0°: {e}")
    
    # If 0° didn't produce good results, try other rotations
    for angle in [90, 180, 270]:
        try:
            # Rotate image
            rotated_img = img.rotate(-angle, expand=True)  # Negative for counter-clockwise
            
            # Perform OCR
            text = pytesseract.image_to_string(rotated_img, config=settings.tesseract_config)
            
            # Score the result: count alphanumeric characters and words
            alnum_count = sum(1 for c in text if c.isalnum())
            word_count = len(text.split())
            
            # Prefer results with more alphanumeric characters and words
            score = alnum_count * 2 + word_count
            
            results.append({
                'angle': angle,
                'text': text,
                'score': score,
                'alnum_count': alnum_count,
                'word_count': word_count
            })
            
            logger.debug(f"OCR rotation {angle}°: {alnum_count} alnum chars, {word_count} words, score: {score}")
            
        except Exception as e:
            logger.warning(f"OCR failed for rotation {angle}°: {e}")
            continue
    
    if not results:
        logger.warning("All OCR rotations failed, returning empty string")
        return ""
    
    # Sort by score (descending) and return best result
    results.sort(key=lambda x: x['score'], reverse=True)
    best = results[0]
    
    if best['angle'] != 0:
        logger.info(f"Best OCR result found at {best['angle']}° rotation ({best['alnum_count']} alnum chars, {best['word_count']} words)")
        if set_rotation_callback:
            set_rotation_callback(best['angle'])
    elif set_rotation_callback:
        set_rotation_callback(0)
    
    return best['text']


def _extract_text_from_image(path: Path, set_rotation_callback: Optional[callable] = None) -> str:
    img = Image.open(path)
    return _ocr_with_rotation_detection(img, set_rotation_callback=set_rotation_callback)


def _extract_text_from_pdf(path: Path, skip_markers: Optional[List[Tuple[str, bool]]] = None, pdf_max_pages: Optional[int] = None, set_rotation_callback: Optional[callable] = None) -> str:
    doc = fitz.open(str(path))
    combined = ""
    try:
        max_pages = pdf_max_pages or int(os.getenv("MPROOF_TRAIN_PDF_MAX_PAGES", "5"))
        for page_num in range(min(len(doc), max_pages)):
            page = doc.load_page(page_num)
            text = page.get_text() or ""
            if len(text.strip()) < 200 or _is_mostly_empty(text):
                pix = page.get_pixmap(dpi=250)
                img = Image.open(BytesIO(pix.tobytes("png")))
                text = _ocr_with_rotation_detection(img, set_rotation_callback=set_rotation_callback)
            combined += text + "\n"
            
            # Check skip markers on combined text
            if skip_markers:
                for pattern, is_regex in skip_markers:
                    try:
                        if is_regex:
                            match = re.search(pattern, combined, re.IGNORECASE)
                            if match:
                                combined = combined[:match.start()].strip()
                                return combined
                        else:
                            pos = combined.lower().find(pattern.lower())
                            if pos != -1:
                                combined = combined[:pos].strip()
                                return combined
                    except re.error:
                        pass
    finally:
        doc.close()
    return combined


def _extract_text_from_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_text(path: Path, skip_markers: Optional[List[Tuple[str, bool]]] = None, pdf_max_pages: Optional[int] = None, set_rotation_callback: Optional[callable] = None) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return _read_text_file(path)
    if ext == ".pdf":
        return _extract_text_from_pdf(path, skip_markers=skip_markers, pdf_max_pages=pdf_max_pages, set_rotation_callback=set_rotation_callback)
    if ext in {".png", ".jpg", ".jpeg"}:
        return _extract_text_from_image(path, set_rotation_callback=set_rotation_callback)
    if ext == ".docx":
        return _extract_text_from_docx(path)
    raise ValueError(f"Unsupported training file type: {ext}")


def _tokenize(text: str, remove_stopwords: bool = True, normalize_pii: bool = True) -> List[str]:
    """
    Simple, robust tokenizer for OCR text.
    
    Args:
        text: The text to tokenize
        remove_stopwords: Whether to filter out common Dutch stopwords
        normalize_pii: Whether to normalize PII before tokenizing (for consistency with classification)
    
    Returns:
        List of tokens (lowercase, min 3 chars, alphanumeric only)
    """
    # Apply PII normalization if enabled (for consistent training/inference)
    if normalize_pii:
        try:
            global_policy = load_global_config()
            if global_policy.normalize_pii_for_classification:
                text = preprocess_text_for_classification(text, normalize_pii=True)
        except Exception:
            pass  # If policy loading fails, proceed without normalization
    
    norm = unicodedata.normalize("NFKC", text or "").lower()
    # Extract tokens: letters and underscores only (exclude numbers)
    # This allows __IBAN__, __DATE__, __AMOUNT__ placeholders but excludes pure numbers
    tokens = re.findall(r"[a-z_]{3,}", norm)  # Only letters and underscore, min 3 chars
    
    # Filter out tokens that are only underscores or contain only numbers
    tokens = [t for t in tokens if not t.replace("_", "").isdigit() and t != "___"]
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in DUTCH_STOPWORDS]
    
    return tokens


def _softmax_from_log_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {k: float(math.exp(v - max_score)) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


import math


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class NaiveBayesTextClassifier:
    def __init__(self, model: Dict[str, Any]):
        self.model = model

    @property
    def threshold(self) -> float:
        return float(self.model.get("threshold", 0.85))

    def predict(self, text: str, allowed_labels: Optional[Iterable[str]] = None) -> Optional[Prediction]:
        labels = self.model.get("labels") or []
        if allowed_labels is not None:
            allowed = set(allowed_labels)
            labels = [l for l in labels if l in allowed]

        if not labels:
            return None

        tokens = _tokenize(text)
        if not tokens:
            return None

        counts = Counter(tokens)
        alpha = float(self.model.get("alpha", 1.0))
        vocab_size = int(self.model.get("vocab_size", 1)) or 1
        class_doc_counts: Dict[str, int] = self.model.get("class_doc_counts", {})
        class_total_tokens: Dict[str, int] = self.model.get("class_total_tokens", {})
        class_token_counts: Dict[str, Dict[str, int]] = self.model.get("class_token_counts", {})

        total_docs = sum(int(v) for v in class_doc_counts.values()) or 1
        num_classes = len(labels) or 1

        log_scores: Dict[str, float] = {}
        for label in labels:
            doc_count = int(class_doc_counts.get(label, 0))
            prior = math.log((doc_count + alpha) / (total_docs + alpha * num_classes))

            denom = (int(class_total_tokens.get(label, 0)) + alpha * vocab_size) or 1.0
            token_counts = class_token_counts.get(label, {})

            score = prior
            for token, n in counts.items():
                c = int(token_counts.get(token, 0))
                score += int(n) * math.log((c + alpha) / denom)
            log_scores[label] = score

        probs = _softmax_from_log_scores(log_scores)
        if not probs:
            return None

        best_label, best_prob = max(probs.items(), key=lambda kv: kv[1])
        return Prediction(label=best_label, confidence=float(best_prob))


class ClassifierService:
    def __init__(self):
        self._model_dir = Path(settings.data_dir) / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Default model path (can be overridden per-training via env var)
        self.model_path = self._model_dir / "doc_type_classifier.json"
        self.index_path = self._model_dir / "doc_type_classifier_index.json"
        self.text_cache_dir = self._model_dir / "doc_type_text_cache"
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state variables
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()  # For sync thread updates
        self._running = False
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_summary: Optional[Dict[str, Any]] = None
        self._current_file: Optional[str] = None
        self._current_label: Optional[str] = None
        self._ocr_rotation: Optional[int] = None
        self._active_files: List[Dict[str, Any]] = []  # List of files currently being processed
        self._train_task: Optional[asyncio.Task] = None
        self._cancelled = threading.Event()  # Cancellation flag for sync thread

        self._classifier: Optional[NaiveBayesTextClassifier] = None
        self._classifier_mtime: Optional[float] = None

    def _get_model_paths(self) -> tuple:
        """Get model and index paths, checking for active model env var."""
        active_model = os.environ.get("MPROOF_ACTIVE_MODEL")
        if active_model:
            return (
                self._model_dir / f"doc_type_classifier_{active_model}.json",
                self._model_dir / f"doc_type_classifier_{active_model}_index.json"
            )
        return (self.model_path, self.index_path)

    def _status_unlocked(self) -> Dict[str, Any]:
        model_path, _ = self._get_model_paths()
        # Thread-safe read of status fields
        with self._sync_lock:
            current_file = self._current_file
            current_label = self._current_label
            ocr_rotation = self._ocr_rotation
            active_files = self._active_files.copy()  # Copy to avoid race conditions
        return {
            "running": self._running,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "last_error": self._last_error,
            "last_summary": self._last_summary,
            "model_path": str(model_path),
            "dataset_dir": str(_dataset_dir()),
            "current_file": current_file,
            "current_label": current_label,
            "ocr_rotation": ocr_rotation,
            "active_files": active_files,  # List of files being processed
        }

    def _load_index(self) -> Dict[str, Any]:
        _, index_path = self._get_model_paths()
        if not index_path.exists():
            return {"version": 1, "dataset_dir": None, "files": {}}
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("files"), dict):
                return data
        except Exception as e:
            logger.warning(f"Failed to load classifier index: {e}")
        return {"version": 1, "dataset_dir": None, "files": {}}

    def _save_index(self, index: Dict[str, Any]) -> None:
        _, index_path = self._get_model_paths()
        tmp = index_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        tmp.replace(index_path)

    def _save_model(self, model: Dict[str, Any]) -> None:
        model_path, _ = self._get_model_paths()
        tmp = model_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(model, f)
        tmp.replace(model_path)

    def _load_classifier_if_changed(self) -> Optional[NaiveBayesTextClassifier]:
        if not self.model_path.exists():
            self._classifier = None
            self._classifier_mtime = None
            return None

        mtime = self.model_path.stat().st_mtime
        if self._classifier is not None and self._classifier_mtime == mtime:
            return self._classifier

        try:
            with open(self.model_path, "r", encoding="utf-8") as f:
                model = json.load(f)
            if not isinstance(model, dict) or not model.get("labels"):
                self._classifier = None
                self._classifier_mtime = mtime
                return None
            self._classifier = NaiveBayesTextClassifier(model)
            self._classifier_mtime = mtime
            return self._classifier
        except Exception as e:
            logger.warning(f"Failed to load classifier model: {e}")
            self._classifier = None
            self._classifier_mtime = mtime
            return None

    def predict(self, text: str, allowed_labels: Optional[Iterable[str]] = None, model_name: str = None) -> Optional[Prediction]:
        """Predict document type. If model_name is provided, load that specific model."""
        if model_name:
            clf = self._load_model_by_name(model_name)
        else:
            clf = self._load_classifier_if_changed()
        if not clf:
            return None
        pred = clf.predict(text, allowed_labels=allowed_labels)
        if not pred:
            return None
        if pred.confidence < clf.threshold:
            return None
        return pred
    
    def predict_with_threshold_info(self, text: str, allowed_labels: Optional[Iterable[str]] = None, model_name: str = None) -> tuple[Optional[Prediction], Optional[float], Optional[Prediction]]:
        """
        Predict document type and return threshold info even if below threshold.
        Returns: (prediction_if_above_threshold, threshold, raw_prediction_below_threshold)
        """
        if model_name:
            clf = self._load_model_by_name(model_name)
        else:
            clf = self._load_classifier_if_changed()
        if not clf:
            return None, None, None
        pred = clf.predict(text, allowed_labels=allowed_labels)
        if not pred:
            return None, clf.threshold, None
        if pred.confidence < clf.threshold:
            return None, clf.threshold, pred  # Return threshold and raw prediction
        return pred, clf.threshold, None

    def _load_model_by_name(self, model_name: str) -> Optional[NaiveBayesTextClassifier]:
        """Load a classifier model by its name (e.g., 'backoffice', 'mdoc')."""
        model_dir = Path(settings.data_dir) / "models"
        
        # Handle "default" specially - use the base model file
        if model_name == "default":
            model_path = model_dir / "doc_type_classifier.json"
        else:
            model_path = model_dir / f"doc_type_classifier_{model_name}.json"
        
        if not model_path.exists():
            # Fallback to default model if named model doesn't exist
            default_path = model_dir / "doc_type_classifier.json"
            if model_name != "default" and default_path.exists():
                logger.info(f"Model '{model_name}' not found, falling back to default NB model")
                model_path = default_path
            else:
                logger.warning(f"Model file not found: {model_path}")
                return None
        
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                model = json.load(f)
            if not isinstance(model, dict) or not model.get("labels"):
                return None
            logger.info(f"Loaded NB classifier from {model_path}")
            return NaiveBayesTextClassifier(model)
        except Exception as e:
            logger.warning(f"Failed to load model '{model_name}': {e}")
            return None

    async def status(self) -> Dict[str, Any]:
        async with self._lock:
            return self._status_unlocked()

    async def train(self) -> Dict[str, Any]:
        should_start = False
        async with self._lock:
            if self._running:
                return self._status_unlocked()

            self._running = True
            self._started_at = _utc_now_iso()
            self._finished_at = None
            self._last_error = None
            self._last_summary = None
            self._current_file = None
            self._current_label = None
            self._ocr_rotation = None
            should_start = True
            status = self._status_unlocked()

        if should_start:
            self._cancelled.clear()  # Reset cancellation flag
            self._train_task = asyncio.create_task(self._train_background())

        return status

    async def _train_background(self) -> None:
        try:
            # Load skip markers and compute signature before thread
            from app.main import async_session_maker
            pdf_max_pages = int(os.getenv("MPROOF_TRAIN_PDF_MAX_PAGES", "5"))
            skip_markers = []
            
            async with async_session_maker() as session:
                result = await session.execute(
                    text("SELECT pattern, is_regex FROM skip_markers WHERE is_active = 1 ORDER BY pattern ASC, is_regex ASC")
                )
                rows = result.fetchall()
                skip_markers = [(row.pattern, bool(row.is_regex)) for row in rows]
            
            # Force sort for determinism
            skip_markers = sorted(skip_markers, key=lambda x: (x[0], int(x[1])))
            skip_sig = hashlib.sha256(json.dumps(skip_markers, sort_keys=True).encode()).hexdigest()[:8]
            
            # Store for training thread
            self._train_skip_markers = skip_markers
            self._train_skip_sig = skip_sig
            self._train_pdf_max_pages = pdf_max_pages
            
            summary = await asyncio.to_thread(self._train_sync)
            async with self._lock:
                self._last_summary = summary
                self._finished_at = _utc_now_iso()
        except asyncio.CancelledError:
            logger.info("Training task cancelled")
            async with self._lock:
                self._last_error = "Training cancelled"
                self._finished_at = _utc_now_iso()
            raise
        except Exception as e:
            logger.exception("Classifier training failed")
            async with self._lock:
                self._last_error = str(e)
                self._finished_at = _utc_now_iso()
        finally:
            async with self._lock:
                self._running = False
                self._train_task = None

    def _train_sync(self) -> Dict[str, Any]:
        dataset_dir = _dataset_dir()
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise RuntimeError(f"Training dataset dir not found: {dataset_dir}")

        # Get model paths for this training session
        model_path, index_path = self._get_model_paths()
        logger.info(f"Training model to: {model_path}")

        label_map = _load_label_map(dataset_dir)
        index = self._load_index()
        index["dataset_dir"] = str(dataset_dir)

        allowed_ext = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".docx"}
        files_index: Dict[str, Any] = index.get("files", {}) if isinstance(index.get("files"), dict) else {}

        new_or_changed = 0
        used_examples: List[Tuple[str, str]] = []
        seen_rel_paths: set[str] = set()

        # Scan folders recursively - handle both first-level and nested subfolders
        # Collect all folders that should be treated as labels
        folders_to_process: List[Tuple[Path, str]] = []  # (folder_path, label)
        processed_folders: set[Path] = set()  # Track processed folders to avoid duplicates
        
        def collect_folders(root: Path, depth: int = 0, max_depth: int = 3):
            """Recursively collect folders that contain training files directly (not via subfolders)."""
            if depth > max_depth:
                return
            
            for folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if not folder.is_dir():
                    continue
                if folder.name.startswith("."):
                    continue
                if folder in processed_folders:
                    continue
                
                # Check if this folder contains training files DIRECTLY (not in subfolders)
                has_direct_files = False
                for path in folder.iterdir():
                    if path.is_file() and path.suffix.lower() in allowed_ext:
                        has_direct_files = True
                        break
                
                if has_direct_files:
                    # This folder has files directly - use it as a label
                    mapped = label_map.get(folder.name)
                    label = mapped or _safe_slug_from_folder(folder.name)
                    if label:
                        folders_to_process.append((folder, label))
                        processed_folders.add(folder)
                else:
                    # No direct files - check subfolders recursively
                    if depth < max_depth:
                        collect_folders(folder, depth + 1, max_depth)
        
        collect_folders(dataset_dir)
        
        # Process each folder
        for folder, label in folders_to_process:
            by_stem: Dict[str, List[Path]] = defaultdict(list)
            for path in folder.rglob("*"):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if ext not in allowed_ext:
                    continue
                by_stem[path.stem].append(path)

            def choose_best(paths: List[Path]) -> Optional[Path]:
                if not paths:
                    return None
                # Prefer text files if present
                for p in paths:
                    if p.suffix.lower() == ".txt":
                        return p
                # Then PDFs
                for p in paths:
                    if p.suffix.lower() == ".pdf":
                        return p
                # Then images
                for ext in (".png", ".jpg", ".jpeg"):
                    for p in paths:
                        if p.suffix.lower() == ext:
                            return p
                # Then docx
                for p in paths:
                    if p.suffix.lower() == ".docx":
                        return p
                return paths[0]

            for _stem, paths in by_stem.items():
                # Check cancellation flag
                if self._cancelled.is_set():
                    logger.info("Training cancelled, stopping file processing")
                    break

                chosen = choose_best(paths)
                if not chosen:
                    continue

                rel_path = chosen.relative_to(dataset_dir).as_posix()
                seen_rel_paths.add(rel_path)

                sha = _compute_sha256(chosen)
                entry = files_index.get(rel_path)

                skip_sig = getattr(self, '_train_skip_sig', '')
                pdf_max_pages = getattr(self, '_train_pdf_max_pages', 5)
                skip_markers = getattr(self, '_train_skip_markers', [])

                # Update status (thread-safe) - add to active files list
                with self._sync_lock:
                    self._current_file = chosen.name
                    self._current_label = label
                    self._ocr_rotation = None
                    # Add to active files list (keep last 10 for display)
                    file_info = {
                        "file": chosen.name,
                        "label": label,
                        "path": str(chosen.relative_to(dataset_dir))
                    }
                    self._active_files.append(file_info)
                    # Keep only last 10 active files to avoid memory issues
                    if len(self._active_files) > 10:
                        self._active_files = self._active_files[-10:]

                cache_name = None
                if isinstance(entry, dict) and entry.get("sha256") == sha and entry.get("skip_sig") == skip_sig and entry.get("pdf_max_pages") == pdf_max_pages and entry.get("text_cache"):
                    cache_name = str(entry.get("text_cache"))
                    cache_path = self.text_cache_dir / cache_name
                    if cache_path.exists():
                        text = _read_text_file(cache_path)
                        if text.strip():
                            used_examples.append((label, text))
                            continue

                # Check cancellation again before expensive OCR
                if self._cancelled.is_set():
                    logger.info("Training cancelled, stopping before OCR")
                    break

                # Need (re-)extract
                def set_rotation(angle: int):
                    with self._sync_lock:
                        self._ocr_rotation = angle
                
                try:
                    text = _extract_text(chosen, skip_markers=skip_markers, pdf_max_pages=pdf_max_pages, set_rotation_callback=set_rotation)
                except Exception as e:
                    logger.warning(f"Skipping training file (extract failed): {chosen} ({e})")
                    continue

                if not text.strip():
                    continue

                cache_name = f"{sha}.{skip_sig}.p{pdf_max_pages}.txt"
                cache_path = self.text_cache_dir / cache_name
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(text)

                if not isinstance(entry, dict) or entry.get("sha256") != sha or entry.get("skip_sig") != skip_sig or entry.get("pdf_max_pages") != pdf_max_pages or entry.get("label") != label:
                    new_or_changed += 1

                files_index[rel_path] = {
                    "sha256": sha,
                    "label": label,
                    "text_cache": cache_name,
                    "skip_sig": skip_sig,
                    "pdf_max_pages": pdf_max_pages,
                    "updated_at": _utc_now_iso()
                }
                used_examples.append((label, text))
                
                # Clear status after processing (thread-safe)
                with self._sync_lock:
                    if self._current_file == chosen.name:
                        self._current_file = None
                        self._current_label = None
                        self._ocr_rotation = None
                    # Remove from active files list
                    self._active_files = [f for f in self._active_files if f.get("file") != chosen.name]

        # Drop deleted files from index
        for rel_path in list(files_index.keys()):
            if rel_path not in seen_rel_paths:
                files_index.pop(rel_path, None)

        index["files"] = files_index
        index["updated_at"] = _utc_now_iso()
        self._save_index(index)

        model = self._train_model(used_examples)
        self._save_model(model)
        # Reset in-memory cache
        self._classifier = None
        self._classifier_mtime = None
        
        # Clear status
        with self._sync_lock:
            self._current_file = None
            self._current_label = None
            self._ocr_rotation = None
            self._active_files = []

        return {
            "ok": True,
            "dataset_dir": str(dataset_dir),
            "examples": len(used_examples),
            "labels": sorted(list(set(label for label, _ in used_examples))),
            "new_or_changed_files": new_or_changed,
            "model_path": str(model_path),
            "trained_at": _utc_now_iso(),
        }

    def _train_model(self, examples: List[Tuple[str, str]]) -> Dict[str, Any]:
        class_doc_counts: Counter[str] = Counter()
        class_total_tokens: Counter[str] = Counter()
        class_token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        vocab: set[str] = set()

        for label, text in examples:
            tokens = _tokenize(text)
            if not tokens:
                continue
            class_doc_counts[label] += 1
            doc_counts = Counter(tokens)
            class_token_counts[label].update(doc_counts)
            class_total_tokens[label] += sum(doc_counts.values())
            vocab.update(doc_counts.keys())

        labels = sorted(class_doc_counts.keys())
        vocab_size = max(1, len(vocab))

        # Serialize Counters to plain dicts (JSON)
        token_counts_json = {label: dict(counter) for label, counter in class_token_counts.items()}

        return {
            "version": 1,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "alpha": 1.0,
            "threshold": 0.85,
            "labels": labels,
            "vocab_size": vocab_size,
            "class_doc_counts": dict(class_doc_counts),
            "class_total_tokens": {k: int(v) for k, v in class_total_tokens.items()},
            "class_token_counts": token_counts_json,
        }


_service = ClassifierService()


def _dataset_dir() -> Path:
    # Use env override if provided, otherwise repo-root/data.
    env_path = os.environ.get("MPROOF_TRAINING_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return _default_dataset_dir()


def classifier_service() -> ClassifierService:
    return _service

