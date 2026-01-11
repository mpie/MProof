"""
BERT Embeddings-based Document Type Classifier

This module provides semantic document classification using BERT embeddings
and cosine similarity. It offers better understanding of text meaning compared
to the traditional Naive Bayes bag-of-words approach.

Usage:
    from app.services.bert_classifier import bert_classifier_service
    
    # Train the model
    await bert_classifier_service().train(model_name="my-model")
    
    # Predict document type
    prediction = bert_classifier_service().predict(text, model_name="my-model")
"""

import asyncio
import json
import logging
import os
import pickle
import unicodedata
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy load sentence-transformers to avoid slow startup
_model = None
# Dutch-specific model optimized for sentence similarity
# Created by Netherlands Forensic Institute - excellent for Dutch financial/legal documents
_model_name = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"


def _is_model_downloaded() -> bool:
    """Check if the BERT model is already downloaded in cache."""
    try:
        from sentence_transformers import util
        import os
        from pathlib import Path
        
        # Sentence-transformers caches models in ~/.cache/torch/sentence_transformers/
        cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
        model_cache_dir = cache_dir / _model_name.replace("/", "_")
        
        # Check if model directory exists and has files
        if model_cache_dir.exists() and model_cache_dir.is_dir():
            # Check if there are model files (usually config.json, pytorch_model.bin, etc.)
            model_files = list(model_cache_dir.glob("*.json")) + list(model_cache_dir.glob("*.bin")) + list(model_cache_dir.glob("*.pt"))
            return len(model_files) > 0
        return False
    except Exception:
        # If we can't check, assume not downloaded
        return False


def _get_sentence_transformer():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading BERT model: {_model_name}")
            _model = SentenceTransformer(_model_name)
            logger.info("BERT model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    return _model


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BertPrediction:
    """Prediction result from BERT classifier."""
    label: str
    confidence: float
    all_scores: Dict[str, float]
    method: str = "bert-embeddings"


class BertClassifierService:
    """
    BERT-based document classifier using sentence embeddings.
    
    This classifier:
    1. Converts training documents to BERT embeddings
    2. Stores average embeddings per document type
    3. Classifies new documents by finding the most similar type
    """
    
    def __init__(self):
        self._model_dir = Path(settings.data_dir) / "models" / "bert"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = asyncio.Lock()
        self._running = False
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_summary: Optional[Dict[str, Any]] = None
        
        # Cached classifiers per model
        self._classifiers: Dict[str, Dict[str, Any]] = {}
    
    def _get_model_path(self, model_name: Optional[str] = None) -> Path:
        """Get the path to the BERT classifier model file."""
        if model_name:
            return self._model_dir / f"bert_classifier_{model_name}.pkl"
        return self._model_dir / "bert_classifier_default.pkl"
    
    def _load_classifier(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load a trained BERT classifier from disk."""
        cache_key = model_name or "default"
        
        # Check cache first
        if cache_key in self._classifiers:
            return self._classifiers[cache_key]
        
        model_path = self._get_model_path(model_name)
        if not model_path.exists():
            return None
        
        try:
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            self._classifiers[cache_key] = classifier
            return classifier
        except Exception as e:
            logger.error(f"Failed to load BERT classifier: {e}")
            return None
    
    def _save_classifier(self, classifier: Dict[str, Any], model_name: Optional[str] = None) -> None:
        """Save a trained BERT classifier to disk."""
        model_path = self._get_model_path(model_name)
        
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        # Update cache
        cache_key = model_name or "default"
        self._classifiers[cache_key] = classifier
        
        # Also save a JSON summary for inspection
        summary_path = model_path.with_suffix('.json')
        summary = {
            "trained_at": classifier.get("trained_at"),
            "labels": classifier.get("labels", []),
            "samples_per_label": classifier.get("samples_per_label", {}),
            "embedding_dim": classifier.get("embedding_dim"),
            "threshold": classifier.get("threshold", 0.7),
            "bert_model": _model_name,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get the status of the BERT classifier."""
        classifier = self._load_classifier(model_name)
        
        return {
            "running": self._running,
            "model_exists": classifier is not None,
            "model_downloaded": _is_model_downloaded(),
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "last_error": self._last_error,
            "last_summary": self._last_summary,
            "model_name": model_name,
            "bert_model": _model_name,
            "labels": classifier.get("labels", []) if classifier else [],
            "threshold": classifier.get("threshold", 0.7) if classifier else 0.7,
        }
    
    async def train(
        self,
        model_name: Optional[str] = None,
        threshold: float = 0.7,
        dataset_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Train the BERT classifier on documents.
        
        Args:
            model_name: Name of the model to train
            threshold: Minimum similarity score to accept a classification
            dataset_dir: Directory containing training data
        
        Returns:
            Training summary with statistics
        """
        async with self._lock:
            if self._running:
                logger.warning("BERT training already in progress")
                return {"error": "Training already in progress"}
            self._running = True
            self._started_at = _utc_now_iso()
            self._last_error = None
            logger.info(f"Starting BERT training for model: {model_name or 'default'}")
        
        try:
            logger.info("BERT training: Starting synchronous training in thread")
            result = await asyncio.to_thread(
                self._train_sync, model_name, threshold, dataset_dir
            )
            logger.info(f"BERT training completed successfully: {result.get('labels', [])}")
            self._last_summary = result
            return result
        except Exception as e:
            self._last_error = str(e)
            logger.exception(f"BERT training failed: {e}")
            return {"error": str(e)}
        finally:
            self._running = False
            self._finished_at = _utc_now_iso()
            logger.info("BERT training: Marked as finished")
    
    def _train_sync(
        self,
        model_name: Optional[str],
        threshold: float,
        dataset_dir: Optional[Path],
    ) -> Dict[str, Any]:
        """Synchronous training implementation."""
        from app.services.doc_type_classifier import (
            _project_root, _default_dataset_dir, _extract_text
        )
        
        # Determine dataset directory
        if dataset_dir is None:
            if model_name:
                dataset_dir = _default_dataset_dir() / model_name
            else:
                dataset_dir = _default_dataset_dir()
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        # Collect training data
        label_texts: Dict[str, List[str]] = {}
        file_count = 0
        
        for subdir in dataset_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            label = subdir.name
            texts = []
            
            for file_path in subdir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in {'.pdf', '.docx', '.doc', '.txt'}:
                    try:
                        text = _extract_text(file_path)
                        if text and len(text.strip()) > 50:
                            # Truncate to first 5000 chars for efficiency
                            texts.append(text[:5000])
                            file_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to extract text from {file_path}: {e}")
            
            if texts:
                label_texts[label] = texts
        
        if not label_texts:
            raise ValueError("No training data found")
        
        # Get BERT model
        logger.info("Loading BERT model (this may take 30-60 seconds on first run)...")
        model = _get_sentence_transformer()
        logger.info(f"BERT model loaded: {_model_name}")
        
        # Compute embeddings for each label
        label_embeddings: Dict[str, np.ndarray] = {}
        samples_per_label: Dict[str, int] = {}
        
        for label, texts in label_texts.items():
            logger.info(f"Computing embeddings for '{label}' ({len(texts)} documents)")
            
            # Encode all texts for this label
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            logger.info(f"Computed {len(embeddings)} embeddings for '{label}'")
            
            # Store mean embedding as the class centroid
            label_embeddings[label] = np.mean(embeddings, axis=0)
            samples_per_label[label] = len(texts)
            logger.info(f"Stored centroid for '{label}' (dim: {len(label_embeddings[label])})")
        
        # Create classifier object
        classifier = {
            "trained_at": _utc_now_iso(),
            "labels": list(label_embeddings.keys()),
            "embeddings": {k: v.tolist() for k, v in label_embeddings.items()},
            "samples_per_label": samples_per_label,
            "embedding_dim": len(next(iter(label_embeddings.values()))),
            "threshold": threshold,
            "bert_model": _model_name,
        }
        
        # Save classifier
        self._save_classifier(classifier, model_name)
        
        return {
            "success": True,
            "model_name": model_name,
            "labels": classifier["labels"],
            "samples_per_label": samples_per_label,
            "total_documents": file_count,
            "embedding_dim": classifier["embedding_dim"],
            "threshold": threshold,
        }
    
    def predict(
        self,
        text: str,
        model_name: Optional[str] = None,
        allowed_labels: Optional[List[str]] = None,
    ) -> Optional[BertPrediction]:
        """
        Predict document type using BERT embeddings.
        
        Args:
            text: The document text to classify
            model_name: Name of the model to use
            allowed_labels: Optional list of allowed labels to consider
        
        Returns:
            BertPrediction with label, confidence and all scores
        """
        classifier = self._load_classifier(model_name)
        if classifier is None:
            return None
        
        labels = classifier.get("labels", [])
        embeddings = classifier.get("embeddings", {})
        threshold = classifier.get("threshold", 0.7)
        
        if allowed_labels:
            labels = [l for l in labels if l in allowed_labels]
        
        if not labels:
            return None
        
        # Get BERT model and encode the text
        model = _get_sentence_transformer()
        
        # Truncate text for efficiency
        text = text[:5000] if len(text) > 5000 else text
        
        text_embedding = model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        
        # Compute similarity with each label
        scores: Dict[str, float] = {}
        for label in labels:
            if label not in embeddings:
                continue
            
            label_emb = np.array(embeddings[label])
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                label_emb.reshape(1, -1)
            )[0][0]
            
            # Convert similarity to 0-1 range (cosine similarity is already -1 to 1)
            scores[label] = float((similarity + 1) / 2)
        
        if not scores:
            return None
        
        # Find best match
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        
        # Check threshold
        if best_score < threshold:
            return None
        
        return BertPrediction(
            label=best_label,
            confidence=best_score,
            all_scores=scores,
        )


# Singleton instance
_bert_service: Optional[BertClassifierService] = None


def bert_classifier_service() -> BertClassifierService:
    """Get the singleton BERT classifier service."""
    global _bert_service
    if _bert_service is None:
        _bert_service = BertClassifierService()
    return _bert_service
