import warnings
# Suppress Pydantic warning about "model_name" conflicting with protected namespace
warnings.filterwarnings("ignore", message=".*Field.*has conflict with protected namespace.*model_.*")

from fastapi import APIRouter, Query
from typing import Optional
import json
import os
from pathlib import Path
from app.services.doc_type_classifier import classifier_service, _default_dataset_dir
from app.services.bert_classifier import bert_classifier_service
from app.config import settings

router = APIRouter()


def _get_available_models() -> list:
    """Scan data directory for available model folders (folders containing document type folders)."""
    data_dir = _default_dataset_dir()
    models = []
    
    if not data_dir.exists():
        return models
    
    for item in data_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'models':
            # Check if it contains subfolders (document types)
            subfolders = [f for f in item.iterdir() if f.is_dir() and not f.name.startswith('.')]
            if subfolders:
                # Count training files in subfolders
                total_files = 0
                doc_types = []
                for subfolder in subfolders:
                    files = list(subfolder.glob('*.pdf')) + list(subfolder.glob('*.txt')) + \
                            list(subfolder.glob('*.docx')) + list(subfolder.glob('*.jpg')) + \
                            list(subfolder.glob('*.jpeg')) + list(subfolder.glob('*.png'))
                    if files:
                        total_files += len(files)
                        doc_types.append({
                            "slug": subfolder.name,
                            "file_count": len(files)
                        })
                
                if doc_types:
                    # Check if model file exists
                    model_file = data_dir / "models" / f"doc_type_classifier_{item.name}.json"
                    models.append({
                        "name": item.name,
                        "path": str(item),
                        "document_types": doc_types,
                        "total_files": total_files,
                        "is_trained": model_file.exists()
                    })
    
    return sorted(models, key=lambda x: x["name"])


@router.get("/classifier/models")
async def list_available_models():
    """List all available models in the data directory."""
    return {
        "models": _get_available_models(),
        "active_model": os.environ.get("MPROOF_ACTIVE_MODEL", "default"),
        "data_dir": str(_default_dataset_dir())
    }


@router.get("/classifier/status")
async def get_classifier_status():
    return await classifier_service().status()


@router.post("/classifier/train")
async def train_classifier(model_name: Optional[str] = Query(None, description="Name of model folder to train")):
    """Train the classifier. Optionally specify a model folder name."""
    if model_name:
        # Set environment variable for this training session
        model_dir = _default_dataset_dir() / model_name
        if not model_dir.exists():
            return {"ok": False, "error": f"Model folder '{model_name}' not found"}
        os.environ["MPROOF_TRAINING_DATA_DIR"] = str(model_dir)
        os.environ["MPROOF_ACTIVE_MODEL"] = model_name
    
    return await classifier_service().train()


@router.get("/classifier/training-details")
async def get_training_details(model_name: Optional[str] = Query(None, description="Name of the model to get details for")):
    """Get detailed training information including model stats, training files, and important tokens."""
    model_dir = Path(settings.data_dir) / "models"
    
    # Use model-specific files if model_name is provided
    if model_name:
        model_path = model_dir / f"doc_type_classifier_{model_name}.json"
        index_path = model_dir / f"doc_type_classifier_{model_name}_index.json"
    else:
        model_path = model_dir / "doc_type_classifier.json"
        index_path = model_dir / "doc_type_classifier_index.json"
    
    result = {
        "model_exists": model_path.exists(),
        "index_exists": index_path.exists(),
        "model": None,
        "index": None,
        "training_files_by_label": {},
        "important_tokens_by_label": {}
    }
    
    # Load model
    if model_path.exists():
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                model = json.load(f)
            result["model"] = {
                "version": model.get("version"),
                "created_at": model.get("created_at"),
                "updated_at": model.get("updated_at"),
                "threshold": model.get("threshold"),
                "alpha": model.get("alpha"),
                "vocab_size": model.get("vocab_size"),
                "labels": model.get("labels", []),
                "class_doc_counts": model.get("class_doc_counts", {}),
                "class_total_tokens": model.get("class_total_tokens", {})
            }
            
            # Get top tokens per label (most important for classification)
            class_token_counts = model.get("class_token_counts", {})
            for label, token_counts in class_token_counts.items():
                if isinstance(token_counts, dict):
                    # Sort by count and get top 50
                    sorted_tokens = sorted(
                        token_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:50]
                    result["important_tokens_by_label"][label] = [
                        {"token": token, "count": count}
                        for token, count in sorted_tokens
                    ]
        except Exception as e:
            result["model_error"] = str(e)
    
    # Load index
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            result["index"] = {
                "version": index.get("version"),
                "dataset_dir": index.get("dataset_dir"),
                "updated_at": index.get("updated_at"),
                "total_files": len(index.get("files", {}))
            }
            
            # Group files by label
            files = index.get("files", {})
            for rel_path, file_info in files.items():
                label = file_info.get("label")
                if label:
                    if label not in result["training_files_by_label"]:
                        result["training_files_by_label"][label] = []
                    result["training_files_by_label"][label].append({
                        "path": rel_path,
                        "sha256": file_info.get("sha256"),
                        "updated_at": file_info.get("updated_at")
                    })
        except Exception as e:
            result["index_error"] = str(e)
    
    return result


# ============== BERT Classifier Endpoints ==============

@router.get("/classifier/bert/status")
async def get_bert_status(model_name: Optional[str] = Query(None)):
    """Get the status of the BERT classifier."""
    return bert_classifier_service().status(model_name)


@router.post("/classifier/bert/train")
async def train_bert_classifier(
    model_name: Optional[str] = Query(None, description="Name of model folder to train"),
    threshold: float = Query(0.7, description="Minimum similarity threshold (0.0-1.0)")
):
    """Train the BERT classifier on documents."""
    if model_name:
        # Set training data directory
        model_dir = _default_dataset_dir() / model_name
        if not model_dir.exists():
            return {"ok": False, "error": f"Model folder '{model_name}' not found"}
        dataset_dir = model_dir
    else:
        dataset_dir = None
    
    return await bert_classifier_service().train(
        model_name=model_name,
        threshold=threshold,
        dataset_dir=dataset_dir
    )


@router.post("/classifier/bert/predict")
async def predict_with_bert(
    text: str,
    model_name: Optional[str] = Query(None),
):
    """Predict document type using BERT embeddings."""
    prediction = bert_classifier_service().predict(text, model_name=model_name)
    if prediction is None:
        return {"prediction": None, "message": "No prediction (model not trained or below threshold)"}
    
    return {
        "prediction": {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "all_scores": prediction.all_scores,
            "method": prediction.method,
        }
    }

