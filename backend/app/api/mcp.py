"""MCP (Model Context Protocol) HTTP endpoint."""

import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text
import httpx

router = APIRouter()
logger = logging.getLogger(__name__)


async def verify_api_key(client_id: str, client_secret: str) -> bool:
    """Verify API key credentials."""
    import hashlib
    from app.main import async_session_maker
    
    secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()
    
    async with async_session_maker() as session:
        result = await session.execute(
            text("""
                SELECT id FROM api_keys 
                WHERE client_id = :client_id 
                AND client_secret_hash = :secret_hash 
                AND is_active = 1
            """),
            {"client_id": client_id, "secret_hash": secret_hash}
        )
        return result.fetchone() is not None


# MCP Tools Definition
TOOLS = [
    {
        "name": "list_documents",
        "description": "List documents with optional filtering by subject_id, status, and limit",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject_id": {"type": "integer", "description": "Filter by subject ID"},
                "status": {"type": "string", "enum": ["queued", "processing", "done", "error"], "description": "Filter by status"},
                "limit": {"type": "integer", "default": 50, "description": "Maximum number of documents to return"}
            }
        }
    },
    {
        "name": "get_document",
        "description": "Get details of a specific document by ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "integer", "description": "The ID of the document to retrieve"}
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "analyze_document",
        "description": "Trigger analysis for a document",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "integer", "description": "The ID of the document to analyze"}
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "list_subjects",
        "description": "Search for subjects (persons, companies, etc.)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "context": {"type": "string", "enum": ["person", "company", "dossier", "other"], "description": "Filter by context"},
                "limit": {"type": "integer", "default": 50, "description": "Maximum number of results"}
            }
        }
    },
    {
        "name": "get_document_text",
        "description": "Get extracted text from a document",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "integer", "description": "The ID of the document"}
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "get_document_metadata",
        "description": "Get extracted metadata from a document",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "integer", "description": "The ID of the document"}
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "get_fraud_analysis",
        "description": "Get comprehensive fraud analysis for a document including PDF metadata, image forensics (ELA), text anomalies, and risk scoring",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "integer", "description": "The ID of the document to analyze"}
            },
            "required": ["document_id"]
        }
    },
    {
        "name": "search_documents",
        "description": "Search for documents by text content, document type, or risk score",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (searches in extracted text)"},
                "doc_type": {"type": "string", "description": "Filter by document type slug"},
                "min_risk_score": {"type": "number", "description": "Minimum risk score (0-100)"},
                "max_risk_score": {"type": "number", "description": "Maximum risk score (0-100)"},
                "subject_id": {"type": "integer", "description": "Filter by subject ID"},
                "limit": {"type": "integer", "default": 50, "description": "Maximum number of results"}
            }
        }
    },
    {
        "name": "train_classifier",
        "description": "Train the Naive Bayes classifier model (optionally for a specific model folder). Incremental training only processes new or changed files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Optional: name of the model folder to train (e.g. 'backoffice', 'mdoc')"},
                "incremental": {"type": "boolean", "default": False, "description": "If true, only train on new or changed files (incremental training). If false, retrain on all files."}
            }
        }
    },
    {
        "name": "train_bert_classifier",
        "description": "Train the BERT embeddings classifier for semantic document classification. Incremental training only processes new or changed files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Optional: name of the model folder to train"},
                "threshold": {"type": "number", "default": 0.7, "description": "Similarity threshold (0.5-0.95)"},
                "incremental": {"type": "boolean", "default": False, "description": "If true, only train on new or changed files (incremental training). If false, retrain on all files."}
            }
        }
    },
    {
        "name": "get_classifier_status",
        "description": "Get status and statistics of the trained classifier models",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Optional: name of the model to get status for"}
            }
        }
    },
    {
        "name": "get_bert_classifier_status",
        "description": "Get status of the BERT embeddings classifier including training info and model details",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Optional: name of the model to get status for"}
            }
        }
    },
    {
        "name": "list_high_risk_documents",
        "description": "List documents with high fraud risk scores, optionally filtered by risk level",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_risk_score": {"type": "number", "default": 50, "description": "Minimum risk score (0-100)"},
                "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"], "description": "Filter by risk level"},
                "subject_id": {"type": "integer", "description": "Filter by subject ID"},
                "limit": {"type": "integer", "default": 50, "description": "Maximum number of results"}
            }
        }
    },
    # Classification Policy & Signals
    {
        "name": "list_signals",
        "description": "List all classification signals (builtin and user-defined). Signals are reusable conditions for document eligibility rules.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["builtin", "user"], "description": "Filter by signal source"}
            }
        }
    },
    {
        "name": "get_signal",
        "description": "Get details of a specific signal by key",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The signal key (e.g., 'iban_present', 'date_count')"}
            },
            "required": ["key"]
        }
    },
    {
        "name": "list_document_types",
        "description": "List all document types with their classification policies",
        "inputSchema": {
            "type": "object",
            "properties": {
                "has_policy": {"type": "boolean", "description": "Filter to only types with custom policies"}
            }
        }
    },
    {
        "name": "get_document_type_policy",
        "description": "Get the classification policy for a specific document type including requirements, exclusions, and acceptance thresholds",
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "The document type slug (e.g., 'bankafschrift', 'factuur')"}
            },
            "required": ["slug"]
        }
    },
    {
        "name": "preview_eligibility",
        "description": "Test if sample text would be eligible for a document type based on its classification policy. Returns computed signals and eligibility result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "The document type slug to test against"},
                "text": {"type": "string", "description": "Sample text to evaluate"}
            },
            "required": ["slug", "text"]
        }
    },
    {
        "name": "compute_signals",
        "description": "Compute all classification signals for given text without checking against any policy. Useful for understanding what signals would be detected.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze for signals"}
            },
            "required": ["text"]
        }
    },
    # LLM Provider Tools
    {
        "name": "get_llm_status",
        "description": "Get the current LLM provider status including active provider (ollama/vllm), health status, and configuration",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "switch_llm_provider",
        "description": "Switch the active LLM provider between 'ollama' and 'vllm'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["ollama", "vllm"], "description": "The LLM provider to switch to"}
            },
            "required": ["provider"]
        }
    },
    {
        "name": "list_llm_models",
        "description": "List available models from the currently active LLM provider",
        "inputSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["ollama", "vllm"], "description": "Optional: specify provider to query (defaults to active provider)"}
            }
        }
    }
]


async def handle_tool_call(tool_name: str, arguments: Dict) -> Dict:
    """Handle a tool call and return the result."""
    from app.main import async_session_maker
    from app.config import settings
    import os
    from pathlib import Path
    
    try:
        async with async_session_maker() as session:
            if tool_name == "list_documents":
                conditions = []
                params = {"limit": arguments.get("limit", 50)}
                
                if arguments.get("subject_id"):
                    conditions.append("subject_id = :subject_id")
                    params["subject_id"] = arguments["subject_id"]
                if arguments.get("status"):
                    conditions.append("status = :status")
                    params["status"] = arguments["status"]
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                sql = f"""SELECT id, original_filename, status, doc_type_slug, created_at 
                          FROM documents WHERE {where_clause} 
                          ORDER BY created_at DESC LIMIT :limit"""
                
                result = await session.execute(text(sql), params)
                docs = [dict(row._mapping) for row in result.fetchall()]
                return {"content": [{"type": "text", "text": json.dumps({"documents": docs, "total": len(docs)}, indent=2, default=str)}]}
            
            elif tool_name == "get_document":
                doc_id = arguments.get("document_id")
                if not doc_id:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "document_id is required"})}]}
                
                result = await session.execute(
                    text("SELECT * FROM documents WHERE id = :id"),
                    {"id": doc_id}
                )
                doc = result.fetchone()
                if not doc:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Document {doc_id} not found"})}]}
                
                doc_dict = dict(doc._mapping)
                # Parse JSON fields
                for field in ['metadata_json', 'metadata_validation_json', 'metadata_evidence_json', 'risk_signals_json']:
                    if doc_dict.get(field) and isinstance(doc_dict[field], str):
                        try:
                            doc_dict[field] = json.loads(doc_dict[field])
                        except:
                            pass
                
                return {"content": [{"type": "text", "text": json.dumps(doc_dict, indent=2, default=str)}]}
            
            elif tool_name == "analyze_document":
                doc_id = arguments.get("document_id")
                if not doc_id:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "document_id is required"})}]}
                
                # Check document exists
                result = await session.execute(
                    text("SELECT id, status FROM documents WHERE id = :id"),
                    {"id": doc_id}
                )
                doc = result.fetchone()
                if not doc:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Document {doc_id} not found"})}]}
                
                # Queue for analysis
                await session.execute(
                    text("UPDATE documents SET status = 'queued', progress = 0, stage = NULL WHERE id = :id"),
                    {"id": doc_id}
                )
                await session.commit()
                
                return {"content": [{"type": "text", "text": json.dumps({"status": "queued", "document_id": doc_id, "message": "Document queued for analysis"})}]}
            
            elif tool_name == "list_subjects":
                conditions = []
                params = {"limit": arguments.get("limit", 50)}
                
                if arguments.get("query"):
                    conditions.append("name LIKE :query")
                    params["query"] = f"%{arguments['query']}%"
                if arguments.get("context"):
                    conditions.append("context = :context")
                    params["context"] = arguments["context"]
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                sql = f"SELECT * FROM subjects WHERE {where_clause} ORDER BY name LIMIT :limit"
                
                result = await session.execute(text(sql), params)
                subjects = [dict(row._mapping) for row in result.fetchall()]
                return {"content": [{"type": "text", "text": json.dumps({"subjects": subjects, "total": len(subjects)}, indent=2, default=str)}]}
            
            elif tool_name == "get_document_text":
                doc_id = arguments.get("document_id")
                if not doc_id:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "document_id is required"})}]}
                
                # Get document to find subject_id
                result = await session.execute(
                    text("SELECT subject_id FROM documents WHERE id = :id"),
                    {"id": doc_id}
                )
                doc = result.fetchone()
                if not doc:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Document {doc_id} not found"})}]}
                
                # Read text file
                text_path = Path(settings.data_dir) / "subjects" / str(doc.subject_id) / "documents" / str(doc_id) / "text" / "extracted.txt"
                if not text_path.exists():
                    return {"content": [{"type": "text", "text": json.dumps({"error": "Text not found. Document may not be processed yet."})}]}
                
                text_content = text_path.read_text(encoding="utf-8")
                return {"content": [{"type": "text", "text": json.dumps({"document_id": doc_id, "text": text_content, "length": len(text_content)}, indent=2)}]}
            
            elif tool_name == "get_document_metadata":
                doc_id = arguments.get("document_id")
                if not doc_id:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "document_id is required"})}]}
                
                result = await session.execute(
                    text("SELECT metadata_json FROM documents WHERE id = :id"),
                    {"id": doc_id}
                )
                doc = result.fetchone()
                if not doc:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Document {doc_id} not found"})}]}
                
                metadata = doc.metadata_json
                if metadata and isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        pass
                
                return {"content": [{"type": "text", "text": json.dumps({"document_id": doc_id, "metadata": metadata}, indent=2)}]}
            
            elif tool_name == "get_fraud_analysis":
                doc_id = arguments.get("document_id")
                if not doc_id:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "document_id is required"})}]}
                
                # Import here to avoid circular dependencies
                from app.services.fraud_detector import fraud_detector
                
                try:
                    # Get document
                    result = await session.execute(
                        text("""
                            SELECT d.*, s.name as subject_name
                            FROM documents d
                            LEFT JOIN subjects s ON d.subject_id = s.id
                            WHERE d.id = :id
                        """),
                        {"id": doc_id}
                    )
                    doc = result.fetchone()
                    if not doc:
                        return {"content": [{"type": "text", "text": json.dumps({"error": f"Document {doc_id} not found"})}]}
                    
                    doc_dict = dict(doc._mapping)
                    
                    # Load document file
                    doc_dir = Path(settings.data_dir) / "subjects" / str(doc_dict["subject_id"]) / "documents" / str(doc_id)
                    original_path = doc_dir / "original" / doc_dict["original_filename"]
                    
                    file_bytes = None
                    if original_path.exists():
                        file_bytes = original_path.read_bytes()
                    
                    # Load extracted text
                    extracted_text = None
                    text_path = doc_dir / "text" / "extracted.txt"
                    if text_path.exists():
                        extracted_text = text_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Load existing fraud signals
                    existing_signals = {}
                    llm_dir = doc_dir / "llm"
                    if llm_dir.exists():
                        for json_file in llm_dir.glob("*.json"):
                            try:
                                data = json.loads(json_file.read_text())
                                if "fpdf" in data:
                                    existing_signals["fpdf"] = data.get("fpdf")
                                if "fraud_signals" in data:
                                    existing_signals.update(data["fraud_signals"])
                            except Exception:
                                pass
                    
                    # Run fraud analysis with LLM client for signal enhancement
                    from app import main as app_main
                    detector = fraud_detector(llm_client=app_main.llm_client)
                    report = await detector.analyze_document(
                        file_bytes=file_bytes,
                        filename=doc_dict["original_filename"],
                        document_id=doc_id,
                        document_dir=doc_dir,
                        extracted_text=extracted_text,
                        classification_confidence=doc_dict.get("doc_type_confidence"),
                        existing_signals=existing_signals,
                    )
                    
                    return {"content": [{"type": "text", "text": json.dumps(report.to_dict(), indent=2, default=str)}]}
                except Exception as e:
                    logger.error(f"Error getting fraud analysis: {e}")
                    return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}
            
            elif tool_name == "search_documents":
                conditions = []
                params = {"limit": arguments.get("limit", 50)}
                
                if arguments.get("subject_id"):
                    conditions.append("d.subject_id = :subject_id")
                    params["subject_id"] = arguments["subject_id"]
                if arguments.get("doc_type"):
                    conditions.append("d.doc_type_slug = :doc_type")
                    params["doc_type"] = arguments["doc_type"]
                if arguments.get("min_risk_score") is not None:
                    conditions.append("d.risk_score >= :min_risk_score")
                    params["min_risk_score"] = arguments["min_risk_score"]
                if arguments.get("max_risk_score") is not None:
                    conditions.append("d.risk_score <= :max_risk_score")
                    params["max_risk_score"] = arguments["max_risk_score"]
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # If query provided, search in text files
                if arguments.get("query"):
                    # This is a simplified search - in production you might want full-text search
                    sql = f"""SELECT DISTINCT d.id, d.original_filename, d.doc_type_slug, d.risk_score, d.created_at
                              FROM documents d
                              WHERE {where_clause}
                              ORDER BY d.created_at DESC LIMIT :limit"""
                else:
                    sql = f"""SELECT d.id, d.original_filename, d.doc_type_slug, d.risk_score, d.created_at
                              FROM documents d
                              WHERE {where_clause}
                              ORDER BY d.created_at DESC LIMIT :limit"""
                
                result = await session.execute(text(sql), params)
                docs = [dict(row._mapping) for row in result.fetchall()]
                return {"content": [{"type": "text", "text": json.dumps({"documents": docs, "total": len(docs)}, indent=2, default=str)}]}
            
            elif tool_name == "train_classifier":
                model_name = arguments.get("model_name")
                incremental = arguments.get("incremental", False)
                try:
                    from app.services.doc_type_classifier import classifier_service
                    from app.api.classifier import _default_dataset_dir
                    import os
                    
                    if model_name:
                        model_dir = _default_dataset_dir() / model_name
                        if not model_dir.exists():
                            return {"content": [{"type": "text", "text": json.dumps({"error": f"Model folder '{model_name}' not found"})}]}
                        os.environ["MPROOF_ACTIVE_MODEL"] = model_name
                        os.environ["MPROOF_TRAINING_DATA_DIR"] = str(model_dir)
                    else:
                        # Clear environment variables for default model
                        if "MPROOF_ACTIVE_MODEL" in os.environ:
                            del os.environ["MPROOF_ACTIVE_MODEL"]
                        if "MPROOF_TRAINING_DATA_DIR" in os.environ:
                            del os.environ["MPROOF_TRAINING_DATA_DIR"]
                    
                    # Note: Incremental training is handled automatically by cache logic
                    # The incremental flag is informational for now
                    result = await classifier_service().train()
                    return {"content": [{"type": "text", "text": json.dumps({"status": "started", "model_name": model_name, "incremental": incremental, "message": "Training queued", "details": result}, indent=2, default=str)}]}
                except Exception as e:
                    logger.error(f"Error training classifier: {e}")
                    return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}
            
            elif tool_name == "train_bert_classifier":
                model_name = arguments.get("model_name")
                threshold = arguments.get("threshold", 0.7)
                incremental = arguments.get("incremental", False)
                try:
                    from app.services.bert_classifier import bert_classifier_service
                    from app.api.classifier import _default_dataset_dir
                    from pathlib import Path
                    
                    dataset_dir = None
                    if model_name:
                        model_dir = _default_dataset_dir() / model_name
                        if not model_dir.exists():
                            return {"content": [{"type": "text", "text": json.dumps({"error": f"Model folder '{model_name}' not found"})}]}
                        dataset_dir = model_dir
                    
                    # Note: BERT incremental training is handled automatically by cache logic
                    # The incremental flag is informational for now
                    result = await bert_classifier_service().train(model_name=model_name, threshold=threshold, dataset_dir=dataset_dir)
                    return {"content": [{"type": "text", "text": json.dumps({"status": "started", "model_name": model_name, "threshold": threshold, "incremental": incremental, "message": "BERT training queued", "details": result}, indent=2, default=str)}]}
                except Exception as e:
                    logger.error(f"Error training BERT classifier: {e}")
                    return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}
            
            elif tool_name == "get_classifier_status":
                model_name = arguments.get("model_name")
                try:
                    from app.services.doc_type_classifier import classifier_service
                    import os
                    
                    if model_name:
                        os.environ["MPROOF_ACTIVE_MODEL"] = model_name
                    
                    status = await classifier_service().status()
                    return {"content": [{"type": "text", "text": json.dumps(status, indent=2, default=str)}]}
                except Exception as e:
                    logger.error(f"Error getting classifier status: {e}")
                    return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}
            
            elif tool_name == "get_bert_classifier_status":
                model_name = arguments.get("model_name")
                try:
                    from app.services.bert_classifier import bert_classifier_service
                    status = bert_classifier_service().status(model_name=model_name)
                    return {"content": [{"type": "text", "text": json.dumps(status, indent=2, default=str)}]}
                except Exception as e:
                    logger.error(f"Error getting BERT classifier status: {e}")
                    return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}
            
            elif tool_name == "list_high_risk_documents":
                conditions = []
                params = {"limit": arguments.get("limit", 50)}
                min_risk = arguments.get("min_risk_score", 50)
                
                conditions.append("d.risk_score >= :min_risk")
                params["min_risk"] = min_risk
                
                if arguments.get("risk_level"):
                    # Map risk level to score ranges
                    risk_ranges = {
                        "LOW": (0, 25),
                        "MEDIUM": (25, 50),
                        "HIGH": (50, 75),
                        "CRITICAL": (75, 100)
                    }
                    if arguments["risk_level"] in risk_ranges:
                        min_r, max_r = risk_ranges[arguments["risk_level"]]
                        conditions.append("d.risk_score >= :risk_min AND d.risk_score < :risk_max")
                        params["risk_min"] = min_r
                        params["risk_max"] = max_r
                
                if arguments.get("subject_id"):
                    conditions.append("d.subject_id = :subject_id")
                    params["subject_id"] = arguments["subject_id"]
                
                where_clause = " AND ".join(conditions)
                sql = f"""SELECT d.id, d.original_filename, d.doc_type_slug, d.risk_score, d.risk_level, d.created_at
                          FROM documents d
                          WHERE {where_clause}
                          ORDER BY d.risk_score DESC, d.created_at DESC
                          LIMIT :limit"""
                
                result = await session.execute(text(sql), params)
                docs = [dict(row._mapping) for row in result.fetchall()]
                return {"content": [{"type": "text", "text": json.dumps({"documents": docs, "total": len(docs), "min_risk_score": min_risk}, indent=2, default=str)}]}
            
            # Classification Policy & Signals tools
            elif tool_name == "list_signals":
                source_filter = arguments.get("source")
                
                sql = """
                    SELECT 
                        key, label, description, signal_type,
                        CASE WHEN is_system = 1 THEN 'builtin' ELSE 'user' END as source,
                        COALESCE(compute_method, CASE WHEN is_system = 1 THEN 'builtin' ELSE 'keyword_set' END) as compute_kind,
                        config_json
                    FROM classification_signals
                """
                params = {}
                
                if source_filter:
                    if source_filter == "builtin":
                        sql += " WHERE is_system = 1"
                    else:
                        sql += " WHERE is_system = 0"
                
                sql += " ORDER BY is_system DESC, key"
                
                result = await session.execute(text(sql), params)
                signals = []
                for row in result.fetchall():
                    signal = dict(row._mapping)
                    if signal.get("config_json"):
                        try:
                            signal["config"] = json.loads(signal["config_json"])
                        except:
                            pass
                    del signal["config_json"]
                    signals.append(signal)
                
                return {"content": [{"type": "text", "text": json.dumps({
                    "signals": signals,
                    "total": len(signals),
                    "builtin_count": sum(1 for s in signals if s["source"] == "builtin"),
                    "user_count": sum(1 for s in signals if s["source"] == "user")
                }, indent=2)}]}
            
            elif tool_name == "get_signal":
                key = arguments.get("key")
                if not key:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "key is required"})}]}
                
                result = await session.execute(
                    text("""
                        SELECT 
                            key, label, description, signal_type,
                            CASE WHEN is_system = 1 THEN 'builtin' ELSE 'user' END as source,
                            COALESCE(compute_method, CASE WHEN is_system = 1 THEN 'builtin' ELSE 'keyword_set' END) as compute_kind,
                            config_json
                        FROM classification_signals WHERE key = :key
                    """),
                    {"key": key}
                )
                row = result.fetchone()
                if not row:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Signal '{key}' not found"})}]}
                
                signal = dict(row._mapping)
                if signal.get("config_json"):
                    try:
                        signal["config"] = json.loads(signal["config_json"])
                    except:
                        pass
                del signal["config_json"]
                
                return {"content": [{"type": "text", "text": json.dumps(signal, indent=2)}]}
            
            elif tool_name == "list_document_types":
                has_policy = arguments.get("has_policy")
                
                sql = """
                    SELECT slug, name, description, classification_policy_json
                    FROM document_types
                """
                
                if has_policy is True:
                    sql += " WHERE classification_policy_json IS NOT NULL AND classification_policy_json != '{}'"
                elif has_policy is False:
                    sql += " WHERE classification_policy_json IS NULL OR classification_policy_json = '{}'"
                
                sql += " ORDER BY name"
                
                result = await session.execute(text(sql))
                types = []
                for row in result.fetchall():
                    doc_type = dict(row._mapping)
                    policy_json = doc_type.get("classification_policy_json")
                    if policy_json:
                        try:
                            doc_type["policy"] = json.loads(policy_json)
                        except:
                            doc_type["policy"] = None
                    else:
                        doc_type["policy"] = None
                    del doc_type["classification_policy_json"]
                    types.append(doc_type)
                
                return {"content": [{"type": "text", "text": json.dumps({
                    "document_types": types,
                    "total": len(types),
                    "with_policy": sum(1 for t in types if t["policy"])
                }, indent=2)}]}
            
            elif tool_name == "get_document_type_policy":
                slug = arguments.get("slug")
                if not slug:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "slug is required"})}]}
                
                result = await session.execute(
                    text("SELECT slug, name, description, classification_policy_json FROM document_types WHERE slug = :slug"),
                    {"slug": slug}
                )
                row = result.fetchone()
                if not row:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Document type '{slug}' not found"})}]}
                
                doc_type = dict(row._mapping)
                policy = None
                if doc_type.get("classification_policy_json"):
                    try:
                        policy = json.loads(doc_type["classification_policy_json"])
                    except:
                        pass
                
                return {"content": [{"type": "text", "text": json.dumps({
                    "slug": doc_type["slug"],
                    "name": doc_type["name"],
                    "description": doc_type["description"],
                    "policy": policy,
                    "has_requirements": bool(policy and policy.get("requirements")),
                    "has_exclusions": bool(policy and policy.get("exclusions"))
                }, indent=2)}]}
            
            elif tool_name == "preview_eligibility":
                slug = arguments.get("slug")
                sample_text = arguments.get("text")
                
                if not slug:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "slug is required"})}]}
                if not sample_text:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "text is required"})}]}
                
                # Get document type policy
                result = await session.execute(
                    text("SELECT slug, name, classification_policy_json FROM document_types WHERE slug = :slug"),
                    {"slug": slug}
                )
                row = result.fetchone()
                if not row:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Document type '{slug}' not found"})}]}
                
                doc_type = dict(row._mapping)
                policy = {}
                if doc_type.get("classification_policy_json"):
                    try:
                        policy = json.loads(doc_type["classification_policy_json"])
                    except:
                        pass
                
                # Get all signals
                sig_result = await session.execute(text("""
                    SELECT key, signal_type,
                        CASE WHEN is_system = 1 THEN 'builtin' ELSE 'user' END as source,
                        COALESCE(compute_method, CASE WHEN is_system = 1 THEN 'builtin' ELSE 'keyword_set' END) as compute_kind,
                        config_json
                    FROM classification_signals
                """))
                signals_db = [dict(r._mapping) for r in sig_result.fetchall()]
                
                # Compute signals
                from app.services.signal_engine import compute_builtin_signal, compute_keyword_set_signal, compute_regex_set_signal
                
                computed_signals = {}
                for sig in signals_db:
                    key = sig["key"]
                    compute_kind = sig["compute_kind"]
                    config = {}
                    if sig.get("config_json"):
                        try:
                            config = json.loads(sig["config_json"])
                        except:
                            pass
                    
                    if compute_kind == "builtin":
                        computed_signals[key] = compute_builtin_signal(key, sample_text)
                    elif compute_kind == "keyword_set":
                        keywords = config.get("keywords", [])
                        match_mode = config.get("match_mode", "any")
                        computed_signals[key] = compute_keyword_set_signal(sample_text, keywords, match_mode)
                    elif compute_kind == "regex_set":
                        patterns = config.get("patterns", [])
                        match_mode = config.get("match_mode", "any")
                        computed_signals[key] = compute_regex_set_signal(sample_text, patterns, match_mode)
                
                # Evaluate eligibility
                requirements = policy.get("requirements", [])
                exclusions = policy.get("exclusions", [])
                
                def evaluate_condition(cond, signals):
                    sig_key = cond.get("signal")
                    op = cond.get("op", "==")
                    expected = cond.get("value")
                    actual = signals.get(sig_key)
                    
                    if actual is None:
                        return False
                    
                    if op == "==":
                        return actual == expected
                    elif op == "!=":
                        return actual != expected
                    elif op == ">=":
                        return actual >= expected
                    elif op == "<=":
                        return actual <= expected
                    elif op == ">":
                        return actual > expected
                    elif op == "<":
                        return actual < expected
                    return False
                
                # Check requirements (all must pass)
                requirements_met = all(evaluate_condition(r, computed_signals) for r in requirements) if requirements else True
                failed_requirements = [r for r in requirements if not evaluate_condition(r, computed_signals)]
                
                # Check exclusions (none should match)
                exclusions_triggered = [e for e in exclusions if evaluate_condition(e, computed_signals)]
                not_excluded = len(exclusions_triggered) == 0
                
                eligible = requirements_met and not_excluded
                
                return {"content": [{"type": "text", "text": json.dumps({
                    "slug": slug,
                    "name": doc_type["name"],
                    "eligible": eligible,
                    "requirements_met": requirements_met,
                    "not_excluded": not_excluded,
                    "failed_requirements": failed_requirements,
                    "triggered_exclusions": exclusions_triggered,
                    "computed_signals": computed_signals,
                    "policy": policy
                }, indent=2)}]}
            
            elif tool_name == "compute_signals":
                sample_text = arguments.get("text")
                if not sample_text:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "text is required"})}]}
                
                # Get all signals
                sig_result = await session.execute(text("""
                    SELECT key, label, signal_type,
                        CASE WHEN is_system = 1 THEN 'builtin' ELSE 'user' END as source,
                        COALESCE(compute_method, CASE WHEN is_system = 1 THEN 'builtin' ELSE 'keyword_set' END) as compute_kind,
                        config_json
                    FROM classification_signals
                """))
                signals_db = [dict(r._mapping) for r in sig_result.fetchall()]
                
                # Compute signals
                from app.services.signal_engine import compute_builtin_signal, compute_keyword_set_signal, compute_regex_set_signal
                
                results = []
                for sig in signals_db:
                    key = sig["key"]
                    compute_kind = sig["compute_kind"]
                    config = {}
                    if sig.get("config_json"):
                        try:
                            config = json.loads(sig["config_json"])
                        except:
                            pass
                    
                    value = None
                    if compute_kind == "builtin":
                        value = compute_builtin_signal(key, sample_text)
                    elif compute_kind == "keyword_set":
                        keywords = config.get("keywords", [])
                        match_mode = config.get("match_mode", "any")
                        value = compute_keyword_set_signal(sample_text, keywords, match_mode)
                    elif compute_kind == "regex_set":
                        patterns = config.get("patterns", [])
                        match_mode = config.get("match_mode", "any")
                        value = compute_regex_set_signal(sample_text, patterns, match_mode)
                    
                    results.append({
                        "key": key,
                        "label": sig["label"],
                        "type": sig["signal_type"],
                        "source": sig["source"],
                        "value": value
                    })
                
                # Sort: builtin first, then by key
                results.sort(key=lambda x: (0 if x["source"] == "builtin" else 1, x["key"]))
                
                return {"content": [{"type": "text", "text": json.dumps({
                    "text_length": len(sample_text),
                    "signals": results,
                    "summary": {
                        "total_signals": len(results),
                        "boolean_true": sum(1 for r in results if r["type"] == "boolean" and r["value"] is True),
                        "counts": {r["key"]: r["value"] for r in results if r["type"] == "count" and r["value"] and r["value"] > 0}
                    }
                }, indent=2)}]}
            
            # LLM Provider Tools
            elif tool_name == "get_llm_status":
                try:
                    from app.models.database import AppSetting
                    from sqlalchemy import select
                    
                    # Get active provider from database
                    result = await session.execute(
                        select(AppSetting).where(AppSetting.key == "llm_provider")
                    )
                    setting = result.scalar_one_or_none()
                    active_provider = setting.value if setting else "ollama"
                    
                    # Get config for active provider
                    config = settings.get_llm_config(active_provider)
                    
                    # Check health
                    from app.services.llm_client import LLMClient
                    client = LLMClient(active_provider)
                    is_healthy = await client.check_health()
                    
                    return {"content": [{"type": "text", "text": json.dumps({
                        "active_provider": active_provider,
                        "healthy": is_healthy,
                        "config": {
                            "base_url": config["base_url"],
                            "model": config["model"],
                            "timeout": config["timeout"],
                            "max_retries": config["max_retries"]
                        },
                        "available_providers": ["ollama", "vllm"]
                    }, indent=2)}]}
                except Exception as e:
                    logger.error(f"Error getting LLM status: {e}")
                    return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}
            
            elif tool_name == "switch_llm_provider":
                provider = arguments.get("provider")
                if not provider:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "provider is required"})}]}
                if provider not in ["ollama", "vllm"]:
                    return {"content": [{"type": "text", "text": json.dumps({"error": "provider must be 'ollama' or 'vllm'"})}]}
                
                try:
                    from app.models.database import AppSetting
                    from sqlalchemy import select
                    
                    # Get current provider
                    result = await session.execute(
                        select(AppSetting).where(AppSetting.key == "llm_provider")
                    )
                    setting = result.scalar_one_or_none()
                    old_provider = setting.value if setting else "ollama"
                    
                    # Update in database
                    if setting:
                        setting.value = provider
                    else:
                        from datetime import datetime
                        new_setting = AppSetting(
                            key="llm_provider",
                            value=provider,
                            description="Active LLM provider (ollama or vllm)",
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        session.add(new_setting)
                    await session.commit()
                    
                    # Refresh LLM client
                    import app.main as app_main
                    app_main.llm_client._refresh_config(provider)
                    
                    # Check health of new provider
                    from app.services.llm_client import LLMClient
                    client = LLMClient(provider)
                    is_healthy = await client.check_health()
                    
                    return {"content": [{"type": "text", "text": json.dumps({
                        "success": True,
                        "old_provider": old_provider,
                        "new_provider": provider,
                        "healthy": is_healthy,
                        "message": f"Switched from {old_provider} to {provider}"
                    }, indent=2)}]}
                except Exception as e:
                    logger.error(f"Error switching LLM provider: {e}")
                    return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}
            
            elif tool_name == "list_llm_models":
                provider = arguments.get("provider")
                
                try:
                    # If no provider specified, use active provider
                    if not provider:
                        from app.models.database import AppSetting
                        from sqlalchemy import select
                        result = await session.execute(
                            select(AppSetting).where(AppSetting.key == "llm_provider")
                        )
                        setting = result.scalar_one_or_none()
                        provider = setting.value if setting else "ollama"
                    
                    config = settings.get_llm_config(provider)
                    base_url = config["base_url"].rstrip('/')
                    
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        if provider == "vllm":
                            url = f"{base_url}/v1/models"
                            response = await client.get(url)
                            response.raise_for_status()
                            data = response.json()
                            models = [{"id": m.get("id"), "object": m.get("object")} for m in data.get("data", [])]
                        else:  # ollama
                            url = f"{base_url}/api/tags"
                            response = await client.get(url)
                            response.raise_for_status()
                            data = response.json()
                            models = [{"name": m.get("name"), "size": m.get("size"), "modified_at": m.get("modified_at")} for m in data.get("models", [])]
                    
                    return {"content": [{"type": "text", "text": json.dumps({
                        "provider": provider,
                        "base_url": base_url,
                        "models": models,
                        "count": len(models),
                        "configured_model": config["model"]
                    }, indent=2, default=str)}]}
                except httpx.HTTPStatusError as e:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"HTTP error {e.response.status_code}: {e.response.text}"})}]}
                except httpx.ConnectError:
                    return {"content": [{"type": "text", "text": json.dumps({"error": f"Cannot connect to {provider} at {base_url}"})}]}
                except Exception as e:
                    logger.error(f"Error listing LLM models: {e}")
                    return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}
            
            else:
                return {"content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}]}
    
    except Exception as e:
        logger.error(f"Error handling tool call {tool_name}: {e}")
        return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}


async def handle_mcp_request_logic(request: Request, body: dict):
    """Handle MCP JSON-RPC request logic and return response dict."""
    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")
    
    # Handle initialization - no auth required
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": "mproof",
                    "version": "1.0.0"
                }
            }
        }
    
    # For other methods, verify authentication from meta, params, or headers
    client_id = None
    client_secret = None
    
    # Try to get credentials from params._meta.client (MCP standard)
    meta = params.get("_meta", {})
    client_info = meta.get("client", {})
    client_id = client_info.get("id")
    client_secret = client_info.get("secret")
    
    # If not in meta, try direct params.client (some clients use this)
    if not client_id or not client_secret:
        direct_client = params.get("client", {})
        if isinstance(direct_client, dict):
            client_id = direct_client.get("id")
            client_secret = direct_client.get("secret")
    
    # If not in params, try headers (for HTTP transport)
    if not client_id or not client_secret:
        client_id = request.headers.get("X-Client-ID")
        client_secret = request.headers.get("X-Client-Secret")
    
    # Log authentication attempt
    if client_id:
        logger.debug(f"MCP auth attempt - client_id: {client_id[:8]}...")
    else:
        logger.warning("MCP request without client_id - authentication may fail")
    
    if client_id and client_secret:
        is_valid = await verify_api_key(client_id, client_secret)
        logger.debug(f"MCP auth result: {is_valid}")
        if not is_valid:
            logger.warning(f"MCP authentication failed for client_id: {client_id[:8]}...")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32001, "message": "Invalid API credentials"}
            }
    else:
        # For initialize and some methods, auth might not be required
        # But log a warning for other methods
        if method not in ["initialize", "notifications/initialized", "tools/list"]:
            logger.warning(f"MCP request without credentials for method: {method}")
    
    # Allow tools/list without auth for discovery (some clients need this)
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": TOOLS}
        }
    
    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        result = await handle_tool_call(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    
    elif method == "resources/list":
        # Resources are not exposed - return empty list
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"resources": []}
        }
    
    elif method == "resources/read":
        # Resources are not supported - use tools instead
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": "Resources are not supported. Use tools instead (e.g., get_document, get_document_text, get_document_metadata, get_fraud_analysis)."
            }
        }
    
    elif method == "notifications/initialized":
        # Acknowledgment, no response needed
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {}
        }
    
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }


async def handle_mcp_request(request: Request, body: dict = None):
    """Handle MCP JSON-RPC request and return JSONResponse."""
    if body is None:
        try:
            body = await request.json()
        except:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            })
    
    response_data = await handle_mcp_request_logic(request, body)
    
    # Check for authentication error to set proper status code
    if "error" in response_data and response_data.get("error", {}).get("code") == -32001:
        return JSONResponse(response_data, status_code=401)
    
    return JSONResponse(response_data)


# =============================================================================
# MCP Endpoints - Streamable HTTP Transport
# =============================================================================

@router.post("/mcp")
async def mcp_post(request: Request):
    """
    MCP JSON-RPC endpoint (POST).
    Handles all MCP requests and returns JSON responses.
    """
    return await handle_mcp_request(request)


@router.get("/mcp")
async def mcp_get(request: Request):
    """
    MCP SSE endpoint (GET) for server-to-client notifications.
    Returns SSE stream when Accept: text/event-stream is set.
    """
    import asyncio
    
    accept = request.headers.get("Accept", "")
    
    if "text/event-stream" in accept:
        async def sse_stream():
            # Initial connection confirmation
            yield "event: open\ndata: {}\n\n"
            # Keep alive with heartbeats
            try:
                while True:
                    await asyncio.sleep(30)
                    yield ": ping\n\n"
            except asyncio.CancelledError:
                pass
        
        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    # Non-SSE GET returns server info
    return JSONResponse({
        "jsonrpc": "2.0",
        "result": {
            "serverInfo": {"name": "mproof", "version": "1.0.0"},
            "protocolVersion": "2024-11-05"
        }
    })
