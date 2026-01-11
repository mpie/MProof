import warnings
# Suppress Pydantic warning about "model_name" conflicting with protected namespace
# This is safe because we're using it as a query parameter, not a model field
warnings.filterwarnings("ignore", message=".*Field.*has conflict with protected namespace.*model_.*")

from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ContextEnum(str, Enum):
    person = "person"
    company = "company"
    dossier = "dossier"
    other = "other"


class FieldTypeEnum(str, Enum):
    string = "string"
    date = "date"
    number = "number"
    money = "money"
    currency = "currency"
    iban = "iban"
    enum = "enum"


class DocumentStatusEnum(str, Enum):
    pending = "pending"
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


# Subject schemas
class SubjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    context: ContextEnum


class SubjectResponse(BaseModel):
    id: int
    name: str
    context: ContextEnum
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SubjectSearchResponse(BaseModel):
    subjects: List[SubjectResponse]
    total: int


class SubjectGroupResponse(BaseModel):
    name: str
    groups: Dict[str, List[SubjectResponse]]


# Document type schemas
class DocumentTypeFieldCreate(BaseModel):
    key: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z_][a-z0-9_]*$")
    label: str = Field(..., min_length=1, max_length=255)
    field_type: FieldTypeEnum
    required: bool = False
    enum_values: Optional[List[str]] = None
    regex: Optional[str] = None
    description: Optional[str] = None
    examples: Optional[List[str]] = None


class DocumentTypeFieldUpdate(BaseModel):
    key: Optional[str] = Field(None, min_length=1, max_length=100, pattern=r"^[a-z_][a-z0-9_]*$")
    label: Optional[str] = Field(None, min_length=1, max_length=255)
    field_type: Optional[FieldTypeEnum] = None
    required: Optional[bool] = None
    enum_values: Optional[List[str]] = None
    regex: Optional[str] = None
    description: Optional[str] = None
    examples: Optional[List[str]] = None


class DocumentTypeFieldResponse(BaseModel):
    id: int
    document_type_id: int
    key: str
    label: str
    field_type: FieldTypeEnum
    required: bool
    enum_values: Optional[List[str]]
    regex: Optional[str]
    description: Optional[str]
    examples: Optional[List[str]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentTypeCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9\-]*$")
    description: Optional[str] = None
    classification_hints: Optional[str] = None
    extraction_prompt_preamble: Optional[str] = None
    classification_policy_json: Optional[Dict[str, Any]] = None


class DocumentTypeUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    slug: Optional[str] = Field(None, min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9\-]*$")
    description: Optional[str] = None
    classification_hints: Optional[str] = None
    extraction_prompt_preamble: Optional[str] = None
    classification_policy_json: Optional[Dict[str, Any]] = None


class DocumentTypeResponse(BaseModel):
    id: int
    name: str
    slug: str
    description: Optional[str]
    classification_hints: Optional[str]
    extraction_prompt_preamble: Optional[str]
    classification_policy_json: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    fields: List[DocumentTypeFieldResponse] = []

    class Config:
        from_attributes = True

    @field_validator('classification_policy_json', mode='before')
    @classmethod
    def parse_policy_json(cls, v):
        """Parse JSON string to dict if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return v


# Document schemas
class DocumentUploadResponse(BaseModel):
    document_id: int


class DocumentResponse(BaseModel):
    id: int
    subject_id: int
    subject_name: Optional[str] = None
    subject_context: Optional[str] = None
    original_filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    status: DocumentStatusEnum
    progress: int
    stage: Optional[str]
    error_message: Optional[str]
    doc_type_slug: Optional[str]
    doc_type_confidence: Optional[float]
    doc_type_rationale: Optional[str]
    metadata_json: Optional[Dict[str, Any]]
    metadata_validation_json: Optional[Dict[str, Any]]
    metadata_evidence_json: Optional[Dict[str, Any]]
    risk_score: Optional[int]
    risk_signals_json: Optional[List[Dict[str, Any]]]
    ocr_used: bool
    ocr_quality: Optional[str]
    skip_marker_used: Optional[str] = None
    skip_marker_position: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


# Health check
class OllamaHealth(BaseModel):
    """Legacy Ollama health schema (kept for backward compatibility)."""
    reachable: bool
    base_url: str
    model: str


class LLMHealth(BaseModel):
    """LLM provider health status."""
    provider: str  # "ollama" or "vllm"
    reachable: bool
    base_url: str
    model: str


class HealthResponse(BaseModel):
    ok: bool
    ollama: OllamaHealth  # Kept for backward compatibility
    llm: Optional[LLMHealth] = None  # New field with provider info


# Evidence schema
class EvidenceSpan(BaseModel):
    page: int = Field(..., ge=0)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    quote: str = ""  # Made optional with empty default - will be filled from text if missing


class ExtractionEvidence(BaseModel):
    data: Dict[str, Any]
    evidence: Dict[str, List[EvidenceSpan]]


# Risk signals
class RiskSignal(BaseModel):
    code: str
    severity: Literal["low", "medium", "high"]
    message: str
    evidence: str
    examples: Optional[Dict[str, Any]] = None


class RiskAnalysis(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    signals: List[RiskSignal]


# OCR results
class OCRResult(BaseModel):
    pages: List[Dict[str, Any]]
    combined_text: str
    ocr_used: bool
    ocr_quality: Literal["high", "medium", "low"]


# Classification result
class ClassificationResult(BaseModel):
    doc_type_slug: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    evidence: Optional[str] = None  # Exact quote from document text proving the classification


# Test extraction
class TestExtractionRequest(BaseModel):
    document_id: int
    document_type_slug: str