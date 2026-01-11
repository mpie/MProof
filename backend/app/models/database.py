from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import datetime
from typing import List, Optional
import enum


class Base(AsyncAttrs, DeclarativeBase):
    pass


class ContextEnum(str, enum.Enum):
    person = "person"
    company = "company"
    dossier = "dossier"
    other = "other"


class FieldTypeEnum(str, enum.Enum):
    string = "string"
    date = "date"
    number = "number"
    money = "money"
    currency = "currency"
    iban = "iban"
    enum = "enum"


class DocumentStatusEnum(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name_normalized: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    context: Mapped[ContextEnum] = mapped_column(Enum(ContextEnum), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    documents: Mapped[List["Document"]] = relationship("Document", back_populates="subject")


class DocumentType(Base):
    __tablename__ = "document_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    classification_hints: Mapped[Optional[str]] = mapped_column(Text)
    extraction_prompt_preamble: Mapped[Optional[str]] = mapped_column(Text)
    classification_policy_json: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    fields: Mapped[List["DocumentTypeField"]] = relationship("DocumentTypeField", back_populates="document_type")


class DocumentTypeField(Base):
    __tablename__ = "document_type_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_type_id: Mapped[int] = mapped_column(Integer, ForeignKey("document_types.id"), nullable=False)
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    field_type: Mapped[FieldTypeEnum] = mapped_column(Enum(FieldTypeEnum), nullable=False)
    required: Mapped[bool] = mapped_column(Boolean, default=False)
    enum_values: Mapped[Optional[list]] = mapped_column(JSON)
    regex: Mapped[Optional[str]] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text)
    examples: Mapped[Optional[list]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document_type: Mapped["DocumentType"] = relationship("DocumentType", back_populates="fields")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[int] = mapped_column(Integer, ForeignKey("subjects.id"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[DocumentStatusEnum] = mapped_column(Enum(DocumentStatusEnum), default=DocumentStatusEnum.queued)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[Optional[str]] = mapped_column(String(100))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    doc_type_slug: Mapped[Optional[str]] = mapped_column(String(100))
    doc_type_confidence: Mapped[Optional[float]] = mapped_column(Float)
    doc_type_rationale: Mapped[Optional[str]] = mapped_column(Text)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON)
    metadata_validation_json: Mapped[Optional[dict]] = mapped_column(JSON)
    metadata_evidence_json: Mapped[Optional[dict]] = mapped_column(JSON)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer)
    risk_signals_json: Mapped[Optional[list]] = mapped_column(JSON)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    ocr_quality: Mapped[Optional[str]] = mapped_column(String(20))
    skip_marker_used: Mapped[Optional[str]] = mapped_column(String(500))  # Pattern that matched
    skip_marker_position: Mapped[Optional[int]] = mapped_column(Integer)  # Position where text was truncated
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    subject: Mapped["Subject"] = relationship("Subject", back_populates="documents")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    client_id: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    client_secret_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA256 hash
    scopes: Mapped[Optional[list]] = mapped_column(JSON)  # ["read", "write", "admin"]
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SkipMarker(Base):
    """Text patterns that signal to stop processing (e.g., 'Algemene Voorwaarden')."""
    __tablename__ = "skip_markers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_regex: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SignalTypeEnum(str, enum.Enum):
    boolean = "boolean"
    count = "count"


class SignalSourceEnum(str, enum.Enum):
    builtin = "builtin"
    user = "user"


class ComputeKindEnum(str, enum.Enum):
    builtin = "builtin"
    keyword_set = "keyword_set"
    regex_set = "regex_set"


class ClassificationSignal(Base):
    """Classification signals for document eligibility evaluation."""
    __tablename__ = "classification_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'boolean' or 'count'
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # 'builtin' or 'user'
    compute_kind: Mapped[str] = mapped_column(String(20), nullable=False)  # 'builtin', 'keyword_set', 'regex_set'
    config_json: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AppSetting(Base):
    """Application settings stored in database (key-value store)."""
    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)