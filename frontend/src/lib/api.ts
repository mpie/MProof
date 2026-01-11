import axios from 'axios';

// API Base URL
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
const API_BASE_URL = `${BACKEND_URL}/api`;

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Suppress 404 errors for artifact requests
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 404 && error.config?.url?.includes('/artifact')) {
      return Promise.reject({ ...error, silent: true });
    }
    return Promise.reject(error);
  }
);

// =============================================================================
// Core Types
// =============================================================================

export interface Subject {
  id: number;
  name: string;
  context: 'person' | 'company' | 'dossier' | 'other';
  created_at: string;
  updated_at: string;
}

export interface DocumentType {
  id: number;
  name: string;
  slug: string;
  description?: string;
  classification_hints?: string;
  extraction_prompt_preamble?: string;
  classification_policy_json?: ClassificationPolicy;
  created_at: string;
  updated_at: string;
  fields: DocumentTypeField[];
}

export interface DocumentTypeField {
  id: number;
  document_type_id: number;
  key: string;
  label: string;
  field_type: 'string' | 'date' | 'number' | 'money' | 'currency' | 'iban' | 'enum';
  required: boolean;
  enum_values?: string[];
  regex?: string;
  description?: string;
  examples?: string[];
  created_at: string;
  updated_at: string;
}

export interface Document {
  id: number;
  subject_id: number;
  subject_name?: string;
  subject_context?: string;
  original_filename: string;
  mime_type: string;
  size_bytes: number;
  sha256: string;
  status: 'pending' | 'queued' | 'processing' | 'done' | 'error';
  progress: number;
  stage?: string;
  error_message?: string;
  doc_type_slug?: string;
  doc_type_confidence?: number;
  doc_type_rationale?: string;
  metadata_json?: Record<string, unknown>;
  metadata_validation_json?: Record<string, unknown>;
  metadata_evidence_json?: Record<string, unknown>;
  risk_score?: number;
  risk_signals_json?: Array<{
    code: string;
    severity: 'low' | 'medium' | 'high';
    message: string;
    evidence: string;
  }>;
  ocr_used: boolean;
  ocr_quality?: string;
  skip_marker_used?: string;
  skip_marker_position?: number;
  created_at: string;
  updated_at: string;
}

// =============================================================================
// Classification Policy Types (Single Canonical Schema)
// =============================================================================

export interface SignalRequirement {
  signal: string;
  op: '==' | '!=' | '>=' | '<=' | '>' | '<';
  value: boolean | number;
}

export interface TrainedModelAcceptance {
  enabled?: boolean;
  min_confidence?: number;
  min_margin?: number;
}

export interface DeterministicAcceptance {
  enabled?: boolean;
}

export interface LLMAcceptance {
  enabled?: boolean;
  require_evidence?: boolean;
}

export interface AcceptanceConfig {
  trained_model?: TrainedModelAcceptance;
  deterministic?: DeterministicAcceptance;
  llm?: LLMAcceptance;
}

export interface ClassificationPolicy {
  requirements?: SignalRequirement[];
  exclusions?: SignalRequirement[];
  acceptance?: AcceptanceConfig;
}

// =============================================================================
// Signal Types
// =============================================================================

export interface Signal {
  id: number;
  key: string;
  label: string;
  description?: string;
  signal_type: 'boolean' | 'count';
  source: 'builtin' | 'user';
  compute_kind: 'builtin' | 'keyword_set' | 'regex_set';
  config_json?: {
    keywords?: string[];
    patterns?: string[];
    match_mode?: 'any' | 'all';
  };
  created_at: string;
  updated_at: string;
}

export interface SignalCreate {
  key: string;
  label: string;
  description?: string;
  signal_type: 'boolean' | 'count';
  compute_kind: 'keyword_set' | 'regex_set';
  config_json: {
    keywords?: string[];
    patterns?: string[];
    match_mode?: 'any' | 'all';
  };
}

export interface SignalUpdate {
  label?: string;
  description?: string;
  config_json?: {
    keywords?: string[];
    patterns?: string[];
    match_mode?: 'any' | 'all';
  };
}

export interface SignalListResponse {
  signals: Signal[];
  builtin_count: number;
  user_count: number;
}

export interface SignalTestResponse {
  signals: Record<string, boolean | number>;
  text_length: number;
  line_count: number;
}

// =============================================================================
// Policy Types
// =============================================================================

export interface TypePolicyResponse {
  slug: string;
  policy: ClassificationPolicy | null;
  has_policy: boolean;
}

export interface SignalValue {
  key: string;
  label: string;
  value: boolean | number;
  signal_type: string;
}

export interface RequirementResult {
  signal: string;
  op: string;
  required_value: boolean | number;
  actual_value: boolean | number;
  passed: boolean;
}

export interface EligibilityPreviewResponse {
  is_eligible: boolean;
  computed_signals: SignalValue[];
  requirement_results: RequirementResult[];
  exclusion_results: RequirementResult[];
  failed_requirements: string[];
  triggered_exclusions: string[];
}

// =============================================================================
// Other Response Types
// =============================================================================

export interface SubjectSearchResponse {
  subjects: Subject[];
  total: number;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface DocumentUploadResponse {
  document_id: number;
}

export interface DocumentEvent {
  type: 'status' | 'result' | 'error';
  status?: string;
  stage?: string;
  progress?: number;
  updated_at?: string;
  doc_type_slug?: string;
  confidence?: number;
  metadata?: Record<string, unknown>;
  risk_score?: number;
  error_message?: string;
}

export interface QueueStatus {
  queue_size: number;
  is_running: boolean;
}

export interface ClassifierStatus {
  running: boolean;
  started_at?: string | null;
  finished_at?: string | null;
  last_error?: string | null;
  last_summary?: Record<string, unknown> | null;
  model_path?: string;
  dataset_dir?: string;
}

export interface AvailableModel {
  name: string;
  path: string;
  document_types: Array<{ slug: string; file_count: number }>;
  total_files: number;
  is_trained: boolean;
}

export interface AvailableModelsResponse {
  models: AvailableModel[];
  active_model: string;
  data_dir: string;
}

export interface TrainingDetails {
  model_exists: boolean;
  index_exists: boolean;
  model?: {
    version: number;
    created_at: string;
    updated_at: string;
    threshold: number;
    alpha: number;
    vocab_size: number;
    labels: string[];
    class_doc_counts: Record<string, number>;
    class_total_tokens: Record<string, number>;
  };
  index?: {
    version: number;
    dataset_dir: string;
    updated_at: string;
    total_files: number;
  };
  training_files_by_label: Record<string, Array<{ path: string; sha256: string; updated_at: string }>>;
  important_tokens_by_label: Record<string, Array<{ token: string; count: number }>>;
}

export interface ApiKey {
  id: number;
  name: string;
  client_id: string;
  scopes: string[] | null;
  is_active: boolean;
  last_used_at: string | null;
  expires_at: string | null;
  created_at: string;
}

export interface ApiKeyCreated extends ApiKey {
  client_secret: string;
}

export interface SkipMarker {
  id: number;
  pattern: string;
  description?: string;
  is_regex: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface BertClassifierStatus {
  running: boolean;
  model_exists: boolean;
  model_downloaded?: boolean;
  started_at: string | null;
  finished_at: string | null;
  last_error: string | null;
  last_summary: {
    success: boolean;
    labels: string[];
    samples_per_label: Record<string, number>;
    total_documents: number;
    embedding_dim: number;
    threshold: number;
  } | null;
  model_name: string | null;
  bert_model: string;
  labels: string[];
  threshold: number;
}

// =============================================================================
// Fraud Analysis Types
// =============================================================================

export interface RiskSignal {
  code: string;
  severity: 'low' | 'medium' | 'high';
  message: string;
  evidence: string;
}

export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

export interface FraudSignal {
  name: string;
  description: string;
  risk_level: RiskLevel;
  confidence: number;
  details: Record<string, unknown>;
  evidence: string[];
}

export interface FraudReport {
  document_id: number | null;
  filename: string;
  overall_risk: RiskLevel;
  risk_score: number;
  signals: FraudSignal[];
  summary: string;
  analyzed_at: string;
}

// =============================================================================
// API Functions - Subjects
// =============================================================================

export const searchSubjects = async (
  query?: string,
  context?: 'person' | 'company' | 'dossier' | 'other',
  limit = 50
): Promise<SubjectSearchResponse> => {
  const params = new URLSearchParams();
  if (query) params.append('query', query);
  if (context) params.append('context', context);
  params.append('limit', limit.toString());
  const response = await api.get(`/subjects?${params}`);
  return response.data;
};

export const listSubjects = async (limit = 100): Promise<Subject[]> => {
  const response = await searchSubjects(undefined, undefined, limit);
  return response.subjects;
};

export const createSubject = async (data: { name: string; context: 'person' | 'company' | 'dossier' | 'other' }): Promise<Subject> => {
  const response = await api.post('/subjects', data);
  return response.data;
};

// =============================================================================
// API Functions - Documents
// =============================================================================

export const listDocuments = async (subjectId?: number, status?: string, limit = 50, offset = 0): Promise<DocumentListResponse> => {
  const params = new URLSearchParams();
  if (subjectId) params.append('subject_id', subjectId.toString());
  if (status) params.append('status', status);
  params.append('limit', limit.toString());
  params.append('offset', offset.toString());
  const response = await api.get(`/documents?${params}`);
  return response.data;
};

export const getDocument = async (documentId: number): Promise<Document> => {
  const response = await api.get(`/documents/${documentId}`);
  return response.data;
};

export const analyzeDocument = async (documentId: number): Promise<{ ok: boolean }> => {
  const response = await api.post(`/documents/${documentId}/analyze`);
  return response.data;
};

export const uploadDocument = async (subjectId: number, file: File, modelName?: string): Promise<DocumentUploadResponse> => {
  const formData = new FormData();
  formData.append('subject_id', subjectId.toString());
  formData.append('file', file);
  if (modelName) formData.append('model_name', modelName);
  const response = await api.post('/upload', formData);
  return response.data;
};

export const getDocumentArtifact = async (documentId: number, path: string): Promise<Blob> => {
  const response = await api.get(`/documents/${documentId}/artifact`, { params: { path }, responseType: 'blob' });
  return response.data;
};

export const getDocumentArtifactText = async (documentId: number, path: string): Promise<string> => {
  try {
    const response = await api.get(`/documents/${documentId}/artifact`, { params: { path }, responseType: 'text' });
    return response.data;
  } catch (error: unknown) {
    if ((error as { response?: { status?: number }; silent?: boolean }).response?.status === 404 || (error as { silent?: boolean }).silent) {
      throw new Error('Artifact not found');
    }
    throw error;
  }
};

export const getDocumentArtifactJson = async <T = unknown>(documentId: number, path: string): Promise<T> => {
  try {
    const response = await api.get(`/documents/${documentId}/artifact`, { params: { path } });
    return response.data as T;
  } catch (error: unknown) {
    if ((error as { response?: { status?: number }; silent?: boolean }).response?.status === 404 || (error as { silent?: boolean }).silent) {
      throw new Error('Artifact not found');
    }
    throw error;
  }
};

export const deleteDocument = async (documentId: number): Promise<void> => {
  await api.delete(`/documents/${documentId}`);
};

// =============================================================================
// API Functions - Document Types
// =============================================================================

export const listDocumentTypes = async (): Promise<DocumentType[]> => {
  const response = await api.get('/document-types');
  return response.data;
};

export const createDocumentType = async (data: { name: string; slug: string; description?: string; classification_hints?: string; extraction_prompt_preamble?: string }): Promise<DocumentType> => {
  const response = await api.post('/document-types', data);
  return response.data;
};

export const getDocumentType = async (slug: string): Promise<DocumentType> => {
  const response = await api.get(`/document-types/${slug}`);
  return response.data;
};

export const updateDocumentType = async (slug: string, data: Partial<{ name: string; slug: string; description: string; classification_hints: string; extraction_prompt_preamble: string }>): Promise<DocumentType> => {
  const response = await api.put(`/document-types/${slug}`, data);
  return response.data;
};

export const deleteDocumentType = async (slug: string): Promise<void> => {
  await api.delete(`/document-types/${slug}`);
};

export const checkDocumentTypeNameUnique = async (name: string, excludeSlug?: string): Promise<boolean> => {
  try {
    const params = new URLSearchParams({ name });
    if (excludeSlug) params.append('exclude_slug', excludeSlug);
    const response = await api.get(`/document-types/check-name?${params}`);
    return response.data.is_unique;
  } catch {
    return false;
  }
};

export const checkDocumentTypeSlugUnique = async (slug: string, excludeSlug?: string): Promise<boolean> => {
  try {
    const params = new URLSearchParams({ slug });
    if (excludeSlug) params.append('exclude_slug', excludeSlug);
    const response = await api.get(`/document-types/check-slug?${params}`);
    return response.data.is_unique;
  } catch {
    return false;
  }
};

// Document Type Fields
export const listDocumentTypeFields = async (slug: string): Promise<DocumentTypeField[]> => {
  const response = await api.get(`/document-types/${slug}/fields`);
  return response.data;
};

export const createDocumentTypeField = async (slug: string, data: { key: string; label: string; field_type: string; required: boolean; enum_values?: string[]; regex?: string; description?: string; examples?: string[] }): Promise<DocumentTypeField> => {
  const response = await api.post(`/document-types/${slug}/fields`, data);
  return response.data;
};

export const updateDocumentTypeField = async (slug: string, fieldId: number, data: Partial<{ key: string; label: string; field_type: string; required: boolean; enum_values: string[]; regex: string; description: string; examples: string[] }>): Promise<DocumentTypeField> => {
  const response = await api.put(`/document-types/${slug}/fields/${fieldId}`, data);
  return response.data;
};

export const deleteDocumentTypeField = async (slug: string, fieldId: number): Promise<void> => {
  await api.delete(`/document-types/${slug}/fields/${fieldId}`);
};

// =============================================================================
// API Functions - Signals
// =============================================================================

export const listSignals = async (): Promise<SignalListResponse> => {
  const response = await api.get('/signals');
  return response.data;
};

export const getSignal = async (key: string): Promise<Signal> => {
  const response = await api.get(`/signals/${key}`);
  return response.data;
};

export const createSignal = async (data: SignalCreate): Promise<Signal> => {
  const response = await api.post('/signals', data);
  return response.data;
};

export const updateSignal = async (key: string, data: SignalUpdate): Promise<Signal> => {
  const response = await api.put(`/signals/${key}`, data);
  return response.data;
};

export const deleteSignal = async (key: string): Promise<void> => {
  await api.delete(`/signals/${key}`);
};

export const testSignals = async (text: string): Promise<SignalTestResponse> => {
  const response = await api.post('/signals/test', { text });
  return response.data;
};

// =============================================================================
// API Functions - Classification Policies
// =============================================================================

export const getDocumentTypePolicy = async (slug: string): Promise<TypePolicyResponse> => {
  const response = await api.get(`/document-types/${slug}/policy`);
  return response.data;
};

export const updateDocumentTypePolicy = async (slug: string, policy: ClassificationPolicy): Promise<TypePolicyResponse> => {
  const response = await api.put(`/document-types/${slug}/policy`, policy);
  return response.data;
};

export const deleteDocumentTypePolicy = async (slug: string): Promise<void> => {
  await api.delete(`/document-types/${slug}/policy`);
};

export const previewEligibility = async (slug: string, text: string, policy?: ClassificationPolicy): Promise<EligibilityPreviewResponse> => {
  const response = await api.post(`/document-types/${slug}/policy/preview`, { text, policy });
  return response.data;
};

// =============================================================================
// API Functions - SSE & Queue
// =============================================================================

export const subscribeToDocumentEvents = (documentId: number, onEvent: (event: DocumentEvent) => void, onError?: (error: Event) => void): EventSource => {
  const eventSource = new EventSource(`${API_BASE_URL}/documents/${documentId}/events`);
  const handleMessage = (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      onEvent(data);
    } catch {
      console.error('Failed to parse SSE event');
    }
  };
  eventSource.addEventListener('document-update', (event) => handleMessage(event as MessageEvent));
  eventSource.onmessage = handleMessage;
  eventSource.onerror = (error) => {
    console.error('SSE connection error:', error);
    if (onError) onError(error);
  };
  return eventSource;
};

export const getQueueStatus = async (): Promise<QueueStatus> => {
  const response = await api.get('/queue/status');
  return response.data;
};

// =============================================================================
// API Functions - Classifier
// =============================================================================

export const getClassifierStatus = async (): Promise<ClassifierStatus> => {
  const response = await api.get('/classifier/status');
  return response.data;
};

export const trainClassifier = async (modelName?: string): Promise<ClassifierStatus> => {
  const params = modelName ? { model_name: modelName } : undefined;
  const response = await api.post('/classifier/train', undefined, { timeout: 0, params });
  return response.data;
};

export const getAvailableModels = async (): Promise<AvailableModelsResponse> => {
  const response = await api.get('/classifier/models');
  return response.data;
};

export const getTrainingDetails = async (modelName?: string): Promise<TrainingDetails> => {
  const params = modelName ? { model_name: modelName } : {};
  const response = await api.get('/classifier/training-details', { params });
  return response.data;
};

export const getBertClassifierStatus = async (modelName?: string): Promise<BertClassifierStatus> => {
  const params = modelName ? { model_name: modelName } : undefined;
  const response = await api.get('/classifier/bert/status', { params });
  return response.data;
};

export const trainBertClassifier = async (modelName?: string, threshold = 0.7): Promise<BertClassifierStatus> => {
  const params: Record<string, string | number> = { threshold };
  if (modelName) params.model_name = modelName;
  const response = await api.post('/classifier/bert/train', undefined, { timeout: 0, params });
  return response.data;
};

// =============================================================================
// API Functions - API Keys
// =============================================================================

export const listApiKeys = async (): Promise<ApiKey[]> => {
  const response = await api.get('/api-keys');
  return response.data;
};

export const createApiKey = async (data: { name: string; scopes?: string[]; expires_days?: number }): Promise<ApiKeyCreated> => {
  const response = await api.post('/api-keys', data);
  return response.data;
};

export const updateApiKey = async (keyId: number, data: { name?: string; scopes?: string[]; is_active?: boolean }): Promise<ApiKey> => {
  const response = await api.put(`/api-keys/${keyId}`, data);
  return response.data;
};

export const deleteApiKey = async (keyId: number): Promise<void> => {
  await api.delete(`/api-keys/${keyId}`);
};

export const regenerateApiKey = async (keyId: number): Promise<ApiKeyCreated> => {
  const response = await api.post(`/api-keys/${keyId}/regenerate`);
  return response.data;
};

// =============================================================================
// API Functions - Skip Markers
// =============================================================================

export const listSkipMarkers = async (activeOnly = false): Promise<SkipMarker[]> => {
  const response = await api.get('/skip-markers', { params: { active_only: activeOnly } });
  return response.data;
};

export const createSkipMarker = async (data: { pattern: string; description?: string; is_regex?: boolean; is_active?: boolean }): Promise<SkipMarker> => {
  const response = await api.post('/skip-markers', data);
  return response.data;
};

export const updateSkipMarker = async (markerId: number, data: { pattern?: string; description?: string; is_regex?: boolean; is_active?: boolean }): Promise<SkipMarker> => {
  const response = await api.put(`/skip-markers/${markerId}`, data);
  return response.data;
};

export const deleteSkipMarker = async (markerId: number): Promise<void> => {
  await api.delete(`/skip-markers/${markerId}`);
};

// =============================================================================
// API Functions - Fraud Analysis
// =============================================================================

export const getFraudAnalysis = async (documentId: number): Promise<FraudReport> => {
  const response = await api.get(`/documents/${documentId}/fraud-analysis`);
  return response.data;
};
