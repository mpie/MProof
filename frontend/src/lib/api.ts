import axios, { AxiosResponse } from 'axios';

// API Base URL
// Backend base URL (without /api suffix)
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
const API_BASE_URL = `${BACKEND_URL}/api`;

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Suppress 404 errors for artifact requests (files may not exist)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Suppress 404 errors for artifact endpoints
    if (error.response?.status === 404 && error.config?.url?.includes('/artifact')) {
      // Return a rejected promise that won't log to console
      return Promise.reject({ ...error, silent: true });
    }
    return Promise.reject(error);
  }
);

// Types
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
  metadata_json?: Record<string, any>;
  metadata_validation_json?: Record<string, any>;
  metadata_evidence_json?: Record<string, any>;
  risk_score?: number;
  risk_signals_json?: Array<{
    code: string;
    severity: 'low' | 'medium' | 'high';
    message: string;
    evidence: string;
  }>;
  ocr_used: boolean;
  ocr_quality?: string;
  created_at: string;
  updated_at: string;
}

export interface HealthResponse {
  ok: boolean;
  ollama: {
    reachable: boolean;
    base_url: string;
    model: string;
  };
}

export interface SubjectSearchResponse {
  subjects: Subject[];
  total: number;
}

export interface SubjectGroupResponse {
  name: string;
  groups: Record<string, Subject[]>;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
}

export interface DocumentUploadResponse {
  document_id: number;
}

export interface RiskSignal {
  code: string;
  severity: 'low' | 'medium' | 'high';
  message: string;
  evidence: string;
}

// API Functions

// Health
export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await api.get('/health');
  return response.data;
};

// Subjects
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

export const createSubject = async (data: {
  name: string;
  context: 'person' | 'company' | 'dossier' | 'other';
}): Promise<Subject> => {
  const response = await api.post('/subjects', data);
  return response.data;
};

export const getSubject = async (subjectId: number): Promise<Subject> => {
  const response = await api.get(`/subjects/${subjectId}`);
  return response.data;
};

export const getSubjectDocuments = async (
  subjectId: number,
  limit = 100,
  offset = 0
): Promise<DocumentListResponse> => {
  const params = new URLSearchParams();
  params.append('limit', limit.toString());
  params.append('offset', offset.toString());

  const response = await api.get(`/subjects/${subjectId}/documents?${params}`);
  return response.data;
};

export const searchSubjectsByName = async (name: string): Promise<SubjectGroupResponse> => {
  const response = await api.get(`/subjects/by-name?name=${encodeURIComponent(name)}`);
  return response.data;
};

// Documents
export const listDocuments = async (
  subjectId?: number,
  status?: string,
  limit = 50,
  offset = 0
): Promise<DocumentListResponse> => {
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

export const uploadDocument = async (
  subjectId: number,
  file: File
): Promise<DocumentUploadResponse> => {
  const formData = new FormData();
  formData.append('subject_id', subjectId.toString());
  formData.append('file', file);

  const response = await api.post('/upload', formData);
  return response.data;
};

export const getDocumentArtifact = async (
  documentId: number,
  path: string
): Promise<Blob> => {
  const response = await api.get(`/documents/${documentId}/artifact`, {
    params: { path },
    responseType: 'blob',
  });
  return response.data;
};

export const getDocumentArtifactText = async (
  documentId: number,
  path: string
): Promise<string> => {
  try {
    const response = await api.get(`/documents/${documentId}/artifact`, {
      params: { path },
      responseType: 'text',
    });
    return response.data;
  } catch (error: any) {
    // Silently handle 404s (artifacts may not exist)
    if (error.response?.status === 404 || error.silent) {
      throw new Error('Artifact not found');
    }
    throw error;
  }
};

export const getDocumentArtifactJson = async <T = any>(
  documentId: number,
  path: string
): Promise<T> => {
  try {
    const response = await api.get(`/documents/${documentId}/artifact`, {
      params: { path },
    });
    return response.data as T;
  } catch (error: any) {
    // Silently handle 404s (artifacts may not exist)
    if (error.response?.status === 404 || error.silent) {
      throw new Error('Artifact not found');
    }
    throw error;
  }
};

export const deleteDocument = async (documentId: number): Promise<void> => {
  await api.delete(`/documents/${documentId}`);
};

// Document Types
export const listDocumentTypes = async (): Promise<DocumentType[]> => {
  const response = await api.get('/document-types');
  return response.data;
};

export const createDocumentType = async (data: {
  name: string;
  slug: string;
  description?: string;
  classification_hints?: string;
  extraction_prompt_preamble?: string;
}): Promise<DocumentType> => {
  const response = await api.post('/document-types', data);
  return response.data;
};

export const getDocumentType = async (slug: string): Promise<DocumentType> => {
  const response = await api.get(`/document-types/${slug}`);
  return response.data;
};

export const updateDocumentType = async (
  slug: string,
  data: Partial<{
    name: string;
    slug: string;
    description: string;
    classification_hints: string;
    extraction_prompt_preamble: string;
  }>
): Promise<DocumentType> => {
  const response = await api.put(`/document-types/${slug}`, data);
  return response.data;
};

export const deleteDocumentType = async (slug: string): Promise<void> => {
  await api.delete(`/document-types/${slug}`);
};

export const checkDocumentTypeNameUnique = async (name: string, excludeSlug?: string): Promise<boolean> => {
  try {
    const params = new URLSearchParams({ name });
    if (excludeSlug) {
      params.append('exclude_slug', excludeSlug);
    }
    const response = await api.get(`/document-types/check-name?${params}`);
    return response.data.is_unique;
  } catch (error) {
    // If there's an error, assume it's not unique for safety
    return false;
  }
};

export const checkDocumentTypeSlugUnique = async (slug: string, excludeSlug?: string): Promise<boolean> => {
  try {
    const params = new URLSearchParams({ slug });
    if (excludeSlug) {
      params.append('exclude_slug', excludeSlug);
    }
    const response = await api.get(`/document-types/check-slug?${params}`);
    return response.data.is_unique;
  } catch (error) {
    // If there's an error, assume it's not unique for safety
    return false;
  }
};

// Document Type Fields
export const listDocumentTypeFields = async (slug: string): Promise<DocumentTypeField[]> => {
  const response = await api.get(`/document-types/${slug}/fields`);
  return response.data;
};

export const createDocumentTypeField = async (
  slug: string,
  data: {
    key: string;
    label: string;
    field_type: 'string' | 'date' | 'number' | 'money' | 'currency' | 'iban' | 'enum';
    required: boolean;
    enum_values?: string[];
    regex?: string;
    description?: string;
    examples?: string[];
  }
): Promise<DocumentTypeField> => {
  const response = await api.post(`/document-types/${slug}/fields`, data);
  return response.data;
};

export const updateDocumentTypeField = async (
  slug: string,
  fieldId: number,
  data: Partial<{
    key: string;
    label: string;
    field_type: 'string' | 'date' | 'number' | 'money' | 'currency' | 'iban' | 'enum';
    required: boolean;
    enum_values: string[];
    regex: string;
    description: string;
    examples: string[];
  }>
): Promise<DocumentTypeField> => {
  const response = await api.put(`/document-types/${slug}/fields/${fieldId}`, data);
  return response.data;
};

export const deleteDocumentTypeField = async (
  slug: string,
  fieldId: number
): Promise<void> => {
  await api.delete(`/document-types/${slug}/fields/${fieldId}`);
};

// SSE Events
export const subscribeToDocumentEvents = (
  documentId: number,
  onEvent: (event: DocumentEvent) => void,
  onError?: (error: Event) => void
): EventSource => {
  const eventSource = new EventSource(`${API_BASE_URL}/documents/${documentId}/events`);

  const handleMessage = (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      onEvent(data);
    } catch (error) {
      console.error('Failed to parse SSE event:', error);
    }
  };

  // Backend emits named events: "document-update"
  eventSource.addEventListener('document-update', (event) => {
    handleMessage(event as MessageEvent);
  });

  // Fallback for default "message" events
  eventSource.onmessage = handleMessage;

  eventSource.onerror = (error) => {
    console.error('SSE connection error:', error);
    if (onError) {
      onError(error);
    }
  };

  return eventSource;
};

export interface DocumentEvent {
  type: 'status' | 'result' | 'error';
  status?: string;
  stage?: string;
  progress?: number;
  updated_at?: string;
  doc_type_slug?: string;
  confidence?: number;
  metadata?: Record<string, any>;
  risk_score?: number;
  error_message?: string;
}

export interface QueueStatus {
  queue_size: number;
  is_running: boolean;
}

export const getQueueStatus = async (): Promise<QueueStatus> => {
  const response = await api.get('/queue/status');
  return response.data;
};