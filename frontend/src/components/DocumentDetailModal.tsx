'use client';

import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faTimes, faEye, faDownload, faRedo, faFile, faCheck,
  faExclamationTriangle, faInfoCircle, faHighlighter,
  faSearch, faCopy, faChevronDown, faChevronUp, faSpinner, faRobot, faAngleDown, faAngleUp, faAngleRight, faImage, faFilePdf, faChevronLeft, faChevronRight, faShieldAlt,
  faCircleExclamation, faBolt, faCircle, faBullseye, faCog
} from '@fortawesome/free-solid-svg-icons';
import {
  Document, DocumentEvent, getDocument, analyzeDocument, getDocumentArtifact, getDocumentArtifactText, getDocumentArtifactJson,
  RiskSignal, subscribeToDocumentEvents, getFraudAnalysis, FraudReport, FraudSignal, AdviceCard,
  submitDocumentFeedback, listDocumentTypeFields, DocumentTypeField,
} from '@/lib/api';

// Dynamically import PDFViewerWithHighlights to avoid SSR issues with react-pdf
const PDFViewerWithHighlights = dynamic(
  () => import('./PDFViewerWithHighlights').then(mod => ({ default: mod.PDFViewerWithHighlights })),
  { 
    ssr: false,
    loading: () => <div className="flex items-center justify-center h-full text-slate-500">PDF viewer laden...</div>
  }
);

// Document Viewer Modal Component
function DocumentViewerModal({ 
  isOpen, 
  onClose, 
  documentUrl, 
  filename, 
  mimeType,
  onDownload,
  evidence
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  documentUrl: string | null; 
  filename: string;
  mimeType: string;
  onDownload: () => void;
  evidence?: Record<string, any[]>;
}) {
  if (!isOpen || !documentUrl) return null;

  const isPDF = mimeType === 'application/pdf';
  const isImage = mimeType.startsWith('image/');
  const isWord = mimeType.includes('word') || mimeType.includes('msword') || filename.toLowerCase().endsWith('.doc') || filename.toLowerCase().endsWith('.docx');
  const hasEvidence = evidence && Object.values(evidence).some(items => Array.isArray(items) && items.length > 0);

  const handleDownload = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDownload();
  };

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-[200] flex items-center justify-center p-0 sm:p-4" onClick={onClose}>
      <div className="glass-card w-full h-full sm:w-[85vw] sm:h-[85vh] max-w-[85vw] max-h-[85vh] flex flex-col rounded-none sm:rounded-xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-200 flex-shrink-0">
          <div className="flex items-center space-x-3 min-w-0">
            <FontAwesomeIcon 
              icon={isPDF ? faFilePdf : isImage ? faImage : faFile} 
              className="text-slate-500 w-5 h-5 flex-shrink-0" 
            />
            <div className="min-w-0">
              <h2 className="text-slate-800 text-lg font-semibold truncate">
                {filename}
              </h2>
              <p className="text-slate-500 text-xs">
                Origineel bestand
                {hasEvidence && <span className="text-blue-600 ml-2">• Met highlights</span>}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2 flex-shrink-0">
            <button
              onClick={handleDownload}
              className="flex items-center space-x-2 px-3 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 cursor-pointer"
            >
              <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
              <span>Download</span>
            </button>
            <button 
              onClick={onClose} 
              className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg cursor-pointer"
            >
              <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Viewer Content */}
        <div className="flex-1 overflow-hidden bg-white">
          {isPDF ? (
            // Use custom PDF viewer for PDFs; it enables highlights when evidence is available.
            <PDFViewerWithHighlights 
              url={documentUrl} 
              evidence={evidence || {}}
            />
          ) : isImage ? (
            <div className="w-full h-full p-2 sm:p-4 flex items-center justify-center">
              <img
                src={documentUrl}
                alt={filename}
                className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
              />
            </div>
          ) : isWord ? (
            <div className="text-center text-slate-500 w-full h-full flex flex-col items-center justify-center">
              <FontAwesomeIcon icon={faFile} className="w-16 h-16 mb-4 opacity-50" />
              <p className="mb-4">Word documenten kunnen niet direct in de browser worden weergegeven.</p>
              <button
                onClick={handleDownload}
                className="inline-flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <FontAwesomeIcon icon={faDownload} className="w-4 h-4" />
                <span>Download om te bekijken</span>
              </button>
            </div>
          ) : (
            <div className="text-center text-slate-500 w-full h-full flex flex-col items-center justify-center">
              <FontAwesomeIcon icon={faFile} className="w-16 h-16 mb-4 opacity-50" />
              <p>Preview niet beschikbaar voor dit bestandstype</p>
              <button
                onClick={handleDownload}
                className="mt-4 inline-flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <FontAwesomeIcon icon={faDownload} className="w-4 h-4" />
                <span>Download bestand</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function formatDocumentTypeName(slug: string): string {
  if (slug === 'unknown') return 'Onbekend';
  return slug.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatStage(stage: string | undefined): string {
  if (!stage) return '';
  const selectedMatch = stage.match(/^extracting_metadata_selecting_(\d+)_of_(\d+)_chunks$/);
  if (selectedMatch) {
    const [, selectedChunks, totalChunks] = selectedMatch;
    return `Metadata: ${selectedChunks}/${totalChunks} relevante chunks geselecteerd`;
  }

  const deterministicMatch = stage.match(/^extracting_metadata_deterministic_(\d+)_of_(\d+)_chunks$/);
  if (deterministicMatch) {
    const [, selectedChunks, totalChunks] = deterministicMatch;
    return `Metadata: deterministic extractie op ${selectedChunks}/${totalChunks} chunks`;
  }

  const llmSelectedMatch = stage.match(/^extracting_metadata_llm_selected_(\d+)_of_(\d+)_chunks$/);
  if (llmSelectedMatch) {
    const [, selectedChunks, totalChunks] = llmSelectedMatch;
    return `Metadata: LLM analyseert ${selectedChunks}/${totalChunks} geselecteerde chunks`;
  }

  const selectedDoneMatch = stage.match(/^extracting_metadata_selected_chunks_done_(\d+)_of_(\d+)$/);
  if (selectedDoneMatch) {
    const [, selectedChunks, totalChunks] = selectedDoneMatch;
    return `Metadata: ${selectedChunks}/${totalChunks} geselecteerde chunks verwerkt`;
  }

  // Replace all underscores with spaces and capitalize words
  return stage
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// Clean and format OCR text by normalizing excessive newlines and whitespace
function formatExampleText(text: string, isUnicode: boolean = false): string {
  if (!text) return '';
  
  if (isUnicode) {
    // For unicode examples, preserve the exact characters but clean up formatting
    return text
      .trim()
      .replace(/\n{3,}/g, '\n\n') // Max 2 consecutive newlines
      .replace(/[ \t]+/g, ' ') // Normalize spaces/tabs
      .replace(/\n[ \t]+/g, '\n') // Remove leading spaces after newlines
      .replace(/[ \t]+\n/g, '\n'); // Remove trailing spaces before newlines
  }
  
  // For repetition examples, normalize more aggressively
  return text
    .trim()
    .replace(/\n{3,}/g, '\n\n') // Max 2 consecutive newlines
    .replace(/[ \t]{2,}/g, ' ') // Multiple spaces/tabs to single space
    .replace(/\n[ \t]+/g, '\n') // Remove leading spaces after newlines
    .replace(/[ \t]+\n/g, '\n') // Remove trailing spaces before newlines
    .replace(/\n/g, ' ') // Convert single newlines to spaces for better readability
    .replace(/\s{2,}/g, ' ') // Multiple spaces to single space
    .trim();
}

type TabType = 'overview' | 'text' | 'metadata' | 'llm' | 'forensics';

interface DocumentDetailModalProps {
  documentId: number | null;
  isOpen: boolean;
  onClose: () => void;
  initialTab?: TabType;
}

// Build direct API URL for artifacts (bypasses blob URL issues)
const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export function DocumentDetailModal({ documentId, isOpen, onClose, initialTab }: DocumentDetailModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>(initialTab ?? 'overview');
  const [highlightSources, setHighlightSources] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedExamples, setExpandedExamples] = useState<Set<number>>(new Set());
  const [showDocumentViewer, setShowDocumentViewer] = useState(false);
  const [documentViewerPath, setDocumentViewerPath] = useState<string | null>(null);
  const [examplesSidebar, setExamplesSidebar] = useState<{ signalIndex: number; examples: any } | null>(null);
  const initializedTabDocumentRef = useRef<number | null>(null);
  const queryClient = useQueryClient();

  // Reset tab when document changes or modal opens with a new initialTab
  useEffect(() => {
    if (isOpen && documentId) setActiveTab(initialTab ?? 'overview');
  }, [documentId, isOpen, initialTab]);

  const { data: document, isLoading, refetch } = useQuery({
    queryKey: ['document', documentId],
    queryFn: () => getDocument(documentId!),
    enabled: !!documentId && isOpen,
    // Poll every 2 seconds when document is processing or queued
    refetchInterval: (query) => {
      const doc = query.state.data as Document | undefined;
      return doc && (doc.status === 'processing' || doc.status === 'queued') ? 2000 : false;
    },
  });

  const documentEvidence = document?.metadata_evidence_json as Record<string, any[]> | undefined;
  const hasDocumentEvidence = !!documentEvidence && Object.values(documentEvidence).some(items => Array.isArray(items) && items.length > 0);
  const { data: viewerEvidenceArtifact } = useQuery({
    queryKey: ['document-viewer-evidence-artifact', documentId],
    queryFn: () => getDocumentArtifactJson<Record<string, any[]>>(documentId!, 'metadata/evidence.json'),
    enabled: !!documentId && isOpen && document?.status === 'done' && !hasDocumentEvidence,
    retry: false,
    staleTime: Infinity,
  });
  const viewerEvidence = hasDocumentEvidence ? documentEvidence : viewerEvidenceArtifact;

  const { data: extractedText, isLoading: textLoading } = useQuery({
    queryKey: ['document-text', documentId],
    queryFn: () => getDocumentArtifactText(documentId!, 'text/extracted.txt'),
    enabled: !!documentId && isOpen && activeTab === 'text' && document?.status === 'done',
  });

  const analyzeMutation = useMutation({
    mutationFn: () => analyzeDocument(documentId!),
    onSuccess: () => {
      // Reset to overview tab and clear all cached data
      setActiveTab('overview');
      
      // Immediately update local state to show processing
      queryClient.setQueryData(['document', documentId], (old: Document | undefined) => {
        if (!old) return old;
        return {
          ...old,
          status: 'queued' as const,
          stage: 'Opnieuw verwerken...',
          progress: 0,
          doc_type_slug: null,
          doc_type_confidence: null,
          doc_type_rationale: null,
          metadata_json: null,
          metadata_validation_json: null,
          metadata_evidence_json: null,
          risk_score: null,
          risk_signals_json: null,
        };
      });
      
      // Remove cached queries immediately (don't wait for invalidation)
      queryClient.removeQueries({ queryKey: ['document-text', documentId] });
      queryClient.removeQueries({ queryKey: ['document-llm', documentId] });
      queryClient.removeQueries({ queryKey: ['ela-heatmap', documentId] });
      queryClient.removeQueries({ queryKey: ['fraud-analysis', documentId] });
      
      // Invalidate queries in background (don't await)
      queryClient.invalidateQueries({ queryKey: ['document', documentId] });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['documents-recent'] });
      
      // Refetch in background
      refetch();
    },
  });

  const [showRejectInput, setShowRejectInput] = useState(false);
  const [rejectCorrectedType, setRejectCorrectedType] = useState('');
  const [feedbackDone, setFeedbackDone] = useState<'confirmed' | 'rejected' | null>(null);

  const [feedbackError, setFeedbackError] = useState<string | null>(null);

  const feedbackMutation = useMutation({
    mutationFn: ({ action, correctedType }: { action: 'confirmed' | 'rejected'; correctedType?: string }) =>
      submitDocumentFeedback(documentId!, action, correctedType),
    onSuccess: (_, vars) => {
      setFeedbackDone(vars.action);
      setFeedbackError(null);
      setShowRejectInput(false);
      queryClient.invalidateQueries({ queryKey: ['document', documentId] });
    },
    onError: (err: any) => {
      const msg = err?.response?.data?.detail || err?.message || 'Feedback opslaan mislukt';
      setFeedbackError(msg);
    },
  });

  useEffect(() => {
    if (!documentId || !isOpen) return;
    const source = subscribeToDocumentEvents(
      documentId,
      (event: DocumentEvent) => {
        // Immediately update the document query cache with event data
        queryClient.setQueryData(['document', documentId], (old: Document | undefined) => {
          if (!old) return old;
          
          const updated = { ...old };
          
          if (event.type === 'status') {
            updated.status = event.status as Document['status'];
            updated.stage = event.stage || undefined;
            updated.progress = event.progress ?? old.progress;
            updated.updated_at = event.updated_at || old.updated_at;
          } else if (event.type === 'result') {
            if (event.doc_type_slug != null) updated.doc_type_slug = event.doc_type_slug;
            if (event.confidence != null) updated.doc_type_confidence = event.confidence;
            if (event.metadata != null) updated.metadata_json = event.metadata;
            if (event.risk_score != null) updated.risk_score = event.risk_score;
            updated.status = 'done';
            updated.progress = 100;
            updated.updated_at = new Date().toISOString();
          } else if (event.type === 'error') {
            updated.status = 'error';
            updated.error_message = event.error_message || 'Unknown error';
            updated.updated_at = new Date().toISOString();
          }
          
          return updated;
        });
        
        // Also refetch to ensure consistency
        refetch();
        queryClient.invalidateQueries({ queryKey: ['documents'] });
        
        // Invalidate LLM artifacts when document status changes or finishes processing
        if (event.type === 'result' || (event.type === 'status' && event.stage?.includes('extract'))) {
          queryClient.removeQueries({ queryKey: ['document-llm', documentId] });
          queryClient.invalidateQueries({ queryKey: ['document-llm', documentId] });
        }
      },
      () => {
        console.warn(`SSE connection lost for document ${documentId}`);
      }
    );
    return () => source.close();
  }, [documentId, isOpen, refetch, queryClient]);

  useEffect(() => {
    if (!isOpen) {
      setActiveTab('overview');
      initializedTabDocumentRef.current = null;
      setHighlightSources(false);
      setSearchTerm('');
    }
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !documentId || !document || initializedTabDocumentRef.current === documentId) return;

    setActiveTab(document.status === 'done' ? 'metadata' : 'overview');
    initializedTabDocumentRef.current = documentId;
  }, [isOpen, documentId, document]);

  if (!isOpen || !documentId) return null;

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return `${date.getDate()}-${date.getMonth() + 1}-${date.getFullYear()}`;
  };

  const formatDateTime = (dateString: string) => {
    const date = new Date(dateString);
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${day}-${month}-${year} ${hours}:${minutes}`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'done': return 'text-green-600';
      case 'processing': return 'text-blue-600';
      case 'error': return 'text-red-600';
      default: return 'text-amber-600';
    }
  };

  const getRiskColor = (score: number) => {
    if (score >= 70) return 'bg-red-500';
    if (score >= 40) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const copyToClipboard = (text: string) => navigator.clipboard.writeText(text);

  const downloadArtifact = async (path: string, filename: string) => {
    try {
      const blob = await getDocumentArtifact(documentId, path);
      const url = window.URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = filename;
      window.document.body.appendChild(a);
      a.click();
      window.document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error: any) {
      console.error('Download failed:', error);
      alert(`Download mislukt: ${error?.message || 'Bestand niet gevonden'}`);
    }
  };

  if (!isOpen) return null;

  return (
    <>
      {!showDocumentViewer && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[150] flex items-center justify-center p-0 sm:p-4 overflow-hidden"
          onClick={(e) => {
            // Close sidebar if clicking outside the modal
            if (examplesSidebar && e.target === e.currentTarget) {
              setExamplesSidebar(null);
            }
          }}
        >
          <div className="relative w-full max-w-4xl h-full sm:h-[90vh] m-0 sm:m-4 overflow-hidden">
            <div className={`glass-card overflow-hidden flex flex-col w-full h-full min-h-0 transition-all duration-300 rounded-none sm:rounded-xl`}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-200 flex-shrink-0">
          <div className="flex items-center space-x-3 min-w-0">
            <FontAwesomeIcon icon={faFile} className="text-slate-500 w-5 h-5 flex-shrink-0" />
            <div className="min-w-0">
              <h2 className="text-slate-800 text-lg font-semibold truncate">
                {document?.original_filename || 'Loading...'}
              </h2>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-slate-500">Document #{documentId}</span>
                {/* Processing Status Indicator */}
                {(document?.status === 'processing' || document?.status === 'queued') && (
                  <span className="flex items-center gap-1.5 px-2 py-0.5 bg-[#22d3d3]/10 text-[#0e9f9f] rounded-full animate-pulse">
                    <FontAwesomeIcon icon={faSpinner} className="w-3 h-3 animate-spin" />
                    <span>{formatStage(document.stage) || 'Verwerken...'}</span>
                    {document.progress > 0 && <span>({document.progress}%)</span>}
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2 flex-shrink-0">
            {document && (
              <button
                onClick={() => {
                  // Use direct API URL instead of blob URL for better browser compatibility
                  const artifactPath = `original/${document.original_filename}`;
                  setDocumentViewerPath(artifactPath);
                  setShowDocumentViewer(true);
                }}
                className="flex items-center space-x-2 px-3 py-1.5 bg-slate-100 text-slate-700 text-sm rounded-lg hover:bg-slate-200 border border-slate-200 cursor-pointer"
                title="Bekijk origineel bestand"
              >
                <FontAwesomeIcon icon={document.mime_type === 'application/pdf' ? faFilePdf : faImage} className="w-3 h-3" />
                <span>Bekijk</span>
              </button>
            )}
            <button
              onClick={() => analyzeMutation.mutate()}
              disabled={analyzeMutation.isPending || document?.status === 'processing' || document?.status === 'queued'}
              className={`flex items-center space-x-2 px-3 py-1.5 text-white text-sm rounded-lg disabled:opacity-50 cursor-pointer transition-all ${
                analyzeMutation.isPending || document?.status === 'processing' || document?.status === 'queued'
                  ? 'bg-slate-300'
                  : 'bg-[#22d3d3] hover:bg-[#1ab8b8]'
              }`}
            >
              <FontAwesomeIcon 
                icon={faRedo} 
                className={`w-3 h-3 ${analyzeMutation.isPending ? 'animate-spin' : ''}`} 
              />
              <span>
                {analyzeMutation.isPending ? 'In queue...' : 
                 document?.status === 'processing' || document?.status === 'queued' ? 'Bezig...' : 
                 'Re-run'}
              </span>
            </button>
            <button onClick={onClose} className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg">
              <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
            </button>
          </div>
        </div>

        {isLoading && (
          <div className="p-8 text-center">
            <FontAwesomeIcon icon={faSpinner} className="text-slate-400 text-3xl mb-4 animate-spin" />
            <p className="text-slate-500">Loading...</p>
          </div>
        )}

        {document && !isLoading && (
          <>
            {/* Processing Overlay */}
            {(document.status === 'processing' || document.status === 'queued') && (
              <div className="bg-gradient-to-r from-[#22d3d3]/8 to-[#FFC1F3]/6 border-b border-[#22d3d3]/20 px-4 py-3">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <FontAwesomeIcon icon={faSpinner} className="w-5 h-5 text-blue-600 animate-spin" />
                    <div className="absolute inset-0 bg-blue-100 rounded-full animate-ping" />
                  </div>
                  <div className="flex-1">
                    <div className="text-slate-800 text-sm font-medium">
                      {document.stage || 'Document wordt verwerkt...'}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] rounded-full transition-all duration-500"
                          style={{ width: `${document.progress || 0}%` }}
                        />
                      </div>
                      <span className="text-slate-500 text-xs">{document.progress || 0}%</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tabs */}
            <div className="flex border-b border-slate-200 flex-shrink-0 overflow-x-auto">
              {[
                { id: 'overview', label: 'Overview', icon: faInfoCircle },
                { id: 'text', label: 'Text', icon: faFile },
                { id: 'metadata', label: 'Data', icon: faSearch },
                { id: 'llm', label: 'LLM', icon: faRobot },
                { id: 'forensics', label: 'Forensics', icon: faShieldAlt },
              ].map((tab) => {
                const isDisabled = (document.status === 'processing' || document.status === 'queued') && tab.id !== 'overview';
                return (
                  <button
                    key={tab.id}
                    onClick={() => !isDisabled && setActiveTab(tab.id as TabType)}
                    disabled={isDisabled}
                    className={`flex items-center space-x-1 sm:space-x-2 px-2 sm:px-4 py-2 text-xs sm:text-sm font-medium transition-colors whitespace-nowrap shrink-0 ${
                      isDisabled ? 'text-slate-500 cursor-not-allowed' :
                      activeTab === tab.id ? 'text-[#22d3d3] border-b-2 border-[#22d3d3]' : 'text-slate-500 hover:text-slate-800'
                    }`}
                  >
                    <FontAwesomeIcon icon={tab.icon} className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
                    <span>{tab.label}</span>
                    {isDisabled && <FontAwesomeIcon icon={faSpinner} className="w-2.5 h-2.5 animate-spin ml-1" />}
                  </button>
                );
              })}
            </div>

            {/* Duplicate warning banner */}
            {document.duplicate_of && (
              <div className="mx-4 mb-2 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 shrink-0" />
                <span>
                  Dit bestand is al eerder geüpload als{' '}
                  <span className="font-mono font-semibold">document #{document.duplicate_of}</span>.
                  Mogelijk duplicaat of gewijzigde versie.
                </span>
              </div>
            )}

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden min-h-0">
              {activeTab === 'overview' && <OverviewTab document={document} documentId={documentId} formatFileSize={formatFileSize} formatDate={formatDate} formatDateTime={formatDateTime} formatStage={formatStage} getStatusColor={getStatusColor} getRiskColor={getRiskColor} onShowExamples={(signalIndex, examples) => setExamplesSidebar({ signalIndex, examples })} feedbackState={{ feedbackDone, showRejectInput, rejectCorrectedType, setShowRejectInput, setRejectCorrectedType, onConfirm: () => { setFeedbackError(null); feedbackMutation.mutate({ action: 'confirmed' }); }, onReject: (t) => { setFeedbackError(null); feedbackMutation.mutate({ action: 'rejected', correctedType: t }); }, isPending: feedbackMutation.isPending, error: feedbackError }} />}
              {activeTab === 'text' && <TextTab document={document} highlightSources={highlightSources} searchTerm={searchTerm} onHighlightToggle={() => setHighlightSources(!highlightSources)} onSearchChange={setSearchTerm} downloadArtifact={downloadArtifact} extractedText={extractedText} textLoading={textLoading} />}
              {activeTab === 'metadata' && <MetadataTab documentId={documentId} document={document} copyToClipboard={copyToClipboard} downloadArtifact={downloadArtifact} />}
              {activeTab === 'llm' && <LLMTab documentId={documentId} document={document} downloadArtifact={downloadArtifact} />}
              {activeTab === 'forensics' && <ForensicsTab documentId={documentId!} document={document} isOpen={isOpen} />}
            </div>
          </>
        )}
            </div>

            {/* Examples Sidebar - Bottom sheet on mobile, right side on desktop */}
            {examplesSidebar && (
              <div className="fixed sm:absolute inset-x-0 bottom-0 sm:inset-auto sm:top-0 sm:left-full sm:ml-0 glass-card w-full sm:w-96 max-h-[70vh] sm:max-h-none flex-shrink-0 border-t sm:border-t-0 sm:border-l border-slate-200 flex flex-col rounded-t-xl sm:rounded-none sm:rounded-r-xl sm:h-full shadow-2xl animate-slide-in-up sm:animate-slide-in-right overflow-hidden z-[60]">
                <div className="flex items-center justify-between p-4 border-b border-slate-200 flex-shrink-0">
                  <h3 className="text-slate-800 font-semibold text-sm">Voorbeeldteksten</h3>
                  <button
                    onClick={() => setExamplesSidebar(null)}
                    className="p-1.5 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer"
                  >
                    <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-4 min-h-0">
                  {examplesSidebar.examples.unicode_examples && examplesSidebar.examples.unicode_examples.length > 0 && (
                    <div>
                      <div className="text-xs font-semibold text-slate-600 mb-2">
                        Unicode tekens gevonden:
                      </div>
                      <div className="space-y-2">
                        {examplesSidebar.examples.unicode_examples.map((example: string, idx: number) => (
                          <div 
                            key={idx}
                            className="bg-white rounded-lg p-3 text-xs font-mono text-slate-700 border border-slate-200 break-all leading-relaxed"
                          >
                            <div className="whitespace-pre-wrap">{formatExampleText(example, true)}</div>
                          </div>
                        ))}
                      </div>
                      <div className="text-xs text-slate-400 mt-2">
                        Deze tekens zijn niet standaard ASCII en kunnen wijzen op manipulatie.
                      </div>
                    </div>
                  )}
                  
                  {examplesSidebar.examples.repetition_examples && examplesSidebar.examples.repetition_examples.length > 0 && (
                    <div className="mt-6">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-1 h-4 bg-gradient-to-b from-orange-400 to-red-400 rounded-full"></div>
                        <h4 className="text-sm font-bold text-slate-800">Herhalende tekst patronen</h4>
                      </div>
                      <div className="space-y-3">
                        {examplesSidebar.examples.repetition_examples.map((example: string, idx: number) => {
                          const formattedText = formatExampleText(example, false);
                          return (
                            <div 
                              key={idx}
                              className="bg-gradient-to-br from-orange-500/10 to-red-500/10 rounded-lg p-4 text-sm text-slate-700 border border-orange-200 break-words shadow-lg"
                            >
                              <div className="flex items-start gap-3">
                                <span className="text-orange-600 font-bold text-xs mt-0.5 flex-shrink-0">#{idx + 1}</span>
                                <div className="flex-1">
                                  <div className="leading-relaxed text-slate-700">{formattedText}</div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded-lg">
                        <div className="text-xs text-slate-500 leading-relaxed">
                          <span className="font-semibold text-orange-600">Waarom is dit verdacht?</span>
                          <br />
                          Deze herhalingen kunnen wijzen op automatische tekst generatie of manipulatie. Woorden die meerdere keren in een zin voorkomen zijn ongebruikelijk in natuurlijke taal.
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Document Viewer Modal - Fullscreen overlay */}
      {showDocumentViewer && documentViewerPath && documentId && (
        <DocumentViewerModal
          isOpen={showDocumentViewer}
          onClose={() => {
            setShowDocumentViewer(false);
            setDocumentViewerPath(null);
          }}
          documentUrl={(() => {
            const token = typeof window !== 'undefined' ? localStorage.getItem('mproof_token') : null;
            const base = `${API_BASE_URL}/api/documents/${documentId}/artifact?path=${encodeURIComponent(documentViewerPath)}`;
            return token ? `${base}&token=${encodeURIComponent(token)}` : base;
          })()}
          filename={document?.original_filename || ''}
          mimeType={document?.mime_type || ''}
          onDownload={() => downloadArtifact(documentViewerPath, document?.original_filename || 'document')}
          evidence={viewerEvidence}
        />
      )}
    </>
  );
}

function OverviewTab({ document, formatFileSize, formatDate, formatDateTime, formatStage, getStatusColor, getRiskColor, onShowExamples, documentId, feedbackState }: {
  document: Document;
  formatFileSize: (bytes: number) => string;
  formatDate: (date: string) => string;
  formatDateTime: (date: string) => string;
  formatStage: (stage: string | undefined) => string;
  getStatusColor: (status: string) => string;
  getRiskColor: (score: number) => string;
  onShowExamples: (signalIndex: number, examples: any) => void;
  documentId?: number;
  feedbackState?: {
    feedbackDone: 'confirmed' | 'rejected' | null;
    showRejectInput: boolean;
    rejectCorrectedType: string;
    setShowRejectInput: (v: boolean | ((prev: boolean) => boolean)) => void;
    setRejectCorrectedType: (v: string) => void;
    onConfirm: () => void;
    onReject: (correctedType?: string) => void;
    isPending: boolean;
    error?: string | null;
  };
}) {
  // Load fraud analysis for forensics signals
  const { data: fraudReport } = useQuery({
    queryKey: ['fraud-analysis', documentId],
    queryFn: () => getFraudAnalysis(documentId!),
    enabled: !!documentId && document?.status === 'done',
    staleTime: Infinity,
    gcTime: Infinity,
  });
  return (
    <div className="p-3 sm:p-4 space-y-3 sm:space-y-4 overflow-y-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4">
        {/* File Info */}
        <div className="bg-blue-50 rounded-lg p-3 sm:p-4 border border-blue-200">
          <h3 className="text-slate-800 font-medium mb-2 sm:mb-3 flex items-center space-x-2 text-sm sm:text-base">
            <FontAwesomeIcon icon={faFile} className="text-blue-600 w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span>File Info</span>
          </h3>
          <div className="space-y-1.5 sm:space-y-2 text-xs sm:text-sm">
            <div className="flex justify-between"><span className="text-slate-500">Size:</span><span className="text-slate-800">{formatFileSize(document.size_bytes)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">Type:</span><span className="text-slate-800 text-xs">{document.mime_type}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">SHA256:</span><span className="text-slate-800 font-mono text-xs">{document.sha256?.substring(0, 12)}...</span></div>
            <div className="flex justify-between"><span className="text-slate-500">Uploaded:</span><span className="text-slate-800">{formatDateTime(document.created_at)}</span></div>
          </div>
        </div>

        {/* Status */}
        <div className={`rounded-lg p-4 border ${
          document.status === 'processing' || document.status === 'queued'
            ? 'bg-blue-50 border-blue-200'
            : document.status === 'error'
            ? 'bg-red-50 border-red-200'
            : 'bg-green-50 border-green-200'
        }`}>
          <h3 className="text-slate-800 font-medium mb-3 flex items-center space-x-2">
            {(document.status === 'processing' || document.status === 'queued') ? (
              <FontAwesomeIcon icon={faSpinner} className="text-blue-600 w-4 h-4 animate-spin" />
            ) : document.status === 'error' ? (
              <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-600 w-4 h-4" />
            ) : (
              <FontAwesomeIcon icon={faCheck} className="text-green-600 w-4 h-4" />
            )}
            <span>Status</span>
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-slate-500">Status:</span>
              <span className={`capitalize font-medium px-2 py-0.5 rounded text-xs ${getStatusColor(document.status)}`}>{document.status}</span>
            </div>
            {document.stage && <div className="flex justify-between"><span className="text-slate-500">Stage:</span><span className="text-slate-800 text-sm">{formatStage(document.stage)}</span></div>}
            {(document.status === 'processing' || document.status === 'queued') && document.progress !== undefined && (
              <div className="mt-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-500">Progress</span>
                  <span className="text-slate-800">{document.progress}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2">
                  <div className="h-2 bg-blue-400 rounded-full transition-all" style={{ width: `${document.progress}%` }} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Classification & Risk */}
      {(document.doc_type_slug || document.risk_score !== null) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {document.doc_type_slug && (
            <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
              <h3 className="text-slate-800 font-medium mb-3 flex items-center space-x-2">
                <FontAwesomeIcon icon={faEye} className="text-purple-600 w-4 h-4" />
                <span>Classification</span>
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Type:</span>
                  <span className="text-slate-800 font-medium px-2 py-0.5 bg-purple-100 rounded">{formatDocumentTypeName(document.doc_type_slug)}</span>
                </div>
                {document.doc_type_confidence && (
                  <div className="flex justify-between"><span className="text-slate-500">Confidence:</span><span className="text-slate-800">{Math.round(document.doc_type_confidence * 100)}%</span></div>
                )}
              </div>

              {/* Feedback buttons — only for done, non-unknown docs */}
              {document.status === 'done' && document.doc_type_slug && document.doc_type_slug !== 'unknown' && feedbackState && (
                <div className="mt-3 pt-3 border-t border-slate-200">
                  {(feedbackState.feedbackDone ?? document.feedback_status) === 'confirmed' ? (
                    <div className="flex items-center gap-1.5 text-green-600 text-xs">
                      <FontAwesomeIcon icon={faCheck} className="w-3 h-3" />
                      <span>Classificatie bevestigd — document toegevoegd aan trainingsdata</span>
                    </div>
                  ) : (feedbackState.feedbackDone ?? document.feedback_status) === 'rejected' ? (
                    <div className="flex items-center gap-1.5 text-red-600 text-xs">
                      <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                      <span>Afgekeurd{document.corrected_doc_type ? ` → ${document.corrected_doc_type}` : ''}</span>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="text-slate-400 text-xs">Classificatie correct?</div>
                      {feedbackState.error && (
                        <div className="text-red-500 text-xs">{feedbackState.error}</div>
                      )}
                      <div className="flex gap-2">
                        <button
                          onClick={() => feedbackState.onConfirm()}
                          disabled={feedbackState.isPending}
                          className="flex items-center gap-1 px-2 py-1 text-xs bg-green-100 hover:bg-green-200 text-green-700 rounded border border-green-200 transition-colors disabled:opacity-50"
                        >
                          <FontAwesomeIcon icon={faCheck} className="w-3 h-3" />
                          Correct
                        </button>
                        <button
                          onClick={() => feedbackState.setShowRejectInput((v: boolean) => !v)}
                          className="flex items-center gap-1 px-2 py-1 text-xs bg-red-100 hover:bg-red-200 text-red-600 rounded border border-red-200 transition-colors"
                        >
                          <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                          Fout type
                        </button>
                      </div>
                      {feedbackState.showRejectInput && (
                        <div className="flex gap-2 items-center">
                          <input
                            type="text"
                            placeholder="Juist type slug (optioneel)"
                            value={feedbackState.rejectCorrectedType}
                            onChange={e => feedbackState.setRejectCorrectedType(e.target.value)}
                            className="flex-1 text-xs bg-slate-50 border border-slate-200 rounded px-2 py-1 text-slate-800 placeholder-slate-300 focus:outline-none focus:border-red-500/50"
                          />
                          <button
                            onClick={() => feedbackState.onReject(feedbackState.rejectCorrectedType || undefined)}
                            disabled={feedbackState.isPending}
                            className="px-2 py-1 text-xs bg-red-100 hover:bg-red-200 text-red-600 rounded border border-red-200 transition-colors disabled:opacity-50"
                          >
                            {feedbackState.isPending ? '...' : 'Bevestig'}
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Show explanation if document is "unknown" and there's a rejection reason */}
              {document.doc_type_slug === 'unknown' && document.metadata_validation_json?.classification_scores && (
                ((document.metadata_validation_json.classification_scores as any)?.naive_bayes?.rejection_reason || 
                 (document.metadata_validation_json.classification_scores as any)?.bert?.rejection_reason) && (
                  <div className="mt-3 pt-3 border-t border-slate-200">
                    <div className="bg-green-50 border border-green-200 rounded p-2">
                      <div className="text-green-700 text-[10px] italic">
                        ✓ Dit is correct gedrag - het systeem voorkomt verkeerde classificaties door deze regel te respecteren
                      </div>
                    </div>
                  </div>
                )
              )}
              
              {/* Classification Scores - Always show if available, including failures */}
              {document.metadata_validation_json?.classification_scores && (
                <div className="mt-3 pt-3 border-t border-slate-200">
                  <div className="text-slate-400 text-xs mb-1.5">Classifier Scores:</div>
                  <div className="grid grid-cols-2 gap-2">
                    {(document.metadata_validation_json.classification_scores as any)?.naive_bayes && (
                      (document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'failed' ||
                      (document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'no_result' ? (
                        <div className="bg-purple-50 border border-purple-200 rounded px-2 py-1.5 opacity-60">
                          <div className="text-purple-500 text-[10px] font-medium">NB: {(document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'failed' ? 'Fout' : 'Geen model'}</div>
                          {/* Show all scores if available - filter out 0% */}
                          {(document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores && (
                            <div className="mt-1 pt-1 border-t border-purple-200">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries((document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores as Record<string, number>)
                                  .sort(([, a], [, b]) => b - a)
                                  .filter(([, score]) => Math.round(score * 100) > 0)
                                  .map(([label, score]) => (
                                    <span key={label} className="text-[9px] bg-purple-100 px-1.5 py-0.5 rounded text-slate-500">
                                      {label}: {Math.round(score * 100)}%
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'rejected' ? (
                        <div className="bg-purple-50 border border-purple-200 rounded px-2 py-1.5 opacity-75">
                          <div className="flex items-center justify-between">
                            <span className="text-purple-500 text-[10px] font-medium">NB:</span>
                            <span className="text-slate-600 text-[10px]">{(document.metadata_validation_json.classification_scores as any).naive_bayes.label}</span>
                            <span className="text-purple-600 text-[10px]">{Math.round((document.metadata_validation_json.classification_scores as any).naive_bayes.confidence * 100)}%</span>
                          </div>
                          <div className="text-red-600 text-[9px]">❌ {(document.metadata_validation_json.classification_scores as any).naive_bayes.rejection_reason || 'Afgewezen'}</div>
                          {/* Show all scores if available - filter out 0% */}
                          {(document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores && (
                            <div className="mt-1 pt-1 border-t border-purple-200">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries((document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores as Record<string, number>)
                                  .sort(([, a], [, b]) => b - a)
                                  .filter(([, score]) => Math.round(score * 100) > 0)
                                  .map(([label, score]) => (
                                    <span key={label} className="text-[9px] bg-purple-100 px-1.5 py-0.5 rounded text-slate-500">
                                      {label}: {Math.round(score * 100)}%
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'below_threshold' ? (
                        <div className="bg-purple-50 border border-purple-200 rounded px-2 py-1.5 opacity-75">
                          <div className="flex items-center justify-between">
                            <span className="text-purple-500 text-[10px] font-medium">NB:</span>
                            <span className="text-slate-600 text-[10px]">{(document.metadata_validation_json.classification_scores as any).naive_bayes.label}</span>
                            <span className="text-purple-600 text-[10px]">{Math.round((document.metadata_validation_json.classification_scores as any).naive_bayes.confidence * 100)}%</span>
                          </div>
                          <div className="text-amber-600 text-[9px]">⚠️ Onder threshold ({Math.round(((document.metadata_validation_json.classification_scores as any).naive_bayes.threshold || 0) * 100)}%)</div>
                          {/* Show all scores if available - filter out 0% */}
                          {(document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores && (
                            <div className="mt-1 pt-1 border-t border-purple-200">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries((document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores as Record<string, number>)
                                  .sort(([, a], [, b]) => b - a)
                                  .filter(([, score]) => Math.round(score * 100) > 0)
                                  .map(([label, score]) => (
                                    <span key={label} className="text-[9px] bg-purple-100 px-1.5 py-0.5 rounded text-slate-500">
                                      {label}: {Math.round(score * 100)}%
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="bg-purple-50 border border-purple-200 rounded px-2 py-1.5">
                          <div className="flex items-center justify-between">
                            <span className="text-purple-500 text-[10px] font-medium">NB:</span>
                            <span className="text-slate-800 text-[10px]">{(document.metadata_validation_json.classification_scores as any).naive_bayes.label}</span>
                            <span className="text-purple-600 text-[10px]">{Math.round((document.metadata_validation_json.classification_scores as any).naive_bayes.confidence * 100)}%</span>
                          </div>
                          {/* Show all scores if available - filter out 0% */}
                          {(document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores && (
                            <div className="mt-1 pt-1 border-t border-purple-200">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries((document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores as Record<string, number>)
                                  .sort(([, a], [, b]) => b - a)
                                  .filter(([, score]) => Math.round(score * 100) > 0)
                                  .map(([label, score]) => (
                                    <span key={label} className="text-[9px] bg-purple-100 px-1.5 py-0.5 rounded text-slate-500">
                                      {label}: {Math.round(score * 100)}%
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    )}
                    {(document.metadata_validation_json.classification_scores as any).bert && (
                      (document.metadata_validation_json.classification_scores as any).bert.status === 'failed' || 
                      (document.metadata_validation_json.classification_scores as any).bert.status === 'no_result' ? (
                        <div className="bg-blue-50 border border-blue-200 rounded p-2 opacity-60">
                          <div className="text-blue-500 text-xs font-medium mb-0.5">BERT</div>
                          <div className="text-slate-500 text-xs">
                            {(document.metadata_validation_json.classification_scores as any).bert.status === 'failed' 
                              ? `Fout: ${(document.metadata_validation_json.classification_scores as any).bert.error?.substring(0, 50) || 'Onbekende fout'}...`
                              : (document.metadata_validation_json.classification_scores as any).bert.reason || 'Geen resultaat'}
                          </div>
                        </div>
                      ) : (document.metadata_validation_json.classification_scores as any).bert.status === 'rejected' ? (
                        <div className="bg-blue-50 border border-blue-200 rounded p-2 opacity-75">
                          <div className="text-blue-500 text-xs font-medium mb-0.5">BERT</div>
                          <div className="text-slate-600 text-xs font-medium">
                            {(document.metadata_validation_json.classification_scores as any).bert.label}
                          </div>
                          <div className="text-blue-600 text-[10px] mt-0.5">
                            {Math.round((document.metadata_validation_json.classification_scores as any).bert.confidence * 100)}% confidence
                          </div>
                          <div className="text-red-600 text-[10px] mt-1">
                            ❌ Afgewezen: {(document.metadata_validation_json.classification_scores as any).bert.rejection_reason || 'Onbekende reden'}
                          </div>
                        </div>
                      ) : (
                        <div className="bg-blue-50 border border-blue-200 rounded p-2">
                          <div className="text-blue-500 text-xs font-medium mb-0.5">BERT</div>
                          <div className="text-slate-800 text-xs">
                            {(document.metadata_validation_json.classification_scores as any).bert.label}
                          </div>
                          <div className="text-blue-600 text-[10px] mt-0.5">
                            {Math.round((document.metadata_validation_json.classification_scores as any).bert.confidence * 100)}% confidence
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
              
              {document.doc_type_rationale ? (
                <div className="mt-3 pt-3 border-t border-slate-200">
                  <div className="text-slate-400 text-xs mb-1.5">Classificatie methode:</div>
                  <div className="flex flex-wrap gap-1.5 mb-2">
                    {document.doc_type_rationale.includes('STRONG keyword match') || document.doc_type_rationale.includes('STRONG') ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-green-100 border border-green-200 text-green-700 text-[10px]">
                        ✅ 100% Keyword match
                      </span>
                    ) : document.doc_type_rationale.includes('Deterministic') ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-yellow-100 border border-yellow-200 text-yellow-700 text-[10px]">
                        ⚠️ Keyword/regex match
                      </span>
                    ) : null}
                    {document.doc_type_rationale.includes('Local classifier') || document.doc_type_rationale.includes('NAIVE_BAYES') || document.doc_type_rationale.includes('BERT') ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-100 border border-blue-200 text-blue-700 text-[10px]">
                        🤖 Getraind model
                      </span>
                    ) : null}
                    {document.doc_type_rationale.includes('LLM') && !document.doc_type_rationale.includes('STRONG') ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-purple-100 border border-purple-200 text-purple-700 text-[10px]">
                        🧠 AI classificatie
                      </span>
                    ) : null}
                  </div>
                  {/* Show matched keywords for strong deterministic matches */}
                  {document.doc_type_rationale.includes('STRONG keyword match') && document.doc_type_rationale.includes('matched keywords:') && (
                    <div className="mt-2 bg-green-50 border border-green-200 rounded p-2">
                      <div className="text-green-700 text-xs font-medium mb-1">Gematchte keywords:</div>
                      <div className="flex flex-wrap gap-1">
                        {document.doc_type_rationale
                          .split('matched keywords:')[1]
                          ?.split('|')[0]
                          ?.split(',')
                          ?.map((kw: string) => kw.trim())
                          ?.filter(Boolean)
                          ?.map((keyword: string, i: number) => (
                            <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-green-100 border border-green-200 text-green-700 text-[10px]">
                              {keyword}
                            </span>
                          ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Skip Marker Info */}
                  {document.skip_marker_used && (
                    <div className="mt-2 bg-cyan-50 border border-cyan-200 rounded p-2">
                      <div className="text-cyan-700 text-xs font-medium mb-1 flex items-center gap-1.5">
                        <span>✂️</span>
                        <span>Skip Marker Toegepast</span>
                      </div>
                      <div className="text-slate-600 text-[11px] font-mono bg-slate-50 rounded px-2 py-1 break-all">
                        {document.skip_marker_used}
                      </div>
                      {document.skip_marker_position != null && (
                        <div className="text-cyan-600 text-[10px] mt-1">
                          Tekst afgekapt na positie {document.skip_marker_position.toLocaleString()}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {document.doc_type_rationale.includes('Deterministic') && (
                    <div className="text-[10px] text-slate-400 italic bg-yellow-500/5 border border-yellow-200 rounded p-2 mt-2">
                      ⚠️ <strong>Waarom onjuist?</strong> Deterministic matching (keywords/regex) heeft voorrang boven het getrainde model. Als je commitment agreement als bankafschrift wordt herkend, controleer de classification_hints van "bankafschrift" in document types - deze bevatten waarschijnlijk keywords die ook in commitment agreements voorkomen.
                    </div>
                  )}
                </div>
              ) : document.doc_type_slug === 'unknown' ? (
                <div className="mt-3 pt-3 border-t border-slate-200">
                  <div className="text-slate-400 text-xs mb-1.5">Classificatie methode:</div>
                  <div className="bg-gray-50 border border-gray-200 rounded p-2">
                    <div className="text-slate-600 text-[10px]">
                      Voor Onbekend document type wordt geen field matching uitgevoerd
                    </div>
                  </div>
                </div>
              ) : null}
              
              {/* Skip Marker Info - Show even if no rationale */}
              {!document.doc_type_rationale && document.skip_marker_used && (
                <div className="mt-3 pt-3 border-t border-slate-200">
                  <div className="bg-cyan-50 border border-cyan-200 rounded p-2">
                    <div className="text-cyan-700 text-xs font-medium mb-1 flex items-center gap-1.5">
                      <span>✂️</span>
                      <span>Skip Marker Toegepast</span>
                    </div>
                    <div className="text-slate-600 text-[11px] font-mono bg-slate-50 rounded px-2 py-1 break-all">
                      {document.skip_marker_used}
                    </div>
                    {document.skip_marker_position != null && (
                      <div className="text-cyan-600 text-[10px] mt-1">
                        Tekst afgekapt na positie {document.skip_marker_position.toLocaleString()}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Risk & Forensics - Always show, even if no signals */}
          {document.status === 'done' && (
            <div className="bg-red-50 rounded-lg p-4 border border-red-200">
              <h3 className="text-slate-800 font-medium mb-3 flex items-center space-x-2">
                <FontAwesomeIcon icon={faShieldAlt} className="text-red-600 w-4 h-4" />
                <span>Risk & Forensics</span>
              </h3>
              {document.risk_score !== null && document.risk_score !== undefined ? (
                <div className="flex items-center space-x-3 mb-3">
                  <span className={`text-2xl font-bold ${(document.risk_score ?? 0) >= 70 ? 'text-red-600' : (document.risk_score ?? 0) >= 40 ? 'text-amber-600' : 'text-green-600'}`}>
                    {document.risk_score ?? 0}
                  </span>
                  <div className="flex-1 bg-slate-200 rounded-full h-2">
                    <div className={`h-2 rounded-full ${getRiskColor(document.risk_score ?? 0)}`} style={{ width: `${document.risk_score ?? 0}%` }} />
                  </div>
                </div>
              ) : (
                <div className="mb-3 text-slate-500 text-xs">
                  Risk score niet beschikbaar
                </div>
              )}
              
              {fraudReport?.semantic_context?.top_matches?.length ? (
                <div className="mb-3 pb-3 border-b border-slate-200">
                  <div className="text-xs text-slate-500 mb-2">Documentcontext (BERT)</div>
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <div className="text-blue-700 text-xs mb-2">{fraudReport.semantic_context.summary}</div>
                    <div className="flex flex-wrap gap-1.5">
                      {fraudReport.semantic_context.top_matches.map((match) => (
                        <span key={match.label} className="text-[10px] bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                          {formatDocumentTypeName(match.label)} {Math.round(match.confidence * 100)}%
                        </span>
                      ))}
                    </div>
                    <div className="text-slate-400 text-[10px] mt-2">
                      Model: {fraudReport.semantic_context.model_used} · margin {Math.round(fraudReport.semantic_context.margin * 100)}%
                    </div>
                  </div>
                </div>
              ) : null}

              {fraudReport && fraudReport.signals.length > 0 ? (() => {
                const actionableSignals = fraudReport.signals.filter(s => s.risk_level?.toLowerCase() !== 'low');
                const signalsToShow = actionableSignals.slice(0, 4);
                if (signalsToShow.length === 0) {
                  return (
                    <div className="mt-3 text-slate-500 text-xs">
                      Alleen lage context- of kwaliteitswaarschuwingen gevonden. Geen sterke fraude-indicatoren.
                    </div>
                  );
                }

                return (
                  <div className="space-y-2 mt-3">
                    <div className="text-xs text-slate-500 mb-2">
                      {actionableSignals.length} fraude-indicator{actionableSignals.length !== 1 ? 'en' : ''} gevonden:
                    </div>
                    {signalsToShow.map((signal, i) => (
                      <div
                        key={`${signal.name}-${i}`}
                        className={`bg-slate-50 rounded-lg p-3 border ${
                          signal.risk_level === 'critical' || signal.risk_level === 'high'
                            ? 'border-red-200 bg-red-50'
                            : signal.risk_level === 'medium'
                              ? 'border-yellow-200 bg-yellow-50'
                              : 'border-blue-200 bg-blue-50'
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          <FontAwesomeIcon
                            icon={signal.risk_level === 'critical' || signal.risk_level === 'high' ? faExclamationTriangle : faInfoCircle}
                            className={`w-4 h-4 mt-0.5 ${
                              signal.risk_level === 'critical' || signal.risk_level === 'high' ? 'text-red-600' : 'text-amber-600'
                            }`}
                          />
                          <div className="flex-1 min-w-0">
                            <div className="font-semibold text-sm text-slate-800 mb-1">{signal.name.replace(/_/g, ' ')}</div>
                            <div className="text-slate-500 text-xs leading-relaxed">{signal.description}</div>
                            {signal.recommendation && (
                              <div className="mt-2 pt-2 border-t border-slate-200 text-slate-500 text-xs italic">
                                {signal.recommendation}
                              </div>
                            )}
                          </div>
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-50 shrink-0">
                            {Math.round(signal.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                    {actionableSignals.length > signalsToShow.length && (
                      <div className="text-[10px] text-slate-400">
                        +{actionableSignals.length - signalsToShow.length} meer in de Forensics tab
                      </div>
                    )}
                  </div>
                );
              })() : (
                <div className="mt-3 text-slate-500 text-xs">
                  Geen fraude-indicatoren gevonden. Document lijkt schoon.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {document.error_message && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="flex items-start space-x-2">
            <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-600 w-4 h-4 mt-0.5" />
            <div>
              <h4 className="text-red-600 font-medium text-sm">Error</h4>
              <p className="text-red-600 text-xs mt-1">{document.error_message}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MetadataTab({ documentId, document, copyToClipboard, downloadArtifact }: {
  documentId: number;
  document: Document;
  copyToClipboard: (text: string) => void;
  downloadArtifact: (path: string, filename: string) => void;
}) {
  const [copiedForExcel, setCopiedForExcel] = useState(false);
  const [expandedEvidenceKey, setExpandedEvidenceKey] = useState<string | null>(null);
  const hasDbData = !!document.metadata_json && Object.keys(document.metadata_json || {}).length > 0;

  const { data: docTypeFields } = useQuery<DocumentTypeField[]>({
    queryKey: ['doc-type-fields', document.doc_type_slug],
    queryFn: () => listDocumentTypeFields(document.doc_type_slug!),
    enabled: !!document.doc_type_slug,
    staleTime: 5 * 60 * 1000,
  });
  const fieldTypeMap = Object.fromEntries((docTypeFields || []).map(f => [f.key, f.field_type]));
  const { data: artifactData } = useQuery({
    queryKey: ['document-metadata-artifact', documentId],
    queryFn: () => getDocumentArtifactJson<Record<string, any>>(documentId, 'metadata/result.json'),
    enabled: !hasDbData && document.status === 'done',
    retry: false,
    staleTime: Infinity,
  });
  const data = hasDbData ? document.metadata_json : artifactData;
  const hasData = !!data && Object.keys(data).length > 0;
  
  // Get evidence data (multiple quotes per field)
  const evidenceFromDb = document.metadata_evidence_json as Record<string, any[]> | null;
  const { data: evidenceArtifact } = useQuery({
    queryKey: ['document-evidence-artifact', documentId],
    queryFn: () => getDocumentArtifactJson<Record<string, any[]>>(documentId, 'metadata/evidence.json'),
    enabled: !evidenceFromDb && document.status === 'done',
    retry: false,
    staleTime: Infinity,
  });
  const evidence = evidenceFromDb || evidenceArtifact || {};

  const formatExcelValue = (value: unknown): string => {
    if (value === null || value === undefined) {
      return '';
    }
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    return String(value).replace(/\s+/g, ' ').trim();
  };

  const copyMetadataForExcel = async () => {
    if (!data) return;

    const rows = [
      'Veld\tWaarde',
      ...Object.entries(data).map(([key, value]) => `${key.replace(/_/g, ' ')}\t${formatExcelValue(value)}`),
    ];

    await copyToClipboard(rows.join('\n'));
    setCopiedForExcel(true);
    window.setTimeout(() => setCopiedForExcel(false), 1500);
  };

  // Detect value type from content for smart rendering
  const detectValueType = (key: string, value: unknown): 'iban' | 'amount' | 'date' | 'email' | 'url' | 'bsn' | 'phone' | 'default' => {
    const k = key.toLowerCase();
    const s = String(value ?? '').trim();
    if (/iban/.test(k) || /^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,}/.test(s)) return 'iban';
    if (/bsn/.test(k) || /^\d{9}$/.test(s)) return 'bsn';
    if (/email/.test(k) || /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(s)) return 'email';
    if (/url|link|website/.test(k) || /^https?:\/\//.test(s)) return 'url';
    if (/bedrag|amount|prijs|waarde|kosten|rente|inkomen|schuld|hypotheek|lening|saldo|totaal/.test(k) ||
        /^[€$]?\s*[\d.,]+\s*$/.test(s) || /^[\d.,]+\s*euro$/i.test(s)) return 'amount';
    if (/phone|telefoon|tel/.test(k) || /^\+?[\d\s\-()]{7,}$/.test(s)) return 'phone';
    if (/datum|date|geldig|vervalt|aanvang|eind/.test(k) ||
        /^\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}$/.test(s) ||
        /^\d{4}[-\/]\d{2}[-\/]\d{2}/.test(s) ||
        /^\d{1,2}\s+(jan|feb|mrt|apr|mei|jun|jul|aug|sep|okt|nov|dec)/i.test(s)) return 'date';
    return 'default';
  };

  const formatIban = (s: string) => s.replace(/\s/g, '').replace(/(.{4})/g, '$1 ').trim();

  const parseAmountString = (s: string): number => {
    let clean = s.replace(/[€$£¥\s]/g, '');
    const lastDot = clean.lastIndexOf('.');
    const lastComma = clean.lastIndexOf(',');
    if (lastDot > lastComma) {
      // American/English: 39,212.40 — comma is thousands separator
      clean = clean.replace(/,/g, '');
    } else if (lastComma > lastDot) {
      // Dutch/European: 39.212,40 — dot is thousands separator
      clean = clean.replace(/\./g, '').replace(',', '.');
    } else if (lastComma !== -1) {
      // Only commas — check if decimal (≤2 digits after) or thousands
      const afterComma = clean.slice(lastComma + 1);
      clean = afterComma.length <= 2
        ? clean.replace(',', '.')
        : clean.replace(/,/g, '');
    }
    return parseFloat(clean);
  };

  const formatAmount = (s: string) => {
    const currency = /\$/.test(s) ? 'USD' : /£/.test(s) ? 'GBP' : /¥/.test(s) ? 'JPY' : 'EUR';
    const num = parseAmountString(s);
    if (isNaN(num)) return s;
    return new Intl.NumberFormat('nl-NL', { style: 'currency', currency }).format(num);
  };

  const renderMetadataValue = (key: string, value: unknown) => {
    if (value === null || value === undefined || value === '') {
      return <span className="text-slate-500 italic">Niet gevonden</span>;
    }

    if (Array.isArray(value)) {
      return (
        <div className="flex flex-wrap gap-1">
          {value.map((item, i) => (
            <span key={i} className="rounded bg-purple-100 px-1.5 py-0.5 text-[10px] text-purple-700">
              {typeof item === 'object' && item !== null ? JSON.stringify(item) : String(item)}
            </span>
          ))}
        </div>
      );
    }

    if (typeof value === 'object') {
      return (
        <pre className="max-h-24 overflow-auto rounded bg-slate-50 p-2 text-[10px] text-slate-500">
          {JSON.stringify(value, null, 2)}
        </pre>
      );
    }

    const s = String(value);
    // Use field_type from document type definition first, fall back to heuristic detection
    const dbFieldType = fieldTypeMap[key];
    const type = dbFieldType === 'money' || dbFieldType === 'currency' ? 'amount'
      : dbFieldType === 'iban' ? 'iban'
      : dbFieldType === 'date' ? 'date'
      : dbFieldType === 'number' ? 'default'
      : detectValueType(key, s);

    switch (type) {
      case 'iban':
        return (
          <span className="font-mono text-green-700 text-xs tracking-wider break-all">
            {formatIban(s)}
          </span>
        );
      case 'amount':
        return (
          <span className="font-mono text-yellow-700 text-xs font-medium">
            {formatAmount(s)}
          </span>
        );
      case 'date':
        return <span className="text-blue-700 text-xs">{s}</span>;
      case 'bsn':
        return (
          <span className="font-mono text-orange-600 text-xs tracking-wider" title="BSN — gevoelig gegeven">
            {s.slice(0, 3)} {s.slice(3, 6)} {s.slice(6)}
          </span>
        );
      case 'email':
        return <span className="text-cyan-700 text-xs break-all">{s}</span>;
      case 'url':
        return <span className="text-cyan-700 text-xs break-all underline">{s}</span>;
      default:
        return <span className="text-slate-700 break-words text-xs">{s}</span>;
    }
  };

  const getEvidenceMethodLabel = (method: string) => {
    switch (method) {
      case 'evidence':              return { label: 'exact', color: 'text-green-700 bg-green-50 border-green-200' };
      case 'auto_found':            return { label: 'auto', color: 'text-amber-700 bg-yellow-50 border-yellow-200' };
      case 'auto_found_normalized': return { label: 'norm', color: 'text-yellow-700 bg-yellow-50 border-yellow-200' };
      case 'regex':                 return { label: 'regex', color: 'text-purple-700 bg-purple-50 border-purple-200' };
      case 'deterministic':         return { label: 'det', color: 'text-blue-600 bg-blue-50 border-blue-200' };
      default:                      return { label: method || '?', color: 'text-slate-400 bg-slate-50 border-slate-200' };
    }
  };

  return (
    <div className="overflow-y-auto p-2 sm:p-3">
      <div className="sticky top-0 z-10 -mx-2 -mt-2 mb-2 flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-white px-2 py-2 backdrop-blur sm:-mx-3 sm:-mt-3 sm:px-3">
        <div className="min-w-0">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Data
            {hasData && <span className="ml-2 font-normal normal-case tracking-normal text-slate-400">{Object.keys(data!).length} velden</span>}
          </div>
          {document.doc_type_slug && (
            <div className="truncate text-[11px] text-slate-500">{formatDocumentTypeName(document.doc_type_slug)}</div>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={copyMetadataForExcel}
            disabled={!hasData}
            className="flex items-center space-x-1.5 rounded-md bg-blue-600 px-2.5 py-1.5 text-[11px] font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <FontAwesomeIcon icon={copiedForExcel ? faCheck : faCopy} className="w-3 h-3" />
            <span>{copiedForExcel ? 'Gekopieerd' : 'Excel copy'}</span>
          </button>
          <button onClick={() => downloadArtifact('metadata/result.json', 'metadata.json')} className="flex items-center space-x-1 rounded-md bg-slate-100 px-2 py-1.5 text-[11px] text-slate-600 hover:bg-slate-200">
            <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
            <span>JSON</span>
          </button>
          <button onClick={() => downloadArtifact('metadata/evidence.json', 'evidence.json')} className="flex items-center space-x-1 rounded-md bg-slate-100 px-2 py-1.5 text-[11px] text-slate-600 hover:bg-slate-200">
            <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
            <span>Bronnen</span>
          </button>
        </div>
      </div>

      {hasData ? (
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-100">
          <div
            className="grid gap-3 border-b border-slate-200 bg-slate-50 px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500"
            style={{ gridTemplateColumns: 'minmax(120px, 180px) minmax(0, 1fr)' }}
          >
            <div>Veld</div>
            <div>Waarde</div>
          </div>
          {Object.entries(data!).map(([key, value]) => {
            const fieldEvidence = evidence[key] || [];
            const hasEvidence = Array.isArray(fieldEvidence) && fieldEvidence.length > 0;
            const isExpanded = expandedEvidenceKey === key;
            
            return (
              <div key={key} className="border-b border-slate-200 last:border-b-0">
                <div
                  className="grid gap-3 px-2.5 py-1.5 text-xs transition-colors odd:bg-slate-50 hover:bg-slate-100"
                  style={{ gridTemplateColumns: 'minmax(120px, 180px) minmax(0, 1fr)' }}
                >
                  <div className="truncate font-medium text-blue-700/75">
                    {key.replace(/_/g, ' ')}
                  </div>
                  <div className="flex min-w-0 items-start justify-between gap-2 leading-relaxed text-slate-700">
                    <div className="min-w-0 flex-1">
                      {renderMetadataValue(key, value)}
                    </div>
                    {hasEvidence ? (
                      <button
                        onClick={() => setExpandedEvidenceKey(isExpanded ? null : key)}
                        className="shrink-0 rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 hover:bg-blue-200"
                        title={`${fieldEvidence.length} bron${fieldEvidence.length === 1 ? '' : 'nen'}`}
                      >
                        bron {fieldEvidence.length}
                      </button>
                    ) : null}
                  </div>
                </div>
                {hasEvidence && isExpanded && (
                  <div className="space-y-1 bg-slate-50 px-2.5 pb-2" style={{ paddingLeft: 'calc(min(max(120px, 28vw), 180px) + 1.25rem)' }}>
                    {fieldEvidence.map((ev: any, i: number) => {
                      const methodInfo = getEvidenceMethodLabel(ev.method || '');
                      return (
                        <div
                          key={i}
                          className="rounded border border-blue-200 bg-blue-50 px-2 py-1 text-[11px] text-blue-700"
                          title={`Pagina ${(ev.page || 0) + 1}, positie ${ev.start}-${ev.end}`}
                        >
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="text-blue-600">p{(ev.page || 0) + 1}</span>
                            <span className={`rounded border px-1 py-0 text-[9px] font-medium ${methodInfo.color}`}>{methodInfo.label}</span>
                            {ev.verified === false && (
                              <span className="rounded border border-red-200 bg-red-50 px-1 py-0 text-[9px] text-red-600">niet geverif.</span>
                            )}
                            <span className="text-blue-600">·</span>
                            <span className="flex-1">{ev.quote || ev.snippet}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-8">
          <FontAwesomeIcon icon={faInfoCircle} className="text-slate-400 w-8 h-8 mb-2" />
          <p className="text-slate-500 text-sm">No extracted data</p>
        </div>
      )}
    </div>
  );
}

// Separate component to prevent ref reset on parent re-render
function CollapsiblePromptBlock({ 
  id, 
  title, 
  content, 
  downloadPath, 
  downloadName, 
  description,
  isExpanded,
  onToggle,
  onDownload
}: { 
  id: string; 
  title: string; 
  content: string | null; 
  downloadPath?: string; 
  downloadName?: string; 
  description?: string;
  isExpanded: boolean;
  onToggle: () => void;
  onDownload: (path: string, filename: string) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const scrollPositionRef = useRef<number>(0);

  // Don't render if no content
  if (!content) return null;
  
  const preview = content.length > 100 ? content.substring(0, 100) + '...' : content;

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    e.stopPropagation();
    scrollPositionRef.current = e.currentTarget.scrollTop;
  };

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden">
      <div 
        className="flex items-center justify-between p-2 bg-slate-50 hover:bg-slate-100 transition-colors cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex-1 text-left text-slate-600 text-xs font-medium hover:text-slate-800">
          <div className="flex items-center gap-2">
            <span>{title}</span>
            {description && (
              <span className="text-slate-400 text-[10px] font-normal">({description})</span>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-2" onClick={(e) => e.stopPropagation()}>
          {downloadPath && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDownload(downloadPath, downloadName || 'download.txt');
              }}
              className="text-slate-400 hover:text-slate-800 p-1 cursor-pointer"
              title="Download"
            >
              <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
            </button>
          )}
          <div className="text-slate-400 p-1">
            <FontAwesomeIcon icon={isExpanded ? faChevronUp : faChevronDown} className="w-3 h-3" />
          </div>
        </div>
      </div>
      {isExpanded ? (
        <div 
          className="max-h-64 overflow-y-auto bg-slate-50"
          onScroll={handleScroll}
          onWheel={(e) => e.stopPropagation()}
          onTouchMove={(e) => e.stopPropagation()}
          ref={scrollRef}
        >
          <pre 
            className="text-slate-600 text-xs font-mono whitespace-pre-wrap break-words p-3 cursor-text select-text"
            style={{ margin: 0 }}
          >
            {content}
          </pre>
        </div>
      ) : (
        <div className="text-slate-400 text-xs font-mono p-2 truncate cursor-pointer" onClick={onToggle}>{preview}</div>
      )}
    </div>
  );
}

function LLMTab({ documentId, document, downloadArtifact }: {
  documentId: number;
  document: Document | undefined;
  downloadArtifact: (path: string, filename: string) => void;
}) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  const toggleSection = (id: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Fetch LLM artifacts for completed and failed runs so extraction errors are visible in the UI.
  const canFetchArtifacts = document?.status === 'done' || document?.status === 'error';
  
  const { data, isLoading } = useQuery({
    queryKey: ['document-llm', documentId],
    queryFn: async () => {
      const safeText = async (path: string): Promise<string | null> => {
        try { return await getDocumentArtifactText(documentId, path); }
        catch { return null; }
      };
      const safeJson = async <T,>(path: string): Promise<T | null> => {
        try { return await getDocumentArtifactJson<T>(documentId, path); }
        catch { return null; }
      };
      const safeChunkArtifacts = async (kind: 'error' | 'warning') => {
        const paths = Array.from({ length: 50 }, (_, index) => {
          const chunkNum = index + 1;
          return {
            chunkNum,
            path: `llm/extraction_${kind}_chunk_${chunkNum}.txt`,
          };
        });
        const results = await Promise.all(
          paths.map(async ({ chunkNum, path }) => ({
            chunkNum,
            path,
            content: await safeText(path),
          }))
        );

        return results.filter((item): item is { chunkNum: number; path: string; content: string } => !!item.content);
      };

      const classification = {
        prompt: await safeText('llm/classification_prompt.txt'),
        response: await safeText('llm/classification_response.txt'),
        result: await safeJson<Record<string, any>>('llm/classification_result.json'),
        error: await safeText('llm/classification_error.txt'),
        deterministic: await safeJson<Record<string, any>>('llm/classification_deterministic.json'),
        local: await safeJson<Record<string, any>>('llm/classification_local.json'),
        timing: await safeJson<Record<string, any>>('llm/classification_timing.json'),
      };
      const extraction = {
        prompt: await safeText('llm/extraction_prompt.txt'),
        response: await safeText('llm/extraction_response.txt'),
        result: await safeJson<Record<string, any>>('llm/extraction_result.json'),
        error: await safeText('llm/extraction_error.txt'),
        chunkErrors: await safeChunkArtifacts('error'),
        chunkWarnings: await safeChunkArtifacts('warning'),
        timing: await safeJson<Record<string, any>>('llm/extraction_timing.json'),
        skipped: await safeJson<Record<string, any>>('llm/extraction_skipped.json'),
      };
      return { classification, extraction };
    },
    enabled: !!documentId && canFetchArtifacts,
    retry: false,
    staleTime: 0, // Always refetch to get latest prompts (especially after rerun)
    refetchOnMount: 'always', // Force refetch when component mounts
    gcTime: 0, // Don't cache at all (previously called cacheTime)
  });

  // Show appropriate message based on document status - BEFORE any data fetching
  if (!canFetchArtifacts) {
    if (document?.status === 'processing') {
      return (
        <div className="p-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
            <FontAwesomeIcon icon={faRobot} className="text-blue-600 w-8 h-8 mb-2 animate-pulse" />
            <p className="text-slate-800 font-medium text-sm">Processing...</p>
            <p className="text-slate-500 text-xs">{formatStage(document.stage)}</p>
            {document.progress !== undefined && (
              <div className="w-full bg-slate-200 rounded-full h-1.5 mt-2">
                <div className="h-1.5 bg-blue-400 rounded-full transition-all" style={{ width: `${document.progress}%` }} />
              </div>
            )}
          </div>
        </div>
      );
    }
    
    // For pending, queued, error, or any non-done status
    return (
      <div className="p-4 text-center">
        <FontAwesomeIcon icon={faRobot} className="text-slate-400 w-8 h-8 mb-2" />
        <p className="text-slate-500 text-sm">
          {document?.status === 'pending' || document?.status === 'queued' 
            ? 'Click "Re-run" to start analysis' 
            : document?.status === 'error'
            ? 'Analysis failed - click "Re-run" to retry'
            : 'Run analysis to see LLM logs'}
        </p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="p-4 flex items-center justify-center">
        <FontAwesomeIcon icon={faSpinner} className="text-blue-600 w-6 h-6 animate-spin" />
      </div>
    );
  }

  // This is now only reached when status is 'done' but no data was found
  if (!data) {
    return (
      <div className="p-4 text-center">
        <FontAwesomeIcon icon={faRobot} className="text-slate-400 w-8 h-8 mb-2" />
        <p className="text-slate-500 text-sm">Run analysis to see LLM logs</p>
      </div>
    );
  }

  const hasData = data?.classification.prompt || data?.classification.result || data?.classification.deterministic || data?.classification.local || data?.extraction.prompt || data?.extraction.result || data?.extraction.error || data?.extraction.chunkErrors?.length || data?.extraction.chunkWarnings?.length || data?.extraction.skipped;

  if (!hasData) {
    return (
      <div className="p-4 text-center">
        <FontAwesomeIcon icon={faRobot} className="text-slate-400 w-8 h-8 mb-2" />
        <p className="text-slate-500 text-sm">No LLM data available</p>
      </div>
    );
  }

  // Helper to render CollapsiblePromptBlock with current state
  const renderCollapsibleBlock = (id: string, title: string, content: string | null, downloadPath?: string, downloadName?: string, description?: string) => (
    <CollapsiblePromptBlock
      id={id}
      title={title}
      content={content}
      downloadPath={downloadPath}
      downloadName={downloadName}
      description={description}
      isExpanded={expandedSections.has(id)}
      onToggle={() => toggleSection(id)}
      onDownload={downloadArtifact}
    />
  );

  return (
    <div className="p-4 space-y-4">
      {/* Classification */}
      {/* Classification - Only show LLM classification logs (not the classification result itself, that's in Overview) */}
      {(data?.classification.prompt || data?.classification.result || data?.classification.error) && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-slate-800 font-medium text-sm flex items-center space-x-2">
              <FontAwesomeIcon icon={faRobot} className="text-blue-600 w-3 h-3" />
              <span>LLM Classification Logs</span>
            </h3>
            {data.classification.timing && (
              <div className="text-slate-500 text-xs flex items-center gap-1.5">
                <FontAwesomeIcon icon={faBolt} className="w-3 h-3" />
                <span>{data.classification.timing.duration_seconds ? `${data.classification.timing.duration_seconds.toFixed(2)}s` : 'N/A'}</span>
                {data.classification.timing.provider && (
                  <span className="text-slate-400">({data.classification.timing.provider})</span>
                )}
              </div>
            )}
          </div>

          {data.classification.error && (
            <div className="bg-red-50 border border-red-200 rounded p-2 text-xs text-red-600">
              <strong>Error:</strong> {data.classification.error.substring(0, 200)}...
            </div>
          )}

          {/* Show message if classification failed (no result but has prompt) */}
          {!data.classification.result && !data.classification.error && data.classification.prompt && (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-2 text-xs text-yellow-700">
              <strong>Let op:</strong> Classificatie is niet gelukt. Er is wel een prompt verzonden, maar geen resultaat ontvangen.
            </div>
          )}

          <div className="space-y-2">
            {renderCollapsibleBlock("class-prompt", "Prompt", data.classification.prompt, "llm/classification_prompt.txt", "classification_prompt.txt", "Volledige prompt naar LLM")}
            {/* Show Response and compare with Result to detect validation issues */}
            {(() => {
              const responseText = data.classification.response?.trim() || '';
              const resultJson = data.classification.result ? JSON.stringify(data.classification.result, null, 2) : '';
              const responseIsJson = responseText && (responseText.startsWith('{') || responseText.startsWith('['));
              
              // Parse response to compare with result
              let responseParsed = null;
              if (responseIsJson) {
                try {
                  responseParsed = JSON.parse(responseText);
                } catch (e) {
                  // Not valid JSON, show as-is
                }
              }
              
              // Check if response was rejected by validation
              const wasRejected = responseParsed && data.classification.result && 
                responseParsed.doc_type_slug !== data.classification.result.doc_type_slug &&
                data.classification.result.doc_type_slug === 'unknown';
              
              // Show Response if it exists and is different from Result
              if (data.classification.response && (!responseIsJson || !responseParsed || responseText !== resultJson)) {
                return (
                  <>
                    {wasRejected && (
                      <div className="bg-yellow-50 border border-yellow-200 rounded p-2 text-xs text-yellow-700">
                        <strong>Validatie waarschuwing:</strong> De LLM response is afgewezen door validatie. 
                        Het evidence dat de LLM heeft gegeven kon niet worden gevonden in de document tekst.
                      </div>
                    )}
                    {renderCollapsibleBlock("class-response", "Response", data.classification.response, "llm/classification_response.txt", "classification_response.txt", wasRejected ? "LLM output (afgewezen door validatie)" : "Ruwe LLM output")}
                  </>
                );
              }
              return null;
            })()}
            {data.classification.result && renderCollapsibleBlock("class-result", "Result (JSON)", JSON.stringify(data.classification.result, null, 2), "llm/classification_result.json", "classification_result.json", data.classification.result.doc_type_slug === 'unknown' ? "Gevalidateerd resultaat (afgewezen)" : "Gevalidateerd JSON resultaat")}
            
            {/* Show local classification info (NB/BERT scores, deterministic matches) */}
            {data.classification.local && (
              <>
                {data.classification.local.method === 'deterministic_strong' && data.classification.local.matched_keywords && (
                  <div className="bg-green-50 border border-green-200 rounded p-3 mt-3">
                    <div className="text-green-700 text-xs font-medium mb-2 flex items-center gap-2">
                      <span>✅</span>
                      <span>STRONG Keyword Match (100% confidence)</span>
                    </div>
                    <div className="text-slate-600 text-xs mb-1">Gematchte keywords:</div>
                    <div className="flex flex-wrap gap-1.5">
                      {data.classification.local.matched_keywords.map((keyword: string, i: number) => (
                        <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-green-100 border border-green-200 text-green-700 text-[10px]">
                          {keyword}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {renderCollapsibleBlock("class-local", "Local Classification (NB/BERT)", JSON.stringify(data.classification.local, null, 2), "llm/classification_local.json", "classification_local.json", "Naive Bayes en BERT scores, deterministic matches")}
              </>
            )}
          </div>
        </div>
      )}

      {/* Extraction */}
      {(data?.extraction.prompt || data?.extraction.result || data?.extraction.error || data?.extraction.chunkErrors?.length || data?.extraction.chunkWarnings?.length || data?.extraction.skipped) && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-slate-800 font-medium text-sm flex items-center space-x-2">
              <FontAwesomeIcon icon={faSearch} className="text-purple-600 w-3 h-3" />
              <span>Metadata Extraction</span>
            </h3>
            {data.extraction.timing && (
              <div className="text-slate-500 text-xs flex items-center gap-1.5">
                <FontAwesomeIcon icon={faBolt} className="w-3 h-3" />
                <span>{data.extraction.timing.duration_seconds ? `${data.extraction.timing.duration_seconds.toFixed(2)}s` : 'N/A'}</span>
                {data.extraction.timing.provider && (
                  <span className="text-slate-400">({data.extraction.timing.provider})</span>
                )}
              </div>
            )}
          </div>

          {/* Show message if extraction was skipped */}
          {data.extraction.skipped && (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-2 text-xs text-yellow-700">
              <strong>Extractie overgeslagen:</strong> {data.extraction.skipped.reason || 'Onbekende reden'}
              {data.extraction.skipped.doc_type_slug && (
                <div className="mt-1 text-yellow-700/80">
                  Document type: <span className="font-mono">{data.extraction.skipped.doc_type_slug}</span>
                </div>
              )}
              <div className="mt-1 text-yellow-700/60 text-[10px]">
                Voeg fields toe aan dit document type om extractie mogelijk te maken.
              </div>
            </div>
          )}

          {data.extraction.error && (
            <div className="bg-red-50 border border-red-200 rounded p-2 text-xs text-red-600">
              <strong>Error:</strong> {data.extraction.error.substring(0, 200)}...
            </div>
          )}

          {data.extraction.chunkWarnings && data.extraction.chunkWarnings.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="text-yellow-700 text-xs font-semibold">Chunk warnings</div>
                  <div className="text-yellow-700/70 text-[11px]">
                    {data.extraction.chunkWarnings.length} chunk{data.extraction.chunkWarnings.length === 1 ? '' : 's'} had schema fallback, maar zijn niet definitief gefaald.
                  </div>
                </div>
                <span className="text-[10px] text-yellow-700/60 bg-yellow-50 border border-yellow-200 rounded px-2 py-0.5">
                  fallback ok
                </span>
              </div>
              <div className="space-y-1.5">
                {data.extraction.chunkWarnings.map((chunkWarning) => (
                  <CollapsiblePromptBlock
                    key={chunkWarning.path}
                    id={`ext-chunk-warning-${chunkWarning.chunkNum}`}
                    title={`Chunk ${chunkWarning.chunkNum} warning`}
                    content={chunkWarning.content}
                    downloadPath={chunkWarning.path}
                    downloadName={`extraction_warning_chunk_${chunkWarning.chunkNum}.txt`}
                    description="Schema poging faalde, fallback kon doorgaan"
                    isExpanded={expandedSections.has(`ext-chunk-warning-${chunkWarning.chunkNum}`)}
                    onToggle={() => toggleSection(`ext-chunk-warning-${chunkWarning.chunkNum}`)}
                    onDownload={downloadArtifact}
                  />
                ))}
              </div>
            </div>
          )}

          {data.extraction.chunkErrors && data.extraction.chunkErrors.length > 0 && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="text-red-600 text-xs font-semibold">Chunk errors</div>
                  <div className="text-red-600 text-[11px]">
                    {data.extraction.chunkErrors.length} chunk{data.extraction.chunkErrors.length === 1 ? '' : 's'} met extractiefouten gevonden.
                  </div>
                </div>
                <span className="text-[10px] text-red-600 bg-red-50 border border-red-200 rounded px-2 py-0.5">
                  zichtbaar bij status error
                </span>
              </div>
              <div className="space-y-1.5">
                {data.extraction.chunkErrors.map((chunkError) => (
                  <CollapsiblePromptBlock
                    key={chunkError.path}
                    id={`ext-chunk-error-${chunkError.chunkNum}`}
                    title={`Chunk ${chunkError.chunkNum} error`}
                    content={chunkError.content}
                    downloadPath={chunkError.path}
                    downloadName={`extraction_error_chunk_${chunkError.chunkNum}.txt`}
                    description="Schema/fallback fout voor deze chunk"
                    isExpanded={expandedSections.has(`ext-chunk-error-${chunkError.chunkNum}`)}
                    onToggle={() => toggleSection(`ext-chunk-error-${chunkError.chunkNum}`)}
                    onDownload={downloadArtifact}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Show message if extraction failed (no result but has prompt) */}
          {!data.extraction.result && !data.extraction.error && !data.extraction.skipped && data.extraction.prompt && (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-2 text-xs text-yellow-700">
              <strong>Let op:</strong> Extractie is niet gelukt. Er is wel een prompt verzonden, maar geen resultaat ontvangen.
            </div>
          )}

          <div className="space-y-2">
            {renderCollapsibleBlock("ext-prompt", "Prompt", data.extraction.prompt, "llm/extraction_prompt.txt", "extraction_prompt.txt", "Volledige prompt naar LLM")}
            {/* Show Response and compare with Result to detect validation issues */}
            {(() => {
              const responseText = data.extraction.response?.trim() || '';
              const resultJson = data.extraction.result ? JSON.stringify(data.extraction.result, null, 2) : '';
              const responseIsJson = responseText && (responseText.startsWith('{') || responseText.startsWith('['));
              
              // Parse response to compare with result
              let responseParsed = null;
              if (responseIsJson) {
                try {
                  responseParsed = JSON.parse(responseText);
                } catch (e) {
                  // Not valid JSON, show as-is
                }
              }
              
              // Check if response was significantly different from result
              const isDifferent = responseParsed && data.extraction.result && 
                JSON.stringify(responseParsed) !== JSON.stringify(data.extraction.result);
              
              // Show Response if it exists and is different from Result
              if (data.extraction.response && (!responseIsJson || !responseParsed || responseText !== resultJson)) {
                return (
                  <>
                    {isDifferent && (
                      <div className="bg-blue-50 border border-blue-200 rounded p-2 text-xs text-blue-500">
                        <strong>Info:</strong> De LLM response is aangepast na validatie. 
                        Sommige velden zijn mogelijk gefilterd of gecorrigeerd.
                      </div>
                    )}
                    {renderCollapsibleBlock("ext-response", "Response", data.extraction.response, "llm/extraction_response.txt", "extraction_response.txt", isDifferent ? "LLM output (aangepast door validatie)" : "Ruwe LLM output")}
                  </>
                );
              }
              return null;
            })()}
            {data.extraction.result && renderCollapsibleBlock("ext-result", "Result (JSON)", JSON.stringify(data.extraction.result, null, 2), "llm/extraction_result.json", "extraction_result.json", "Gevalidateerd JSON resultaat")}
          </div>
        </div>
      )}
    </div>
  );
}

function TextTab({ document, highlightSources, searchTerm, onHighlightToggle, onSearchChange, downloadArtifact, extractedText, textLoading }: {
  document: Document;
  highlightSources: boolean;
  searchTerm: string;
  onHighlightToggle: () => void;
  onSearchChange: (term: string) => void;
  downloadArtifact: (path: string, filename: string) => void;
  extractedText?: string;
  textLoading?: boolean;
}) {
  const [expandedHighlights, setExpandedHighlights] = useState<Set<string>>(new Set());

  return (
    <div className="p-4 space-y-4">
      <div className="flex flex-col sm:flex-row gap-2">
        <div className="flex-1 relative">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search..."
            className="w-full px-3 py-1.5 pl-8 bg-slate-100 border border-slate-300 rounded text-slate-800 text-sm placeholder-slate-400 focus:ring-1 focus:ring-blue-400"
          />
          <FontAwesomeIcon icon={faSearch} className="absolute left-2.5 top-1/2 transform -translate-y-1/2 text-slate-500 w-3 h-3" />
        </div>
        <div className="flex space-x-2">
          <button
            onClick={onHighlightToggle}
            className={`flex items-center space-x-1 px-2 py-1.5 text-xs rounded ${highlightSources ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
          >
            <FontAwesomeIcon icon={faHighlighter} className="w-3 h-3" />
            <span>Evidence</span>
          </button>
          <button onClick={() => downloadArtifact('text/extracted.txt', 'text.txt')} className="flex items-center space-x-1 px-2 py-1.5 bg-slate-100 text-slate-800 text-xs rounded hover:bg-slate-200">
            <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
          </button>
        </div>
      </div>

      <div className="bg-slate-50 rounded-lg p-3">
        <div className="max-h-64 overflow-y-auto">
          {textLoading ? (
            <div className="flex items-center justify-center py-4">
              <FontAwesomeIcon icon={faSpinner} className="text-blue-600 w-5 h-5 animate-spin" />
            </div>
          ) : (
            <pre className="text-slate-800 text-xs whitespace-pre-wrap">{extractedText || 'No text'}</pre>
          )}
        </div>
      </div>

      {highlightSources && document.metadata_evidence_json && Object.keys(document.metadata_evidence_json).length > 0 && (
        <div className="space-y-2">
          <h3 className="text-slate-800 font-medium text-sm">Evidence</h3>
          {Object.entries(document.metadata_evidence_json).map(([field, spans]) => (
            <div key={field} className="bg-slate-50 rounded overflow-hidden">
              <button
                onClick={() => setExpandedHighlights(prev => { const n = new Set(prev); n.has(field) ? n.delete(field) : n.add(field); return n; })}
                className="w-full flex items-center justify-between p-2 text-left hover:bg-slate-100"
              >
                <span className="text-slate-800 text-xs font-medium">{field}</span>
                <FontAwesomeIcon icon={expandedHighlights.has(field) ? faChevronUp : faChevronDown} className="text-slate-500 w-3 h-3" />
              </button>
              {expandedHighlights.has(field) && Array.isArray(spans) && (
                <div className="px-2 pb-2 space-y-1">
                  {spans.map((span: any, i: number) => (
                    <div key={i} className="bg-blue-50 rounded p-1.5 text-xs">
                      <span className="text-blue-500">Page {span.page}</span>
                      {span.quote && <div className="text-slate-800 mt-1">"{span.quote}"</div>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Forensics Tab Component
function ForensicsTab({ documentId, document, isOpen }: {
  documentId: number;
  document: Document | undefined;
  isOpen: boolean;
}) {
  const [expandedSignalIndex, setExpandedSignalIndex] = useState<number | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);

  const { data: fraudReport, isLoading, error } = useQuery({
    queryKey: ['fraud-analysis', documentId],
    queryFn: () => getFraudAnalysis(documentId),
    enabled: !!documentId && document?.status === 'done',
    staleTime: Infinity, // Never refetch - data is static once analyzed
    gcTime: Infinity, // Keep in cache forever
  });

  // Filter out LOW risk signals for display
  const filteredSignals = fraudReport?.signals?.filter(s => s.risk_level?.toLowerCase() !== 'low') || [];

  // Check if ELA heatmap exists
  // Use a timestamp in the query key to force fresh fetch when modal opens
  const { data: elaHeatmapUrl, error: elaHeatmapError } = useQuery({
    queryKey: ['ela-heatmap', documentId, isOpen], // Include isOpen to force refetch when modal opens
    queryFn: async () => {
      try {
        const blob = await getDocumentArtifact(documentId, 'risk/ela_heatmap.png');
        // Verify blob is valid
        if (!blob || blob.size === 0) {
          return null;
        }
        
        // Always create a PNG blob to ensure correct type
        const pngBlob = new Blob([blob], { type: 'image/png' });
        const url = URL.createObjectURL(pngBlob);
        return url;
      } catch (error: any) {
        // Silently fail for 404 (heatmap doesn't exist yet)
        if (error?.response?.status === 404 || error?.silent) {
          return null;
        }
        console.error('Failed to load ELA heatmap:', error);
        return null;
      }
    },
    enabled: !!documentId && document?.status === 'done' && isOpen,
    staleTime: 0, // Always consider stale to force refetch
    gcTime: 0, // Don't cache - always create fresh object URL
    retry: false, // Don't retry on 404
    refetchOnMount: 'always', // Always refetch when component mounts
  });

  // Track previous URLs to cleanup when they're replaced
  const prevUrlRef = useRef<string | null>(null);
  
  // Cleanup previous object URL when a new one is created
  useEffect(() => {
    if (prevUrlRef.current && prevUrlRef.current !== elaHeatmapUrl) {
      // Previous URL exists and is different, revoke it
      try {
        URL.revokeObjectURL(prevUrlRef.current);
      } catch (e) {
        // URL might already be revoked, ignore
      }
    }
    prevUrlRef.current = elaHeatmapUrl || null;
  }, [elaHeatmapUrl]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Cleanup on component unmount
      if (prevUrlRef.current) {
        try {
          URL.revokeObjectURL(prevUrlRef.current);
        } catch (e) {
          // URL might already be revoked, ignore
        }
      }
    };
  }, []); // Only run on unmount

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'critical': return 'text-red-700 bg-red-100 border-red-200';
      case 'high': return 'text-orange-700 bg-orange-100 border-orange-200';
      case 'medium': return 'text-amber-700 bg-yellow-100 border-yellow-200';
      case 'low': return 'text-green-700 bg-green-100 border-green-200';
      default: return 'text-slate-500 bg-slate-100 border-slate-300';
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'critical': return faCircleExclamation;
      case 'high': return faExclamationTriangle;
      case 'medium': return faBolt;
      case 'low': return faCheck;
      default: return faCircle;
    }
  };

  const getRiskLabel = (risk: string) => {
    switch (risk) {
      case 'critical': return 'Kritiek';
      case 'high': return 'Hoog';
      case 'medium': return 'Gemiddeld';
      case 'low': return 'Laag';
      default: return 'Onbekend';
    }
  };

  if (document?.status !== 'done') {
    return (
      <div className="p-8 text-center">
        <FontAwesomeIcon icon={faSpinner} className="text-slate-400 text-2xl animate-spin mb-2" />
        <p className="text-slate-500 text-sm">Document wordt verwerkt...</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="p-8 text-center">
        <FontAwesomeIcon icon={faSpinner} className="text-slate-400 text-2xl animate-spin mb-2" />
        <p className="text-slate-500 text-sm">Forensische analyse wordt uitgevoerd...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-600 text-sm">Fout bij laden van forensische analyse</p>
        </div>
      </div>
    );
  }

  if (!fraudReport) {
    return (
      <div className="p-8 text-center">
        <p className="text-slate-500 text-sm">Geen forensische data beschikbaar</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
      {/* Risk Score Header - More prominent */}
      <div className={`rounded-lg p-4 border-2 ${getRiskColor(fraudReport.overall_risk)}`}>
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 flex-1 min-w-0">
            <FontAwesomeIcon icon={getRiskIcon(fraudReport.overall_risk)} className="text-2xl shrink-0 mt-0.5" />
            <div className="min-w-0 flex-1">
              <div className="text-lg font-bold mb-1">
                Risico: {Math.round(fraudReport.risk_score)}%
              </div>
              <div className="text-sm font-medium mb-2 opacity-90">
                {getRiskLabel(fraudReport.overall_risk)}
              </div>
              <div className="text-xs opacity-70 flex items-center gap-2 flex-wrap">
                <span>{filteredSignals.length} signaal{filteredSignals.length !== 1 ? 'en' : ''} gedetecteerd</span>
                <span>•</span>
                <span>Geanalyseerd op {new Date(fraudReport.analyzed_at).toLocaleDateString('nl-NL', { day: 'numeric', month: 'long', year: 'numeric' })}</span>
              </div>
            </div>
          </div>
        </div>
        {fraudReport.summary && (
          <div className="mt-3 pt-3 border-t border-current/20">
            <p className="text-xs opacity-90 leading-relaxed">{fraudReport.summary}</p>
          </div>
        )}
      </div>

      {fraudReport.semantic_context?.top_matches?.length ? (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="text-slate-800 font-semibold text-base mb-2">Documentcontext (BERT)</h3>
          <p className="text-blue-700 text-sm mb-3">{fraudReport.semantic_context.summary}</p>
          <div className="flex flex-wrap gap-2">
            {fraudReport.semantic_context.top_matches.map((match) => (
              <span key={match.label} className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                {match.label}: {Math.round(match.confidence * 100)}%
              </span>
            ))}
          </div>
          <p className="text-slate-400 text-xs mt-3">
            BERT is hier context en fallback, geen harde fraudebeslisser. Model: {fraudReport.semantic_context.model_used} · margin {Math.round(fraudReport.semantic_context.margin * 100)}%
          </p>
        </div>
      ) : null}

      {/* Advice Cards */}
      {fraudReport.advice && fraudReport.advice.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-slate-800 font-semibold text-sm flex items-center gap-2 px-1">
            <FontAwesomeIcon icon={faExclamationTriangle} className="text-[#FFC1F3] w-4 h-4" />
            Adviezen ({fraudReport.advice.length})
          </h3>
          {fraudReport.advice.map((card: AdviceCard, i: number) => {
            const priorityColors: Record<string, string> = {
              high: 'border-red-300 bg-red-50 text-red-600',
              medium: 'border-amber-300 bg-amber-50 text-amber-700',
              low: 'border-green-200 bg-green-50 text-green-700',
            };
            return (
              <div key={i} className={`rounded-lg border p-3 ${priorityColors[card.priority] ?? 'border-slate-300 bg-slate-50 text-slate-600'}`}>
                <div className="flex items-start gap-2">
                  <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase font-bold shrink-0 ${priorityColors[card.priority] ?? ''}`}>
                    {card.priority}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="font-medium text-xs mb-1">{card.title}</p>
                    <p className="text-[11px] opacity-80 leading-relaxed">{card.action}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Signals List - Accordion (only one open at a time) */}
      {filteredSignals.length > 0 ? (
        <div className="space-y-2">
          {/* Static header - not clickable */}
          <div className="w-full flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-200">
            <h3 className="text-slate-800 font-semibold text-base flex items-center gap-2">
              <FontAwesomeIcon icon={faShieldAlt} className="text-blue-600 w-5 h-5" />
              Gedetecteerde Signalen
            </h3>
            <div className="flex items-center gap-2">
              <span className="text-slate-500 text-xs bg-slate-50 px-2 py-1 rounded">
                {filteredSignals.length} totaal
              </span>
              {expandedSignalIndex !== null && (
                <span className="text-blue-600 text-xs">
                  1 open
                </span>
              )}
            </div>
          </div>

          {filteredSignals.map((signal, index) => {
            const isExpanded = expandedSignalIndex === index;
            return (
              <div
                key={index}
                className={`rounded-lg border overflow-hidden ${getRiskColor(signal.risk_level)}`}
              >
                <button
                  onClick={() => {
                    // Accordion: only one signal open at a time
                    if (isExpanded) {
                      setExpandedSignalIndex(null); // Close if already open
                    } else {
                      setExpandedSignalIndex(index); // Open this one, closes others automatically
                    }
                  }}
                  className="w-full flex items-start justify-between gap-2 p-2.5 hover:bg-slate-100 transition-colors text-left cursor-pointer"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                      <FontAwesomeIcon icon={getRiskIcon(signal.risk_level)} className="w-3.5 h-3.5 shrink-0" />
                      <span className="font-medium text-xs truncate">{signal.name.replace(/_/g, ' ')}</span>
                      {signal.category && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-50 shrink-0">
                          {signal.category.replace(/_/g, ' ')}
                        </span>
                      )}
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-50 shrink-0">
                        {Math.round(signal.confidence * 100)}%
                      </span>
                    </div>
                    <p className="text-xs opacity-80 line-clamp-2">{signal.description}</p>
                  </div>
                  <FontAwesomeIcon 
                    icon={isExpanded ? faChevronUp : faChevronDown} 
                    className="text-slate-400 w-3 h-3 shrink-0 mt-0.5"
                  />
                </button>

                {isExpanded && (
                  <div className="px-2.5 pb-2.5 space-y-2 border-t border-current/20">
                    {/* Confidence explanation for ELA signals */}
                    {signal.name === 'ela_manipulation_detected' && (
                      <div className="pt-2 pb-1">
                        <div className="text-[10px] opacity-60 mb-1">Zekerheid ({Math.round(signal.confidence * 100)}%):</div>
                        <div className="text-[10px] opacity-80 leading-relaxed">
                          Dit percentage geeft aan hoe zeker het systeem is dat er manipulatie is gedetecteerd. Het wordt berekend op basis van:
                          <ul className="list-disc list-inside mt-1 space-y-0.5 ml-1">
                            <li>Compressieverschillen (std_error): standaarddeviatie van de verschillen</li>
                            <li>Lokale verschillen (max_error): maximale verschil in specifieke gebieden</li>
                            <li>Visuele verschillen: percentage heldere pixels in de heatmap</li>
                          </ul>
                          <div className="mt-1.5 pt-1.5 border-t border-current/20">
                            <strong>Interpretatie:</strong> {signal.confidence >= 0.8 ? 'Zeer sterk signaal (80-95%)' : signal.confidence >= 0.6 ? 'Redelijk sterk signaal (60-79%)' : 'Subtiel signaal (50-59%)'} - er zijn duidelijke compressieverschillen zichtbaar.
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Evidence */}
                    {signal.evidence.length > 0 && (
                      <div className="pt-2">
                        <div className="text-[10px] opacity-60 mb-1">Bewijs:</div>
                        <div className="flex flex-wrap gap-1">
                          {signal.evidence.map((e, i) => (
                            <span key={i} className="text-[10px] bg-slate-50 px-1.5 py-0.5 rounded">
                              {e}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {signal.recommendation && (
                      <div className="pt-2">
                        <div className="text-[10px] opacity-60 mb-1">Aanbeveling:</div>
                        <div className="text-[10px] opacity-80 leading-relaxed bg-slate-50 rounded p-2">
                          {signal.recommendation}
                        </div>
                      </div>
                    )}

                    {/* Technical Details - Auto-opened when signal is expanded */}
                    {Object.keys(signal.details).length > 0 && (
                      <div className="pt-2">
                        <div className="text-[10px] opacity-60 mb-1.5 font-medium">Technische details</div>
                        <pre className="text-[10px] bg-slate-50 p-1.5 rounded overflow-x-auto max-h-32 overflow-y-auto">
                          {JSON.stringify(signal.details, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-4 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-green-700 font-medium text-sm">✓ Geen verdachte signalen</p>
          <p className="text-slate-400 text-xs mt-0.5">Dit document lijkt veilig</p>
        </div>
      )}

      {/* ELA Heatmap - Collapsible */}
      <div className="bg-slate-50 rounded-lg border border-slate-200 overflow-hidden">
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="w-full flex items-center justify-between p-2.5 hover:bg-slate-50 transition-colors cursor-pointer"
        >
          <h4 className="text-slate-500 text-sm font-medium flex items-center gap-2">
            <FontAwesomeIcon icon={faImage} className="text-purple-600 w-4 h-4" />
            ELA Heatmap
          </h4>
          <FontAwesomeIcon 
            icon={showHeatmap ? faChevronUp : faChevronDown} 
            className="text-slate-500 w-3 h-3"
          />
        </button>
        
        {showHeatmap && (
          <div className="px-2.5 pb-2.5">
            {elaHeatmapUrl ? (
              <div className="bg-slate-50 rounded-lg p-2 border border-slate-200">
                <img 
                  key={`ela-heatmap-${documentId}-${elaHeatmapUrl}`}
                  src={elaHeatmapUrl || undefined} 
                  alt="ELA Heatmap" 
                  className="w-full h-auto rounded border border-slate-200 max-h-64 object-contain mx-auto"
                  onError={(e) => {
                    console.error('Failed to load ELA heatmap image', {
                      src: elaHeatmapUrl,
                      error: e,
                      currentTarget: e.currentTarget,
                      documentId
                    });
                    const container = e.currentTarget.parentElement;
                    if (container) {
                      e.currentTarget.style.display = 'none';
                      const errorMsg = window.document.createElement('p');
                      errorMsg.className = 'text-red-600 text-xs mt-2';
                      errorMsg.textContent = 'Kon heatmap niet laden. Probeer de pagina te verversen.';
                      if (!container.querySelector('.error-msg')) {
                        errorMsg.classList.add('error-msg');
                        container.appendChild(errorMsg);
                      }
                    }
                  }}
                  onLoad={() => {
                    console.log('ELA heatmap image loaded successfully', { documentId, url: elaHeatmapUrl });
                  }}
                />
                <p className="text-slate-400 text-xs mt-2">
                  Compressieverschillen. Heldere gebieden = mogelijke manipulatie.
                  {fraudReport?.signals?.some(s => s.name === 'ela_manipulation_detected') && (
                    <span className="block mt-1 text-yellow-700 text-[10px]">
                      ⚠️ Manipulatie gedetecteerd
                    </span>
                  )}
                </p>
              </div>
            ) : (
              <p className="text-slate-400 text-xs py-2">
                Geen heatmap beschikbaar
              </p>
            )}
          </div>
        )}
      </div>

      {/* Analysis Methods Info - Always visible */}
      <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
        <h4 className="text-slate-500 text-sm font-medium mb-2">Analyse Methodes</h4>
        <div className="grid grid-cols-2 gap-1.5 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-blue-100 border border-blue-200 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faFilePdf} className="w-2.5 h-2.5 text-blue-600" />
            </div>
            <span className="text-slate-500 text-[10px]">PDF Metadata</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-purple-100 border border-purple-200 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faImage} className="w-2.5 h-2.5 text-purple-600" />
            </div>
            <span className="text-slate-500 text-[10px]">ELA Analyse</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-amber-100 border border-amber-200 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-2.5 h-2.5 text-amber-400" />
            </div>
            <span className="text-slate-500 text-[10px]">Unicode</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-green-100 border border-green-200 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faRedo} className="w-2.5 h-2.5 text-green-600" />
            </div>
            <span className="text-slate-500 text-[10px]">Herhaling</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-cyan-100 border border-cyan-200 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faBullseye} className="w-2.5 h-2.5 text-cyan-400" />
            </div>
            <span className="text-slate-500 text-[10px]">Confidence</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-pink-100 border border-pink-200 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faCog} className="w-2.5 h-2.5 text-pink-400" />
            </div>
            <span className="text-slate-500 text-[10px]">Software</span>
          </div>
        </div>
      </div>
    </div>
  );
}
