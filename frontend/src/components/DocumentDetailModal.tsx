'use client';

import { useState, useEffect, useRef } from 'react';
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
  RiskSignal, subscribeToDocumentEvents, getFraudAnalysis, FraudReport, FraudSignal
} from '@/lib/api';
import { PDFViewerWithHighlights } from './PDFViewerWithHighlights';

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
  const hasEvidence = evidence && Object.keys(evidence).length > 0;

  const handleDownload = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDownload();
  };

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-[60] flex items-center justify-center p-0 sm:p-4" onClick={onClose}>
      <div className="glass-card w-full h-full sm:w-[85vw] sm:h-[85vh] max-w-[85vw] max-h-[85vh] flex flex-col rounded-none sm:rounded-xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center space-x-3 min-w-0">
            <FontAwesomeIcon 
              icon={isPDF ? faFilePdf : isImage ? faImage : faFile} 
              className="text-white/70 w-5 h-5 flex-shrink-0" 
            />
            <div className="min-w-0">
              <h2 className="text-white text-lg font-semibold truncate">
                {filename}
              </h2>
              <p className="text-white/60 text-xs">
                Origineel bestand
                {hasEvidence && <span className="text-blue-400 ml-2">• Met highlights</span>}
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
              className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded-lg cursor-pointer"
            >
              <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Viewer Content */}
        <div className="flex-1 overflow-hidden bg-black/30">
          {isPDF && hasEvidence ? (
            // Use custom PDF viewer with highlights when we have evidence
            <PDFViewerWithHighlights 
              url={documentUrl} 
              evidence={evidence}
            />
          ) : isPDF ? (
            // Fall back to iframe for PDFs without evidence
            <div className="w-full h-full p-2 sm:p-4">
              <iframe
                src={`${documentUrl}#toolbar=1&navpanes=1&scrollbar=1&view=FitH`}
                className="w-full h-full min-h-[80vh] rounded-lg border border-white/10 bg-white"
                title={filename}
                style={{ colorScheme: 'light' }}
              />
            </div>
          ) : isImage ? (
            <div className="w-full h-full p-2 sm:p-4 flex items-center justify-center">
              <img
                src={documentUrl}
                alt={filename}
                className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
              />
            </div>
          ) : isWord ? (
            <div className="text-center text-white/60 w-full h-full flex flex-col items-center justify-center">
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
            <div className="text-center text-white/60 w-full h-full flex flex-col items-center justify-center">
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

interface DocumentDetailModalProps {
  documentId: number | null;
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'overview' | 'text' | 'metadata' | 'llm' | 'forensics';

// Build direct API URL for artifacts (bypasses blob URL issues)
const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export function DocumentDetailModal({ documentId, isOpen, onClose }: DocumentDetailModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [highlightSources, setHighlightSources] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedExamples, setExpandedExamples] = useState<Set<number>>(new Set());
  const [showDocumentViewer, setShowDocumentViewer] = useState(false);
  const [documentViewerPath, setDocumentViewerPath] = useState<string | null>(null);
  const [examplesSidebar, setExamplesSidebar] = useState<{ signalIndex: number; examples: any } | null>(null);
  const queryClient = useQueryClient();

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
      setHighlightSources(false);
      setSearchTerm('');
    }
  }, [isOpen]);

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
      case 'done': return 'text-green-400';
      case 'processing': return 'text-blue-400';
      case 'error': return 'text-red-400';
      default: return 'text-yellow-400';
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
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-0 sm:p-4 overflow-hidden"
          onClick={(e) => {
            // Close sidebar if clicking outside the modal
            if (examplesSidebar && e.target === e.currentTarget) {
              setExamplesSidebar(null);
            }
          }}
        >
          <div className="relative w-full max-w-4xl h-full sm:h-auto sm:max-h-[90vh] m-0 sm:m-4 overflow-hidden sm:overflow-visible">
            <div className={`glass-card overflow-hidden flex flex-col w-full h-full sm:h-auto transition-all duration-300 rounded-none sm:rounded-xl`}>
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center space-x-3 min-w-0">
            <FontAwesomeIcon icon={faFile} className="text-white/70 w-5 h-5 flex-shrink-0" />
            <div className="min-w-0">
              <h2 className="text-white text-lg font-semibold truncate">
                {document?.original_filename || 'Loading...'}
              </h2>
              <div className="flex items-center gap-2 text-xs">
                <span className="text-white/60">Document #{documentId}</span>
                {/* Processing Status Indicator */}
                {(document?.status === 'processing' || document?.status === 'queued') && (
                  <span className="flex items-center gap-1.5 px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded-full animate-pulse">
                    <FontAwesomeIcon icon={faSpinner} className="w-3 h-3 animate-spin" />
                    <span>{document.stage || 'Verwerken...'}</span>
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
                className="flex items-center space-x-2 px-3 py-1.5 bg-purple-600 text-white text-sm rounded-lg hover:bg-purple-700 cursor-pointer"
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
                  ? 'bg-gray-600'
                  : 'bg-blue-600 hover:bg-blue-700'
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
            <button onClick={onClose} className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded-lg">
              <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
            </button>
          </div>
        </div>

        {isLoading && (
          <div className="p-8 text-center">
            <FontAwesomeIcon icon={faSpinner} className="text-white/40 text-3xl mb-4 animate-spin" />
            <p className="text-white/60">Loading...</p>
          </div>
        )}

        {document && !isLoading && (
          <>
            {/* Processing Overlay */}
            {(document.status === 'processing' || document.status === 'queued') && (
              <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border-b border-blue-500/20 px-4 py-3">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <FontAwesomeIcon icon={faSpinner} className="w-5 h-5 text-blue-400 animate-spin" />
                    <div className="absolute inset-0 bg-blue-400/20 rounded-full animate-ping" />
                  </div>
                  <div className="flex-1">
                    <div className="text-white text-sm font-medium">
                      {document.stage || 'Document wordt verwerkt...'}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500"
                          style={{ width: `${document.progress || 0}%` }}
                        />
                      </div>
                      <span className="text-white/60 text-xs">{document.progress || 0}%</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tabs */}
            <div className="flex border-b border-white/10 flex-shrink-0 overflow-x-auto">
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
                      isDisabled ? 'text-white/30 cursor-not-allowed' :
                      activeTab === tab.id ? 'text-blue-400 border-b-2 border-blue-400' : 'text-white/60 hover:text-white'
                    }`}
                  >
                    <FontAwesomeIcon icon={tab.icon} className="w-3 h-3 sm:w-3.5 sm:h-3.5" />
                    <span>{tab.label}</span>
                    {isDisabled && <FontAwesomeIcon icon={faSpinner} className="w-2.5 h-2.5 animate-spin ml-1" />}
                  </button>
                );
              })}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto overflow-x-hidden min-h-0">
              {activeTab === 'overview' && <OverviewTab document={document} documentId={documentId} formatFileSize={formatFileSize} formatDate={formatDate} formatDateTime={formatDateTime} formatStage={formatStage} getStatusColor={getStatusColor} getRiskColor={getRiskColor} onShowExamples={(signalIndex, examples) => setExamplesSidebar({ signalIndex, examples })} />}
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
              <div className="fixed sm:absolute inset-x-0 bottom-0 sm:inset-auto sm:top-0 sm:left-full sm:ml-0 glass-card w-full sm:w-96 max-h-[70vh] sm:max-h-none flex-shrink-0 border-t sm:border-t-0 sm:border-l border-white/10 flex flex-col rounded-t-xl sm:rounded-none sm:rounded-r-xl sm:h-full shadow-2xl animate-slide-in-up sm:animate-slide-in-right overflow-hidden z-[60]">
                <div className="flex items-center justify-between p-4 border-b border-white/10 flex-shrink-0">
                  <h3 className="text-white font-semibold text-sm">Voorbeeldteksten</h3>
                  <button
                    onClick={() => setExamplesSidebar(null)}
                    className="p-1.5 text-white/60 hover:text-white hover:bg-white/10 rounded-lg transition-colors cursor-pointer"
                  >
                    <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
                  </button>
                </div>
                <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-4 min-h-0">
                  {examplesSidebar.examples.unicode_examples && examplesSidebar.examples.unicode_examples.length > 0 && (
                    <div>
                      <div className="text-xs font-semibold text-white/80 mb-2">
                        Unicode tekens gevonden:
                      </div>
                      <div className="space-y-2">
                        {examplesSidebar.examples.unicode_examples.map((example: string, idx: number) => (
                          <div 
                            key={idx}
                            className="bg-black/30 rounded-lg p-3 text-xs font-mono text-white/90 border border-white/10 break-all leading-relaxed"
                          >
                            <div className="whitespace-pre-wrap">{formatExampleText(example, true)}</div>
                          </div>
                        ))}
                      </div>
                      <div className="text-xs text-white/50 mt-2">
                        Deze tekens zijn niet standaard ASCII en kunnen wijzen op manipulatie.
                      </div>
                    </div>
                  )}
                  
                  {examplesSidebar.examples.repetition_examples && examplesSidebar.examples.repetition_examples.length > 0 && (
                    <div className="mt-6">
                      <div className="flex items-center gap-2 mb-3">
                        <div className="w-1 h-4 bg-gradient-to-b from-orange-400 to-red-400 rounded-full"></div>
                        <h4 className="text-sm font-bold text-white">Herhalende tekst patronen</h4>
                      </div>
                      <div className="space-y-3">
                        {examplesSidebar.examples.repetition_examples.map((example: string, idx: number) => {
                          const formattedText = formatExampleText(example, false);
                          return (
                            <div 
                              key={idx}
                              className="bg-gradient-to-br from-orange-500/10 to-red-500/10 rounded-lg p-4 text-sm text-white/95 border border-orange-500/20 break-words shadow-lg"
                            >
                              <div className="flex items-start gap-3">
                                <span className="text-orange-400 font-bold text-xs mt-0.5 flex-shrink-0">#{idx + 1}</span>
                                <div className="flex-1">
                                  <div className="leading-relaxed text-white/95">{formattedText}</div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="mt-4 p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                        <div className="text-xs text-white/70 leading-relaxed">
                          <span className="font-semibold text-orange-300">Waarom is dit verdacht?</span>
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
          documentUrl={`${API_BASE_URL}/api/documents/${documentId}/artifact?path=${encodeURIComponent(documentViewerPath)}`}
          filename={document?.original_filename || ''}
          mimeType={document?.mime_type || ''}
          onDownload={() => downloadArtifact(documentViewerPath, document?.original_filename || 'document')}
          evidence={document?.metadata_evidence_json as Record<string, any[]> | undefined}
        />
      )}
    </>
  );
}

function OverviewTab({ document, formatFileSize, formatDate, formatDateTime, formatStage, getStatusColor, getRiskColor, onShowExamples, documentId }: {
  document: Document;
  formatFileSize: (bytes: number) => string;
  formatDate: (date: string) => string;
  formatDateTime: (date: string) => string;
  formatStage: (stage: string | undefined) => string;
  getStatusColor: (status: string) => string;
  getRiskColor: (score: number) => string;
  onShowExamples: (signalIndex: number, examples: any) => void;
  documentId?: number;
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
        <div className="bg-blue-500/10 rounded-lg p-3 sm:p-4 border border-blue-500/20">
          <h3 className="text-white font-medium mb-2 sm:mb-3 flex items-center space-x-2 text-sm sm:text-base">
            <FontAwesomeIcon icon={faFile} className="text-blue-400 w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span>File Info</span>
          </h3>
          <div className="space-y-1.5 sm:space-y-2 text-xs sm:text-sm">
            <div className="flex justify-between"><span className="text-white/60">Size:</span><span className="text-white">{formatFileSize(document.size_bytes)}</span></div>
            <div className="flex justify-between"><span className="text-white/60">Type:</span><span className="text-white text-xs">{document.mime_type}</span></div>
            <div className="flex justify-between"><span className="text-white/60">SHA256:</span><span className="text-white font-mono text-xs">{document.sha256?.substring(0, 12)}...</span></div>
            <div className="flex justify-between"><span className="text-white/60">Uploaded:</span><span className="text-white">{formatDateTime(document.created_at)}</span></div>
          </div>
        </div>

        {/* Status */}
        <div className={`rounded-lg p-4 border ${
          document.status === 'processing' || document.status === 'queued'
            ? 'bg-blue-500/10 border-blue-500/20'
            : document.status === 'error'
            ? 'bg-red-500/10 border-red-500/20'
            : 'bg-green-500/10 border-green-500/20'
        }`}>
          <h3 className="text-white font-medium mb-3 flex items-center space-x-2">
            {(document.status === 'processing' || document.status === 'queued') ? (
              <FontAwesomeIcon icon={faSpinner} className="text-blue-400 w-4 h-4 animate-spin" />
            ) : document.status === 'error' ? (
              <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-400 w-4 h-4" />
            ) : (
              <FontAwesomeIcon icon={faCheck} className="text-green-400 w-4 h-4" />
            )}
            <span>Status</span>
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-white/60">Status:</span>
              <span className={`capitalize font-medium px-2 py-0.5 rounded text-xs ${getStatusColor(document.status)}`}>{document.status}</span>
            </div>
            {document.stage && <div className="flex justify-between"><span className="text-white/60">Stage:</span><span className="text-white text-sm">{formatStage(document.stage)}</span></div>}
            {(document.status === 'processing' || document.status === 'queued') && document.progress !== undefined && (
              <div className="mt-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-white/60">Progress</span>
                  <span className="text-white">{document.progress}%</span>
                </div>
                <div className="w-full bg-white/20 rounded-full h-2">
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
            <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
              <h3 className="text-white font-medium mb-3 flex items-center space-x-2">
                <FontAwesomeIcon icon={faEye} className="text-purple-400 w-4 h-4" />
                <span>Classification</span>
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-white/60">Type:</span>
                  <span className="text-white font-medium px-2 py-0.5 bg-purple-500/20 rounded">{formatDocumentTypeName(document.doc_type_slug)}</span>
                </div>
                {document.doc_type_confidence && (
                  <div className="flex justify-between"><span className="text-white/60">Confidence:</span><span className="text-white">{Math.round(document.doc_type_confidence * 100)}%</span></div>
                )}
              </div>
              {/* Show explanation if document is "unknown" and there's a rejection reason */}
              {document.doc_type_slug === 'unknown' && document.metadata_validation_json?.classification_scores && (
                ((document.metadata_validation_json.classification_scores as any)?.naive_bayes?.rejection_reason || 
                 (document.metadata_validation_json.classification_scores as any)?.bert?.rejection_reason) && (
                  <div className="mt-3 pt-3 border-t border-white/10">
                    <div className="bg-green-500/10 border border-green-500/20 rounded p-2">
                      <div className="text-green-300 text-[10px] italic">
                        ✓ Dit is correct gedrag - het systeem voorkomt verkeerde classificaties door deze regel te respecteren
                      </div>
                    </div>
                  </div>
                )
              )}
              
              {/* Classification Scores - Always show if available, including failures */}
              {document.metadata_validation_json?.classification_scores && (
                <div className="mt-3 pt-3 border-t border-white/10">
                  <div className="text-white/50 text-xs mb-1.5">Classifier Scores:</div>
                  <div className="grid grid-cols-2 gap-2">
                    {(document.metadata_validation_json.classification_scores as any)?.naive_bayes && (
                      (document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'failed' ||
                      (document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'no_result' ? (
                        <div className="bg-purple-500/10 border border-purple-500/20 rounded px-2 py-1.5 opacity-60">
                          <div className="text-purple-300 text-[10px] font-medium">NB: {(document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'failed' ? 'Fout' : 'Geen model'}</div>
                          {/* Show all scores if available - filter out 0% */}
                          {(document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores && (
                            <div className="mt-1 pt-1 border-t border-purple-500/20">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries((document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores as Record<string, number>)
                                  .sort(([, a], [, b]) => b - a)
                                  .filter(([, score]) => Math.round(score * 100) > 0)
                                  .map(([label, score]) => (
                                    <span key={label} className="text-[9px] bg-purple-500/20 px-1.5 py-0.5 rounded text-white/70">
                                      {label}: {Math.round(score * 100)}%
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'rejected' ? (
                        <div className="bg-purple-500/10 border border-purple-500/20 rounded px-2 py-1.5 opacity-75">
                          <div className="flex items-center justify-between">
                            <span className="text-purple-300 text-[10px] font-medium">NB:</span>
                            <span className="text-white/80 text-[10px]">{(document.metadata_validation_json.classification_scores as any).naive_bayes.label}</span>
                            <span className="text-purple-400 text-[10px]">{Math.round((document.metadata_validation_json.classification_scores as any).naive_bayes.confidence * 100)}%</span>
                          </div>
                          <div className="text-red-400 text-[9px]">❌ {(document.metadata_validation_json.classification_scores as any).naive_bayes.rejection_reason || 'Afgewezen'}</div>
                          {/* Show all scores if available - filter out 0% */}
                          {(document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores && (
                            <div className="mt-1 pt-1 border-t border-purple-500/20">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries((document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores as Record<string, number>)
                                  .sort(([, a], [, b]) => b - a)
                                  .filter(([, score]) => Math.round(score * 100) > 0)
                                  .map(([label, score]) => (
                                    <span key={label} className="text-[9px] bg-purple-500/20 px-1.5 py-0.5 rounded text-white/70">
                                      {label}: {Math.round(score * 100)}%
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (document.metadata_validation_json.classification_scores as any).naive_bayes.status === 'below_threshold' ? (
                        <div className="bg-purple-500/10 border border-purple-500/20 rounded px-2 py-1.5 opacity-75">
                          <div className="flex items-center justify-between">
                            <span className="text-purple-300 text-[10px] font-medium">NB:</span>
                            <span className="text-white/80 text-[10px]">{(document.metadata_validation_json.classification_scores as any).naive_bayes.label}</span>
                            <span className="text-purple-400 text-[10px]">{Math.round((document.metadata_validation_json.classification_scores as any).naive_bayes.confidence * 100)}%</span>
                          </div>
                          <div className="text-yellow-400 text-[9px]">⚠️ Onder threshold ({Math.round(((document.metadata_validation_json.classification_scores as any).naive_bayes.threshold || 0) * 100)}%)</div>
                          {/* Show all scores if available - filter out 0% */}
                          {(document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores && (
                            <div className="mt-1 pt-1 border-t border-purple-500/20">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries((document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores as Record<string, number>)
                                  .sort(([, a], [, b]) => b - a)
                                  .filter(([, score]) => Math.round(score * 100) > 0)
                                  .map(([label, score]) => (
                                    <span key={label} className="text-[9px] bg-purple-500/20 px-1.5 py-0.5 rounded text-white/70">
                                      {label}: {Math.round(score * 100)}%
                                    </span>
                                  ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="bg-purple-500/10 border border-purple-500/20 rounded px-2 py-1.5">
                          <div className="flex items-center justify-between">
                            <span className="text-purple-300 text-[10px] font-medium">NB:</span>
                            <span className="text-white text-[10px]">{(document.metadata_validation_json.classification_scores as any).naive_bayes.label}</span>
                            <span className="text-purple-400 text-[10px]">{Math.round((document.metadata_validation_json.classification_scores as any).naive_bayes.confidence * 100)}%</span>
                          </div>
                          {/* Show all scores if available - filter out 0% */}
                          {(document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores && (
                            <div className="mt-1 pt-1 border-t border-purple-500/20">
                              <div className="flex flex-wrap gap-1">
                                {Object.entries((document.metadata_validation_json.classification_scores as any).naive_bayes.all_scores as Record<string, number>)
                                  .sort(([, a], [, b]) => b - a)
                                  .filter(([, score]) => Math.round(score * 100) > 0)
                                  .map(([label, score]) => (
                                    <span key={label} className="text-[9px] bg-purple-500/20 px-1.5 py-0.5 rounded text-white/70">
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
                        <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2 opacity-60">
                          <div className="text-blue-300 text-xs font-medium mb-0.5">BERT</div>
                          <div className="text-white/60 text-xs">
                            {(document.metadata_validation_json.classification_scores as any).bert.status === 'failed' 
                              ? `Fout: ${(document.metadata_validation_json.classification_scores as any).bert.error?.substring(0, 50) || 'Onbekende fout'}...`
                              : (document.metadata_validation_json.classification_scores as any).bert.reason || 'Geen resultaat'}
                          </div>
                        </div>
                      ) : (document.metadata_validation_json.classification_scores as any).bert.status === 'rejected' ? (
                        <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2 opacity-75">
                          <div className="text-blue-300 text-xs font-medium mb-0.5">BERT</div>
                          <div className="text-white/80 text-xs font-medium">
                            {(document.metadata_validation_json.classification_scores as any).bert.label}
                          </div>
                          <div className="text-blue-400 text-[10px] mt-0.5">
                            {Math.round((document.metadata_validation_json.classification_scores as any).bert.confidence * 100)}% confidence
                          </div>
                          <div className="text-red-400 text-[10px] mt-1">
                            ❌ Afgewezen: {(document.metadata_validation_json.classification_scores as any).bert.rejection_reason || 'Onbekende reden'}
                          </div>
                        </div>
                      ) : (
                        <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2">
                          <div className="text-blue-300 text-xs font-medium mb-0.5">BERT</div>
                          <div className="text-white text-xs">
                            {(document.metadata_validation_json.classification_scores as any).bert.label}
                          </div>
                          <div className="text-blue-400 text-[10px] mt-0.5">
                            {Math.round((document.metadata_validation_json.classification_scores as any).bert.confidence * 100)}% confidence
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
              
              {document.doc_type_rationale ? (
                <div className="mt-3 pt-3 border-t border-white/10">
                  <div className="text-white/50 text-xs mb-1.5">Classificatie methode:</div>
                  <div className="flex flex-wrap gap-1.5 mb-2">
                    {document.doc_type_rationale.includes('STRONG keyword match') || document.doc_type_rationale.includes('STRONG') ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-green-500/20 border border-green-500/30 text-green-200 text-[10px]">
                        ✅ 100% Keyword match
                      </span>
                    ) : document.doc_type_rationale.includes('Deterministic') ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-yellow-500/20 border border-yellow-500/30 text-yellow-200 text-[10px]">
                        ⚠️ Keyword/regex match
                      </span>
                    ) : null}
                    {document.doc_type_rationale.includes('Local classifier') || document.doc_type_rationale.includes('NAIVE_BAYES') || document.doc_type_rationale.includes('BERT') ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-500/20 border border-blue-500/30 text-blue-200 text-[10px]">
                        🤖 Getraind model
                      </span>
                    ) : null}
                    {document.doc_type_rationale.includes('LLM') && !document.doc_type_rationale.includes('STRONG') ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-purple-500/20 border border-purple-500/30 text-purple-200 text-[10px]">
                        🧠 AI classificatie
                      </span>
                    ) : null}
                  </div>
                  {/* Show matched keywords for strong deterministic matches */}
                  {document.doc_type_rationale.includes('STRONG keyword match') && document.doc_type_rationale.includes('matched keywords:') && (
                    <div className="mt-2 bg-green-500/10 border border-green-500/20 rounded p-2">
                      <div className="text-green-300 text-xs font-medium mb-1">Gematchte keywords:</div>
                      <div className="flex flex-wrap gap-1">
                        {document.doc_type_rationale
                          .split('matched keywords:')[1]
                          ?.split('|')[0]
                          ?.split(',')
                          ?.map((kw: string) => kw.trim())
                          ?.filter(Boolean)
                          ?.map((keyword: string, i: number) => (
                            <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-green-500/20 border border-green-500/30 text-green-200 text-[10px]">
                              {keyword}
                            </span>
                          ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Skip Marker Info */}
                  {document.skip_marker_used && (
                    <div className="mt-2 bg-cyan-500/10 border border-cyan-500/20 rounded p-2">
                      <div className="text-cyan-300 text-xs font-medium mb-1 flex items-center gap-1.5">
                        <span>✂️</span>
                        <span>Skip Marker Toegepast</span>
                      </div>
                      <div className="text-white/80 text-[11px] font-mono bg-black/20 rounded px-2 py-1 break-all">
                        {document.skip_marker_used}
                      </div>
                      {document.skip_marker_position != null && (
                        <div className="text-cyan-400/70 text-[10px] mt-1">
                          Tekst afgekapt na positie {document.skip_marker_position.toLocaleString()}
                        </div>
                      )}
                    </div>
                  )}
                  
                  {document.doc_type_rationale.includes('Deterministic') && (
                    <div className="text-[10px] text-white/50 italic bg-yellow-500/5 border border-yellow-500/20 rounded p-2 mt-2">
                      ⚠️ <strong>Waarom onjuist?</strong> Deterministic matching (keywords/regex) heeft voorrang boven het getrainde model. Als je commitment agreement als bankafschrift wordt herkend, controleer de classification_hints van "bankafschrift" in document types - deze bevatten waarschijnlijk keywords die ook in commitment agreements voorkomen.
                    </div>
                  )}
                </div>
              ) : document.doc_type_slug === 'unknown' ? (
                <div className="mt-3 pt-3 border-t border-white/10">
                  <div className="text-white/50 text-xs mb-1.5">Classificatie methode:</div>
                  <div className="bg-gray-500/10 border border-gray-500/20 rounded p-2">
                    <div className="text-gray-300 text-[10px]">
                      Voor Onbekend document type wordt geen field matching uitgevoerd
                    </div>
                  </div>
                </div>
              ) : null}
              
              {/* Skip Marker Info - Show even if no rationale */}
              {!document.doc_type_rationale && document.skip_marker_used && (
                <div className="mt-3 pt-3 border-t border-white/10">
                  <div className="bg-cyan-500/10 border border-cyan-500/20 rounded p-2">
                    <div className="text-cyan-300 text-xs font-medium mb-1 flex items-center gap-1.5">
                      <span>✂️</span>
                      <span>Skip Marker Toegepast</span>
                    </div>
                    <div className="text-white/80 text-[11px] font-mono bg-black/20 rounded px-2 py-1 break-all">
                      {document.skip_marker_used}
                    </div>
                    {document.skip_marker_position != null && (
                      <div className="text-cyan-400/70 text-[10px] mt-1">
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
            <div className="bg-red-500/10 rounded-lg p-4 border border-red-500/20">
              <h3 className="text-white font-medium mb-3 flex items-center space-x-2">
                <FontAwesomeIcon icon={faShieldAlt} className="text-red-400 w-4 h-4" />
                <span>Risk & Forensics</span>
              </h3>
              {document.risk_score !== null && document.risk_score !== undefined ? (
                <div className="flex items-center space-x-3 mb-3">
                  <span className={`text-2xl font-bold ${(document.risk_score ?? 0) >= 70 ? 'text-red-400' : (document.risk_score ?? 0) >= 40 ? 'text-yellow-400' : 'text-green-400'}`}>
                    {document.risk_score ?? 0}
                  </span>
                  <div className="flex-1 bg-white/20 rounded-full h-2">
                    <div className={`h-2 rounded-full ${getRiskColor(document.risk_score ?? 0)}`} style={{ width: `${document.risk_score ?? 0}%` }} />
                  </div>
                </div>
              ) : (
                <div className="mb-3 text-white/60 text-xs">
                  Risk score niet beschikbaar
                </div>
              )}
              
              {/* Forensics Signals */}
              {fraudReport && fraudReport.signals.length > 0 && (() => {
                const filteredSignals = fraudReport.signals.filter(s => s.risk_level?.toLowerCase() !== 'low');
                if (filteredSignals.length === 0) return null;
                // Ensure we show all filtered signals (up to 3 in overview)
                // Filter out any signals without a name to avoid rendering issues
                const signalsToShow = filteredSignals
                  .filter(s => s.name) // Only show signals with a name
                  .slice(0, 3);
                return (
                  <div className="mb-3 pb-3 border-b border-white/10">
                    <div className="text-xs text-white/60 mb-2">
                      Forensics: {filteredSignals.length} signaal{filteredSignals.length !== 1 ? 'en' : ''} gedetecteerd
                    </div>
                    <div className="space-y-1.5">
                      {signalsToShow.map((signal, i) => (
                      <div key={`signal-${signal.name || 'unknown'}-${i}-${documentId || 'unknown'}`} className="flex items-start gap-2 text-xs">
                        <FontAwesomeIcon 
                          icon={signal.risk_level === 'critical' || signal.risk_level === 'high' ? faExclamationTriangle : faInfoCircle} 
                          className={`w-3 h-3 mt-0.5 shrink-0 ${
                            signal.risk_level === 'critical' ? 'text-red-400' :
                            signal.risk_level === 'high' ? 'text-orange-400' :
                            signal.risk_level === 'medium' ? 'text-yellow-400' : 'text-green-400'
                          }`}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-white font-medium truncate">{signal.name?.replace(/_/g, ' ') || 'Onbekend signaal'}</div>
                          <div className="text-white/70 text-[10px] line-clamp-1">{signal.description || 'Geen beschrijving'}</div>
                        </div>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/20 shrink-0">
                          {Math.round((signal.confidence || 0) * 100)}%
                        </span>
                      </div>
                      ))}
                      {filteredSignals.length > signalsToShow.length && (
                        <div className="text-[10px] text-white/50 pt-1">
                          +{filteredSignals.length - signalsToShow.length} meer (zie Forensics tab)
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}
              
              {(document.risk_signals_json?.length ?? 0) > 0 ? (
                <div className="space-y-2 mt-3">
                  <div className="text-xs text-white/60 mb-2">
                    {document.risk_signals_json?.length || 0} fraude detectie{(document.risk_signals_json?.length || 0) !== 1 ? 's' : ''} gevonden:
                  </div>
                  {document.risk_signals_json?.map((signal: RiskSignal, i: number) => {
                    // Translate risk signal codes to user-friendly explanations
                    const getSignalExplanation = (code: string, message: string, evidence: string) => {
                      if (code === 'FILE_METADATA_PROGRAMMATIC') {
                        return {
                          title: 'PDF gemaakt met programmatische tool',
                          description: 'Dit PDF-bestand is gemaakt met een programmatische library (zoals FPDF, TCPDF, of Python). Dit kan legitiem zijn, maar wordt ook vaak gebruikt om documenten te manipuleren of vervalsen.',
                          recommendation: 'Controleer de inhoud extra goed en vergelijk met het origineel indien mogelijk.'
                        };
                      }
                      if (code === 'FILE_METADATA_SUSPICIOUS') {
                        return {
                          title: 'Verdachte PDF maker',
                          description: 'Dit PDF-bestand is gemaakt met een online converter of tool die vaak gebruikt wordt voor document manipulatie.',
                          recommendation: 'Wees extra voorzichtig en controleer de authenticiteit van dit document.'
                        };
                      }
                      if (code === 'TEXT_ANOMALY') {
                        return {
                          title: 'Tekst afwijkingen gedetecteerd',
                          description: 'De tekst in dit document bevat ongebruikelijke patronen die kunnen wijzen op manipulatie of vervalsing.',
                          recommendation: 'Vergelijk de tekst met het origineel en controleer op onregelmatigheden.',
                          hasExamples: true
                        };
                      }
                      if (code === 'CONSISTENCY_CHECK_FAILED') {
                        return {
                          title: 'Inconsistente gegevens',
                          description: 'De geëxtraheerde gegevens uit dit document bevatten tegenstrijdigheden die kunnen wijzen op manipulatie.',
                          recommendation: 'Controleer de gegevens handmatig en vergelijk met andere bronnen.'
                        };
                      }
                      return {
                        title: message,
                        description: evidence,
                        recommendation: null
                      };
                    };

                    const explanation = getSignalExplanation(signal.code, signal.message, signal.evidence);
                    
                    const hasExamples = explanation.hasExamples && (signal as any).examples && (
                      ((signal as any).examples.unicode_examples && (signal as any).examples.unicode_examples.length > 0) ||
                      ((signal as any).examples.repetition_examples && (signal as any).examples.repetition_examples.length > 0)
                    );
                    
                    return (
                      <div 
                        key={i} 
                        onClick={hasExamples ? () => onShowExamples(i, (signal as any).examples) : undefined}
                        className={`bg-white/5 rounded-lg p-3 border transition-all relative ${
                          hasExamples 
                            ? 'cursor-pointer hover:bg-white/10 hover:border-white/20 group' 
                            : ''
                        } ${
                          signal.severity === 'high' 
                            ? 'border-red-500/30 bg-red-500/5' 
                            : signal.severity === 'medium'
                              ? 'border-yellow-500/30 bg-yellow-500/5'
                              : 'border-blue-500/30 bg-blue-500/5'
                        } ${
                          hasExamples ? 'pr-8' : ''
                        }`}
                      >
                        {hasExamples && (
                          <div className="absolute top-3 right-3 opacity-40 group-hover:opacity-70 transition-opacity">
                            <FontAwesomeIcon 
                              icon={faAngleRight} 
                              className="w-4 h-4 text-white/60" 
                            />
                          </div>
                        )}
                        <div className="flex items-start gap-2 mb-1.5">
                          <FontAwesomeIcon 
                            icon={signal.severity === 'high' ? faExclamationTriangle : faInfoCircle} 
                            className={`w-4 h-4 mt-0.5 ${
                              signal.severity === 'high' ? 'text-red-400' : signal.severity === 'medium' ? 'text-yellow-400' : 'text-blue-400'
                            }`} 
                          />
                          <div className="flex-1 min-w-0">
                            <div className={`font-semibold text-sm mb-1 flex items-center gap-2 flex-wrap ${
                              signal.severity === 'high' ? 'text-red-300' : signal.severity === 'medium' ? 'text-yellow-300' : 'text-blue-300'
                            }`}>
                              {explanation.title}
                              {hasExamples && (
                                <div className="flex items-center gap-1.5">
                                  {(signal as any).examples.unicode_examples && (signal as any).examples.unicode_examples.length > 0 && (
                                    <span className="inline-flex items-center gap-1 text-xs font-normal text-white/60 bg-white/10 px-2 py-0.5 rounded-full border border-white/10" title={`${(signal as any).examples.unicode_examples.length} unicode afwijking${(signal as any).examples.unicode_examples.length !== 1 ? 'en' : ''}`}>
                                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full"></span>
                                      <span>{(signal as any).examples.unicode_examples.length}</span>
                                    </span>
                                  )}
                                  {(signal as any).examples.repetition_examples && (signal as any).examples.repetition_examples.length > 0 && (
                                    <span className="inline-flex items-center gap-1 text-xs font-normal text-white/60 bg-white/10 px-2 py-0.5 rounded-full border border-white/10" title={`${(signal as any).examples.repetition_examples.length} herhalingspatroon${(signal as any).examples.repetition_examples.length !== 1 ? 'en' : ''}`}>
                                      <span className="w-1.5 h-1.5 bg-orange-400 rounded-full"></span>
                                      <span>{(signal as any).examples.repetition_examples.length}</span>
                                    </span>
                                  )}
                                  <span className="text-xs font-normal text-white/50 bg-white/10 px-2 py-0.5 rounded-full border border-white/10">
                                    Klik voor details
                                  </span>
                                </div>
                              )}
                            </div>
                            <div className="text-white/70 text-xs leading-relaxed">
                              {explanation.description}
                            </div>
                            {explanation.recommendation && (
                              <div className="mt-2 pt-2 border-t border-white/10 text-white/60 text-xs italic">
                                💡 {explanation.recommendation}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="mt-3 text-white/60 text-xs">
                  Geen fraude detecties gevonden. Document lijkt schoon.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {document.error_message && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
          <div className="flex items-start space-x-2">
            <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-400 w-4 h-4 mt-0.5" />
            <div>
              <h4 className="text-red-400 font-medium text-sm">Error</h4>
              <p className="text-red-300 text-xs mt-1">{document.error_message}</p>
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
  const hasDbData = !!document.metadata_json && Object.keys(document.metadata_json || {}).length > 0;
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

  const formatValue = (value: any): string => {
    if (value === null || value === undefined) {
      return String(value);
    }
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  return (
    <div className="p-3 sm:p-4 space-y-3 sm:space-y-4 overflow-y-auto">
      <div className="flex flex-wrap gap-2">
        <button onClick={() => downloadArtifact('metadata/result.json', 'metadata.json')} className="flex items-center space-x-1 px-2 py-1.5 bg-white/10 text-white text-xs rounded hover:bg-white/20">
          <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
          <span>Metadata</span>
        </button>
        <button onClick={() => downloadArtifact('metadata/evidence.json', 'evidence.json')} className="flex items-center space-x-1 px-2 py-1.5 bg-white/10 text-white text-xs rounded hover:bg-white/20">
          <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
          <span>Evidence</span>
        </button>
      </div>

      {hasData ? (
        <div className="grid gap-1.5">
          {Object.entries(data!).map(([key, value]) => {
            const isArray = Array.isArray(value);
            const isNull = value === null || value === undefined;
            const isObject = typeof value === 'object' && !isArray && !isNull;
            const fieldEvidence = evidence[key] || [];
            const hasMultipleEvidence = Array.isArray(fieldEvidence) && fieldEvidence.length > 1;
            
            return (
              <div key={key} className="bg-white/5 rounded px-2 py-1.5 border border-white/10 hover:bg-white/10 transition-colors">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-white/50 text-[10px] font-medium uppercase">{key.replace(/_/g, ' ')}</span>
                      {hasMultipleEvidence && (
                        <span className="text-[9px] bg-blue-500/20 text-blue-300 px-1 py-0.5 rounded">
                          {fieldEvidence.length} bronnen
                        </span>
                      )}
                    </div>
                    {isNull ? (
                      <span className="block text-white/30 text-xs italic">null</span>
                    ) : isArray ? (
                      <div className="flex flex-wrap gap-1 mt-0.5">
                        {(value as any[]).map((item, i) => (
                          <span key={i} className="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded">
                            {typeof item === 'object' ? JSON.stringify(item) : String(item)}
                          </span>
                        ))}
                      </div>
                    ) : isObject ? (
                      <pre className="text-white/80 text-[10px] font-mono mt-0.5 bg-black/20 rounded p-1 overflow-x-auto">
                        {JSON.stringify(value, null, 2)}
                      </pre>
                    ) : (
                      <span className="block text-white font-mono text-xs break-all">{String(value)}</span>
                    )}
                    
                    {/* Show all evidence quotes if multiple */}
                    {Array.isArray(fieldEvidence) && fieldEvidence.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {fieldEvidence.map((ev: any, i: number) => (
                          <span 
                            key={i} 
                            className="text-[9px] bg-blue-500/20 text-blue-200 px-1.5 py-0.5 rounded border border-blue-500/30"
                            title={`Pagina ${(ev.page || 0) + 1}, positie ${ev.start}-${ev.end}`}
                          >
                            "{ev.quote?.substring(0, 40)}{ev.quote?.length > 40 ? '...' : ''}"
                            <span className="text-blue-400/60 ml-1">p{(ev.page || 0) + 1}</span>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <button onClick={() => copyToClipboard(formatValue(value))} className="text-white/30 hover:text-white/60 shrink-0 p-0.5">
                    <FontAwesomeIcon icon={faCopy} className="w-2.5 h-2.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-8">
          <FontAwesomeIcon icon={faInfoCircle} className="text-white/40 w-8 h-8 mb-2" />
          <p className="text-white/60 text-sm">No extracted data</p>
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
    <div className="border border-white/10 rounded-lg overflow-hidden">
      <div 
        className="flex items-center justify-between p-2 bg-white/5 hover:bg-white/10 transition-colors cursor-pointer"
        onClick={onToggle}
      >
        <div className="flex-1 text-left text-white/80 text-xs font-medium hover:text-white">
          <div className="flex items-center gap-2">
            <span>{title}</span>
            {description && (
              <span className="text-white/40 text-[10px] font-normal">({description})</span>
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
              className="text-white/40 hover:text-white p-1 cursor-pointer"
              title="Download"
            >
              <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
            </button>
          )}
          <div className="text-white/40 p-1">
            <FontAwesomeIcon icon={isExpanded ? faChevronUp : faChevronDown} className="w-3 h-3" />
          </div>
        </div>
      </div>
      {isExpanded ? (
        <div 
          className="max-h-64 overflow-y-auto bg-black/20"
          onScroll={handleScroll}
          onWheel={(e) => e.stopPropagation()}
          onTouchMove={(e) => e.stopPropagation()}
          ref={scrollRef}
        >
          <pre 
            className="text-white/80 text-xs font-mono whitespace-pre-wrap break-words p-3 cursor-text select-text"
            style={{ margin: 0 }}
          >
            {content}
          </pre>
        </div>
      ) : (
        <div className="text-white/50 text-xs font-mono p-2 truncate cursor-pointer" onClick={onToggle}>{preview}</div>
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

  // Only fetch LLM artifacts when document is done - prevents 404 errors for pending/processing docs
  const isDone = document?.status === 'done';
  
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
        timing: await safeJson<Record<string, any>>('llm/extraction_timing.json'),
        skipped: await safeJson<Record<string, any>>('llm/extraction_skipped.json'),
      };
      return { classification, extraction };
    },
    enabled: !!documentId && isDone,
    retry: false,
    staleTime: 0, // Always refetch to get latest prompts (especially after rerun)
    refetchOnMount: 'always', // Force refetch when component mounts
    gcTime: 0, // Don't cache at all (previously called cacheTime)
  });

  // Show appropriate message based on document status - BEFORE any data fetching
  if (!isDone) {
    if (document?.status === 'processing') {
      return (
        <div className="p-4">
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 text-center">
            <FontAwesomeIcon icon={faRobot} className="text-blue-400 w-8 h-8 mb-2 animate-pulse" />
            <p className="text-white font-medium text-sm">Processing...</p>
            <p className="text-white/60 text-xs">{formatStage(document.stage)}</p>
            {document.progress !== undefined && (
              <div className="w-full bg-white/20 rounded-full h-1.5 mt-2">
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
        <FontAwesomeIcon icon={faRobot} className="text-white/40 w-8 h-8 mb-2" />
        <p className="text-white/60 text-sm">
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
        <FontAwesomeIcon icon={faSpinner} className="text-blue-400 w-6 h-6 animate-spin" />
      </div>
    );
  }

  // This is now only reached when status is 'done' but no data was found
  if (!data) {
    return (
      <div className="p-4 text-center">
        <FontAwesomeIcon icon={faRobot} className="text-white/40 w-8 h-8 mb-2" />
        <p className="text-white/60 text-sm">Run analysis to see LLM logs</p>
      </div>
    );
  }

  const hasData = data?.classification.prompt || data?.classification.result || data?.classification.deterministic || data?.classification.local || data?.extraction.prompt || data?.extraction.result || data?.extraction.skipped;

  if (!hasData) {
    return (
      <div className="p-4 text-center">
        <FontAwesomeIcon icon={faRobot} className="text-white/40 w-8 h-8 mb-2" />
        <p className="text-white/60 text-sm">No LLM data available</p>
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
            <h3 className="text-white font-medium text-sm flex items-center space-x-2">
              <FontAwesomeIcon icon={faRobot} className="text-blue-400 w-3 h-3" />
              <span>LLM Classification Logs</span>
            </h3>
            {data.classification.timing && (
              <div className="text-white/60 text-xs flex items-center gap-1.5">
                <FontAwesomeIcon icon={faBolt} className="w-3 h-3" />
                <span>{data.classification.timing.duration_seconds ? `${data.classification.timing.duration_seconds.toFixed(2)}s` : 'N/A'}</span>
                {data.classification.timing.provider && (
                  <span className="text-white/40">({data.classification.timing.provider})</span>
                )}
              </div>
            )}
          </div>

          {data.classification.error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded p-2 text-xs text-red-300">
              <strong>Error:</strong> {data.classification.error.substring(0, 200)}...
            </div>
          )}

          {/* Show message if classification failed (no result but has prompt) */}
          {!data.classification.result && !data.classification.error && data.classification.prompt && (
            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded p-2 text-xs text-yellow-300">
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
                      <div className="bg-yellow-500/10 border border-yellow-500/20 rounded p-2 text-xs text-yellow-300">
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
                  <div className="bg-green-500/10 border border-green-500/20 rounded p-3 mt-3">
                    <div className="text-green-300 text-xs font-medium mb-2 flex items-center gap-2">
                      <span>✅</span>
                      <span>STRONG Keyword Match (100% confidence)</span>
                    </div>
                    <div className="text-white/80 text-xs mb-1">Gematchte keywords:</div>
                    <div className="flex flex-wrap gap-1.5">
                      {data.classification.local.matched_keywords.map((keyword: string, i: number) => (
                        <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-green-500/20 border border-green-500/30 text-green-200 text-[10px]">
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
      {(data?.extraction.prompt || data?.extraction.result || data?.extraction.error || data?.extraction.skipped) && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-white font-medium text-sm flex items-center space-x-2">
              <FontAwesomeIcon icon={faSearch} className="text-purple-400 w-3 h-3" />
              <span>Metadata Extraction</span>
            </h3>
            {data.extraction.timing && (
              <div className="text-white/60 text-xs flex items-center gap-1.5">
                <FontAwesomeIcon icon={faBolt} className="w-3 h-3" />
                <span>{data.extraction.timing.duration_seconds ? `${data.extraction.timing.duration_seconds.toFixed(2)}s` : 'N/A'}</span>
                {data.extraction.timing.provider && (
                  <span className="text-white/40">({data.extraction.timing.provider})</span>
                )}
              </div>
            )}
          </div>

          {/* Show message if extraction was skipped */}
          {data.extraction.skipped && (
            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded p-2 text-xs text-yellow-300">
              <strong>Extractie overgeslagen:</strong> {data.extraction.skipped.reason || 'Onbekende reden'}
              {data.extraction.skipped.doc_type_slug && (
                <div className="mt-1 text-yellow-200/80">
                  Document type: <span className="font-mono">{data.extraction.skipped.doc_type_slug}</span>
                </div>
              )}
              <div className="mt-1 text-yellow-200/60 text-[10px]">
                Voeg fields toe aan dit document type om extractie mogelijk te maken.
              </div>
            </div>
          )}

          {data.extraction.error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded p-2 text-xs text-red-300">
              <strong>Error:</strong> {data.extraction.error.substring(0, 200)}...
            </div>
          )}

          {/* Show message if extraction failed (no result but has prompt) */}
          {!data.extraction.result && !data.extraction.error && !data.extraction.skipped && data.extraction.prompt && (
            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded p-2 text-xs text-yellow-300">
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
                      <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2 text-xs text-blue-300">
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
            className="w-full px-3 py-1.5 pl-8 bg-white/10 border border-white/20 rounded text-white text-sm placeholder-white/60 focus:ring-1 focus:ring-blue-400"
          />
          <FontAwesomeIcon icon={faSearch} className="absolute left-2.5 top-1/2 transform -translate-y-1/2 text-white/60 w-3 h-3" />
        </div>
        <div className="flex space-x-2">
          <button
            onClick={onHighlightToggle}
            className={`flex items-center space-x-1 px-2 py-1.5 text-xs rounded ${highlightSources ? 'bg-blue-600 text-white' : 'bg-white/10 text-white hover:bg-white/20'}`}
          >
            <FontAwesomeIcon icon={faHighlighter} className="w-3 h-3" />
            <span>Evidence</span>
          </button>
          <button onClick={() => downloadArtifact('text/extracted.txt', 'text.txt')} className="flex items-center space-x-1 px-2 py-1.5 bg-white/10 text-white text-xs rounded hover:bg-white/20">
            <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
          </button>
        </div>
      </div>

      <div className="bg-white/5 rounded-lg p-3">
        <div className="max-h-64 overflow-y-auto">
          {textLoading ? (
            <div className="flex items-center justify-center py-4">
              <FontAwesomeIcon icon={faSpinner} className="text-blue-400 w-5 h-5 animate-spin" />
            </div>
          ) : (
            <pre className="text-white text-xs whitespace-pre-wrap">{extractedText || 'No text'}</pre>
          )}
        </div>
      </div>

      {highlightSources && document.metadata_evidence_json && Object.keys(document.metadata_evidence_json).length > 0 && (
        <div className="space-y-2">
          <h3 className="text-white font-medium text-sm">Evidence</h3>
          {Object.entries(document.metadata_evidence_json).map(([field, spans]) => (
            <div key={field} className="bg-white/5 rounded overflow-hidden">
              <button
                onClick={() => setExpandedHighlights(prev => { const n = new Set(prev); n.has(field) ? n.delete(field) : n.add(field); return n; })}
                className="w-full flex items-center justify-between p-2 text-left hover:bg-white/10"
              >
                <span className="text-white text-xs font-medium">{field}</span>
                <FontAwesomeIcon icon={expandedHighlights.has(field) ? faChevronUp : faChevronDown} className="text-white/60 w-3 h-3" />
              </button>
              {expandedHighlights.has(field) && Array.isArray(spans) && (
                <div className="px-2 pb-2 space-y-1">
                  {spans.map((span: any, i: number) => (
                    <div key={i} className="bg-blue-500/10 rounded p-1.5 text-xs">
                      <span className="text-blue-300">Page {span.page}</span>
                      {span.quote && <div className="text-white mt-1">"{span.quote}"</div>}
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
      case 'critical': return 'text-red-400 bg-red-500/20 border-red-500/30';
      case 'high': return 'text-orange-400 bg-orange-500/20 border-orange-500/30';
      case 'medium': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      case 'low': return 'text-green-400 bg-green-500/20 border-green-500/30';
      default: return 'text-white/60 bg-white/10 border-white/20';
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
        <FontAwesomeIcon icon={faSpinner} className="text-white/40 text-2xl animate-spin mb-2" />
        <p className="text-white/60 text-sm">Document wordt verwerkt...</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="p-8 text-center">
        <FontAwesomeIcon icon={faSpinner} className="text-white/40 text-2xl animate-spin mb-2" />
        <p className="text-white/60 text-sm">Forensische analyse wordt uitgevoerd...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
          <p className="text-red-400 text-sm">Fout bij laden van forensische analyse</p>
        </div>
      </div>
    );
  }

  if (!fraudReport) {
    return (
      <div className="p-8 text-center">
        <p className="text-white/60 text-sm">Geen forensische data beschikbaar</p>
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

      {/* Signals List - Accordion (only one open at a time) */}
      {filteredSignals.length > 0 ? (
        <div className="space-y-2">
          {/* Static header - not clickable */}
          <div className="w-full flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10">
            <h3 className="text-white font-semibold text-base flex items-center gap-2">
              <FontAwesomeIcon icon={faShieldAlt} className="text-blue-400 w-5 h-5" />
              Gedetecteerde Signalen
            </h3>
            <div className="flex items-center gap-2">
              <span className="text-white/60 text-xs bg-black/20 px-2 py-1 rounded">
                {filteredSignals.length} totaal
              </span>
              {expandedSignalIndex !== null && (
                <span className="text-blue-400 text-xs">
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
                  className="w-full flex items-start justify-between gap-2 p-2.5 hover:bg-black/10 transition-colors text-left cursor-pointer"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                      <FontAwesomeIcon icon={getRiskIcon(signal.risk_level)} className="w-3.5 h-3.5 shrink-0" />
                      <span className="font-medium text-xs truncate">{signal.name.replace(/_/g, ' ')}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/20 shrink-0">
                        {Math.round(signal.confidence * 100)}%
                      </span>
                    </div>
                    <p className="text-xs opacity-80 line-clamp-2">{signal.description}</p>
                  </div>
                  <FontAwesomeIcon 
                    icon={isExpanded ? faChevronUp : faChevronDown} 
                    className="text-white/40 w-3 h-3 shrink-0 mt-0.5"
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
                            <span key={i} className="text-[10px] bg-black/20 px-1.5 py-0.5 rounded">
                              {e}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Technical Details - Auto-opened when signal is expanded */}
                    {Object.keys(signal.details).length > 0 && (
                      <div className="pt-2">
                        <div className="text-[10px] opacity-60 mb-1.5 font-medium">Technische details</div>
                        <pre className="text-[10px] bg-black/20 p-1.5 rounded overflow-x-auto max-h-32 overflow-y-auto">
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
        <div className="text-center py-4 bg-green-500/10 border border-green-500/20 rounded-lg">
          <p className="text-green-400 font-medium text-sm">✓ Geen verdachte signalen</p>
          <p className="text-white/50 text-xs mt-0.5">Dit document lijkt veilig</p>
        </div>
      )}

      {/* ELA Heatmap - Collapsible */}
      <div className="bg-white/5 rounded-lg border border-white/10 overflow-hidden">
        <button
          onClick={() => setShowHeatmap(!showHeatmap)}
          className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 transition-colors cursor-pointer"
        >
          <h4 className="text-white/70 text-sm font-medium flex items-center gap-2">
            <FontAwesomeIcon icon={faImage} className="text-purple-400 w-4 h-4" />
            ELA Heatmap
          </h4>
          <FontAwesomeIcon 
            icon={showHeatmap ? faChevronUp : faChevronDown} 
            className="text-white/60 w-3 h-3"
          />
        </button>
        
        {showHeatmap && (
          <div className="px-2.5 pb-2.5">
            {elaHeatmapUrl ? (
              <div className="bg-black/20 rounded-lg p-2 border border-white/10">
                <img 
                  key={`ela-heatmap-${documentId}-${elaHeatmapUrl}`}
                  src={elaHeatmapUrl || undefined} 
                  alt="ELA Heatmap" 
                  className="w-full h-auto rounded border border-white/10 max-h-64 object-contain mx-auto"
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
                      errorMsg.className = 'text-red-400 text-xs mt-2';
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
                <p className="text-white/50 text-xs mt-2">
                  Compressieverschillen. Heldere gebieden = mogelijke manipulatie.
                  {fraudReport?.signals?.some(s => s.name === 'ela_manipulation_detected') && (
                    <span className="block mt-1 text-yellow-300 text-[10px]">
                      ⚠️ Manipulatie gedetecteerd
                    </span>
                  )}
                </p>
              </div>
            ) : (
              <p className="text-white/50 text-xs py-2">
                Geen heatmap beschikbaar
              </p>
            )}
          </div>
        )}
      </div>

      {/* Analysis Methods Info - Always visible */}
      <div className="bg-white/5 rounded-lg p-3 border border-white/10">
        <h4 className="text-white/70 text-sm font-medium mb-2">Analyse Methodes</h4>
        <div className="grid grid-cols-2 gap-1.5 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-blue-500/20 border border-blue-500/30 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faFilePdf} className="w-2.5 h-2.5 text-blue-400" />
            </div>
            <span className="text-white/60 text-[10px]">PDF Metadata</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-purple-500/20 border border-purple-500/30 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faImage} className="w-2.5 h-2.5 text-purple-400" />
            </div>
            <span className="text-white/60 text-[10px]">ELA Analyse</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-amber-500/20 border border-amber-500/30 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-2.5 h-2.5 text-amber-400" />
            </div>
            <span className="text-white/60 text-[10px]">Unicode</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-green-500/20 border border-green-500/30 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faRedo} className="w-2.5 h-2.5 text-green-400" />
            </div>
            <span className="text-white/60 text-[10px]">Herhaling</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faBullseye} className="w-2.5 h-2.5 text-cyan-400" />
            </div>
            <span className="text-white/60 text-[10px]">Confidence</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 rounded bg-pink-500/20 border border-pink-500/30 flex items-center justify-center shrink-0">
              <FontAwesomeIcon icon={faCog} className="w-2.5 h-2.5 text-pink-400" />
            </div>
            <span className="text-white/60 text-[10px]">Software</span>
          </div>
        </div>
      </div>
    </div>
  );
}
