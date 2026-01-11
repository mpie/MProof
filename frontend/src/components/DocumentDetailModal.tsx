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

// Document Viewer Modal Component
function DocumentViewerModal({ 
  isOpen, 
  onClose, 
  documentUrl, 
  filename, 
  mimeType,
  onDownload
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  documentUrl: string | null; 
  filename: string;
  mimeType: string;
  onDownload: () => void;
}) {
  if (!isOpen || !documentUrl) return null;

  const isPDF = mimeType === 'application/pdf';
  const isImage = mimeType.startsWith('image/');
  const isWord = mimeType.includes('word') || mimeType.includes('msword') || filename.toLowerCase().endsWith('.doc') || filename.toLowerCase().endsWith('.docx');

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
              <p className="text-white/60 text-xs">Origineel bestand</p>
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
        <div className="flex-1 overflow-auto bg-black/30 p-2 sm:p-4 flex items-center justify-center">
          {isPDF ? (
            <iframe
              src={`${documentUrl}#toolbar=1&navpanes=1&scrollbar=1&view=FitH`}
              className="w-full h-full min-h-[80vh] rounded-lg border border-white/10 bg-white"
              title={filename}
              style={{ colorScheme: 'light' }}
            />
          ) : isImage ? (
            <img
              src={documentUrl}
              alt={filename}
              className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
            />
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
            <div className="text-center text-white/60">
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
    onSuccess: async () => {
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
      
      // Invalidate all related queries to force fresh data
      await queryClient.invalidateQueries({ queryKey: ['document', documentId] });
      queryClient.removeQueries({ queryKey: ['document-text', documentId] });
      queryClient.removeQueries({ queryKey: ['document-llm', documentId] });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['documents-recent'] });
      
      await refetch();
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
    } catch (error) {
      console.error('Download failed:', error);
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
              {activeTab === 'overview' && <OverviewTab document={document} formatFileSize={formatFileSize} formatDate={formatDate} formatStage={formatStage} getStatusColor={getStatusColor} getRiskColor={getRiskColor} onShowExamples={(signalIndex, examples) => setExamplesSidebar({ signalIndex, examples })} />}
              {activeTab === 'text' && <TextTab document={document} highlightSources={highlightSources} searchTerm={searchTerm} onHighlightToggle={() => setHighlightSources(!highlightSources)} onSearchChange={setSearchTerm} downloadArtifact={downloadArtifact} extractedText={extractedText} textLoading={textLoading} />}
              {activeTab === 'metadata' && <MetadataTab documentId={documentId} document={document} copyToClipboard={copyToClipboard} downloadArtifact={downloadArtifact} />}
              {activeTab === 'llm' && <LLMTab documentId={documentId} document={document} downloadArtifact={downloadArtifact} />}
              {activeTab === 'forensics' && <ForensicsTab documentId={documentId!} document={document} />}
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
        />
      )}
    </>
  );
}

function OverviewTab({ document, formatFileSize, formatDate, formatStage, getStatusColor, getRiskColor, onShowExamples }: {
  document: Document;
  formatFileSize: (bytes: number) => string;
  formatDate: (date: string) => string;
  formatStage: (stage: string | undefined) => string;
  getStatusColor: (status: string) => string;
  getRiskColor: (score: number) => string;
  onShowExamples: (signalIndex: number, examples: any) => void;
}) {
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
            <div className="flex justify-between"><span className="text-white/60">Uploaded:</span><span className="text-white">{formatDate(document.created_at)}</span></div>
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
              {document.doc_type_rationale && (
                <div className="mt-3 pt-3 border-t border-white/10">
                  <div className="text-white/50 text-xs mb-1.5">Classificatie methode:</div>
                  <div className="flex flex-wrap gap-1.5 mb-2">
                    {document.doc_type_rationale.includes('Deterministic') && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-yellow-500/20 border border-yellow-500/30 text-yellow-200 text-[10px]">
                        ⚠️ Keyword/regex match
                      </span>
                    )}
                    {document.doc_type_rationale.includes('Local classifier') && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-500/20 border border-blue-500/30 text-blue-200 text-[10px]">
                        🤖 Getraind model
                      </span>
                    )}
                    {document.doc_type_rationale.includes('LLM') && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-purple-500/20 border border-purple-500/30 text-purple-200 text-[10px]">
                        🧠 AI classificatie
                      </span>
                    )}
                  </div>
                  
                  {/* Classification Scores */}
                  {document.metadata_validation_json?.classification_scores && (
                    <div className="mt-2 space-y-1.5">
                      <div className="text-white/50 text-xs">Classifier Scores:</div>
                      <div className="grid grid-cols-2 gap-2">
                        {document.metadata_validation_json.classification_scores.naive_bayes && (
                          <div className="bg-purple-500/10 border border-purple-500/20 rounded p-2">
                            <div className="text-purple-300 text-xs font-medium mb-0.5">Naive Bayes</div>
                            <div className="text-white text-xs">
                              {document.metadata_validation_json.classification_scores.naive_bayes.label}
                            </div>
                            <div className="text-purple-400 text-[10px] mt-0.5">
                              {Math.round(document.metadata_validation_json.classification_scores.naive_bayes.confidence * 100)}% confidence
                            </div>
                          </div>
                        )}
                        {document.metadata_validation_json.classification_scores.bert && (
                          <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2">
                            <div className="text-blue-300 text-xs font-medium mb-0.5">BERT</div>
                            <div className="text-white text-xs">
                              {document.metadata_validation_json.classification_scores.bert.label}
                            </div>
                            <div className="text-blue-400 text-[10px] mt-0.5">
                              {Math.round(document.metadata_validation_json.classification_scores.bert.confidence * 100)}% confidence
                            </div>
                          </div>
                        )}
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
              )}
            </div>
          )}

          {document.risk_score !== null && (
            <div className="bg-red-500/10 rounded-lg p-4 border border-red-500/20">
              <h3 className="text-white font-medium mb-3 flex items-center space-x-2">
                <FontAwesomeIcon icon={faExclamationTriangle} className="text-red-400 w-4 h-4" />
                <span>Risk</span>
              </h3>
              <div className="flex items-center space-x-3 mb-2">
                <span className={`text-2xl font-bold ${(document.risk_score ?? 0) >= 70 ? 'text-red-400' : (document.risk_score ?? 0) >= 40 ? 'text-yellow-400' : 'text-green-400'}`}>
                  {document.risk_score ?? 0}
                </span>
                <div className="flex-1 bg-white/20 rounded-full h-2">
                  <div className={`h-2 rounded-full ${getRiskColor(document.risk_score ?? 0)}`} style={{ width: `${document.risk_score ?? 0}%` }} />
                </div>
              </div>
              {(document.risk_signals_json?.length ?? 0) > 0 && (
                <div className="space-y-2 mt-3">
                  <div className="text-xs text-white/60 mb-2">
                    {document.risk_signals_json.length} fraude detectie{document.risk_signals_json.length !== 1 ? 's' : ''} gevonden:
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
                    
                    const hasExamples = explanation.hasExamples && signal.examples && (
                      (signal.examples.unicode_examples && signal.examples.unicode_examples.length > 0) ||
                      (signal.examples.repetition_examples && signal.examples.repetition_examples.length > 0)
                    );
                    
                    return (
                      <div 
                        key={i} 
                        onClick={hasExamples ? () => onShowExamples(i, signal.examples) : undefined}
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
                                  {signal.examples.unicode_examples && signal.examples.unicode_examples.length > 0 && (
                                    <span className="inline-flex items-center gap-1 text-xs font-normal text-white/60 bg-white/10 px-2 py-0.5 rounded-full border border-white/10" title={`${signal.examples.unicode_examples.length} unicode afwijking${signal.examples.unicode_examples.length !== 1 ? 'en' : ''}`}>
                                      <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full"></span>
                                      <span>{signal.examples.unicode_examples.length}</span>
                                    </span>
                                  )}
                                  {signal.examples.repetition_examples && signal.examples.repetition_examples.length > 0 && (
                                    <span className="inline-flex items-center gap-1 text-xs font-normal text-white/60 bg-white/10 px-2 py-0.5 rounded-full border border-white/10" title={`${signal.examples.repetition_examples.length} herhalingspatroon${signal.examples.repetition_examples.length !== 1 ? 'en' : ''}`}>
                                      <span className="w-1.5 h-1.5 bg-orange-400 rounded-full"></span>
                                      <span>{signal.examples.repetition_examples.length}</span>
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
        <div className="grid gap-2 sm:gap-3">
          {Object.entries(data!).map(([key, value]) => (
            <div key={key} className="bg-purple-500/10 rounded-lg p-2.5 sm:p-3 border border-purple-500/20">
              <div className="flex items-center justify-between gap-2 sm:gap-3">
                <div className="flex items-start sm:items-center gap-1.5 sm:gap-2 flex-1 min-w-0">
                  <span className="text-white/70 text-[10px] sm:text-xs font-medium uppercase shrink-0 pt-0.5 sm:pt-0">{key.replace(/_/g, ' ')}:</span>
                  <span className="text-white font-mono text-xs sm:text-sm break-all whitespace-pre-wrap">{formatValue(value)}</span>
                </div>
                <button onClick={() => copyToClipboard(formatValue(value))} className="text-purple-400 hover:text-purple-300 shrink-0 p-1">
                  <FontAwesomeIcon icon={faCopy} className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}
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
      };
      const extraction = {
        prompt: await safeText('llm/extraction_prompt.txt'),
        response: await safeText('llm/extraction_response.txt'),
        result: await safeJson<Record<string, any>>('llm/extraction_result.json'),
        error: await safeText('llm/extraction_error.txt'),
      };
      return { classification, extraction };
    },
    enabled: !!documentId && isDone,
    retry: false,
    staleTime: Infinity, // Don't refetch automatically
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

  const hasData = data?.classification.prompt || data?.classification.result || data?.extraction.prompt || data?.extraction.result;

  if (!hasData) {
    return (
      <div className="p-4 text-center">
        <FontAwesomeIcon icon={faRobot} className="text-white/40 w-8 h-8 mb-2" />
        <p className="text-white/60 text-sm">No LLM data available</p>
      </div>
    );
  }

  const CollapsibleBlock = ({ id, title, content, downloadPath, downloadName }: { id: string; title: string; content: string | null; downloadPath?: string; downloadName?: string }) => {
    if (!content) return null;
    const isExpanded = expandedSections.has(id);
    const preview = content.length > 100 ? content.substring(0, 100) + '...' : content;

    return (
      <div className="border border-white/10 rounded-lg overflow-hidden">
        <div className="flex items-center justify-between p-2 bg-white/5">
          <button
            onClick={() => toggleSection(id)}
            className="flex-1 text-left text-white/80 text-xs font-medium hover:text-white"
          >
            {title}
          </button>
          <div className="flex items-center space-x-2">
            {downloadPath && (
              <button
                onClick={() => downloadArtifact(downloadPath, downloadName || 'download.txt')}
                className="text-white/40 hover:text-white p-1"
                title="Download"
              >
                <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
              </button>
            )}
            <button onClick={() => toggleSection(id)} className="text-white/40 hover:text-white p-1">
              <FontAwesomeIcon icon={isExpanded ? faChevronUp : faChevronDown} className="w-3 h-3" />
            </button>
          </div>
        </div>
        {isExpanded ? (
          <pre className="text-white/80 text-xs font-mono whitespace-pre-wrap break-words p-3 max-h-48 overflow-y-auto bg-black/20">
            {content}
          </pre>
        ) : (
          <div className="text-white/50 text-xs font-mono p-2 truncate">{preview}</div>
        )}
      </div>
    );
  };

  return (
    <div className="p-4 space-y-4">
      {/* Classification */}
      {(data?.classification.prompt || data?.classification.result || data?.classification.error) && (
        <div className="space-y-2">
          <h3 className="text-white font-medium text-sm flex items-center space-x-2">
            <FontAwesomeIcon icon={faRobot} className="text-blue-400 w-3 h-3" />
            <span>Classification</span>
          </h3>

          {data.classification.error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded p-2 text-xs text-red-300">
              <strong>Error:</strong> {data.classification.error.substring(0, 200)}...
            </div>
          )}

          <div className="space-y-2">
            <CollapsibleBlock id="class-prompt" title="Prompt" content={data.classification.prompt} downloadPath="llm/classification_prompt.txt" downloadName="classification_prompt.txt" />
            {/* Only show Response if it's different from Result (e.g., contains non-JSON text or is formatted differently) */}
            {(() => {
              const responseText = data.classification.response?.trim() || '';
              const resultJson = data.classification.result ? JSON.stringify(data.classification.result, null, 2) : '';
              const responseIsJson = responseText && (responseText.startsWith('{') || responseText.startsWith('['));
              const responseMatchesResult = responseIsJson && responseText === resultJson;
              
              // Show Response only if it's different from Result
              if (data.classification.response && !responseMatchesResult) {
                return (
                  <CollapsibleBlock id="class-response" title="Response" content={data.classification.response} downloadPath="llm/classification_response.txt" downloadName="classification_response.txt" />
                );
              }
              return null;
            })()}
            {data.classification.result && (
              <CollapsibleBlock id="class-result" title="Result (JSON)" content={JSON.stringify(data.classification.result, null, 2)} downloadPath="llm/classification_result.json" downloadName="classification_result.json" />
            )}
          </div>
        </div>
      )}

      {/* Extraction */}
      {(data?.extraction.prompt || data?.extraction.result || data?.extraction.error) && (
        <div className="space-y-2">
          <h3 className="text-white font-medium text-sm flex items-center space-x-2">
            <FontAwesomeIcon icon={faSearch} className="text-purple-400 w-3 h-3" />
            <span>Metadata Extraction</span>
          </h3>

          {data.extraction.error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded p-2 text-xs text-red-300">
              <strong>Error:</strong> {data.extraction.error.substring(0, 200)}...
            </div>
          )}

          <div className="space-y-2">
            <CollapsibleBlock id="ext-prompt" title="Prompt" content={data.extraction.prompt} downloadPath="llm/extraction_prompt.txt" downloadName="extraction_prompt.txt" />
            {/* Only show Response if it's different from Result (e.g., contains non-JSON text or is formatted differently) */}
            {(() => {
              const responseText = data.extraction.response?.trim() || '';
              const resultJson = data.extraction.result ? JSON.stringify(data.extraction.result, null, 2) : '';
              const responseIsJson = responseText && (responseText.startsWith('{') || responseText.startsWith('['));
              const responseMatchesResult = responseIsJson && responseText === resultJson;
              
              // Show Response only if it's different from Result
              if (data.extraction.response && !responseMatchesResult) {
                return (
                  <CollapsibleBlock id="ext-response" title="Response" content={data.extraction.response} downloadPath="llm/extraction_response.txt" downloadName="extraction_response.txt" />
                );
              }
              return null;
            })()}
            {data.extraction.result && (
              <CollapsibleBlock id="ext-result" title="Result (JSON)" content={JSON.stringify(data.extraction.result, null, 2)} downloadPath="llm/extraction_result.json" downloadName="extraction_result.json" />
            )}
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
function ForensicsTab({ documentId, document }: {
  documentId: number;
  document: Document | undefined;
}) {
  const { data: fraudReport, isLoading, error } = useQuery({
    queryKey: ['fraud-analysis', documentId],
    queryFn: () => getFraudAnalysis(documentId),
    enabled: !!documentId && document?.status === 'done',
    staleTime: Infinity, // Never refetch - data is static once analyzed
    cacheTime: Infinity, // Keep in cache forever
  });

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
    <div className="p-4 space-y-4">
      {/* Risk Score Header */}
      <div className={`rounded-xl p-4 border ${getRiskColor(fraudReport.overall_risk)}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FontAwesomeIcon icon={getRiskIcon(fraudReport.overall_risk)} className="text-3xl" />
            <div>
              <div className="text-lg font-bold">
                Risico Score: {Math.round(fraudReport.risk_score)}%
              </div>
              <div className="text-sm opacity-80">
                {getRiskLabel(fraudReport.overall_risk)} risico - {fraudReport.signals.length} signalen
              </div>
            </div>
          </div>
          <div className="text-right text-xs opacity-60">
            <div>Geanalyseerd:</div>
            <div>{new Date(fraudReport.analyzed_at).toLocaleString('nl-NL')}</div>
          </div>
        </div>
        <p className="mt-3 text-sm opacity-90">{fraudReport.summary}</p>
      </div>

      {/* Signals List */}
      {fraudReport.signals.length > 0 ? (
        <div className="space-y-3">
          <h3 className="text-white font-medium text-sm flex items-center gap-2">
            <FontAwesomeIcon icon={faShieldAlt} className="text-blue-400" />
            Gedetecteerde Signalen ({fraudReport.signals.length})
          </h3>

          {fraudReport.signals.map((signal, index) => (
            <div
              key={index}
              className={`rounded-lg border p-3 ${getRiskColor(signal.risk_level)}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <FontAwesomeIcon icon={getRiskIcon(signal.risk_level)} className="w-4 h-4" />
                    <span className="font-medium text-sm">{signal.name.replace(/_/g, ' ')}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/20">
                      {Math.round(signal.confidence * 100)}% zeker
                    </span>
                  </div>
                  <p className="text-sm opacity-80">{signal.description}</p>
                </div>
              </div>

              {/* Evidence */}
              {signal.evidence.length > 0 && (
                <div className="mt-2 pt-2 border-t border-current/20">
                  <div className="text-xs opacity-60 mb-1">Bewijs:</div>
                  <div className="flex flex-wrap gap-1">
                    {signal.evidence.map((e, i) => (
                      <span key={i} className="text-xs bg-black/20 px-2 py-0.5 rounded">
                        {e}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Technical Details */}
              {Object.keys(signal.details).length > 0 && (
                <details className="mt-2">
                  <summary className="text-xs opacity-60 cursor-pointer hover:opacity-100">
                    Technische details
                  </summary>
                  <pre className="mt-1 text-[10px] bg-black/20 p-2 rounded overflow-x-auto">
                    {JSON.stringify(signal.details, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8">
          <span className="text-4xl mb-2 block">✅</span>
          <p className="text-green-400 font-medium">Geen verdachte signalen gedetecteerd</p>
          <p className="text-white/50 text-sm mt-1">Dit document lijkt veilig</p>
        </div>
      )}

      {/* Analysis Methods Info */}
      <div className="bg-white/5 rounded-lg p-4 border border-white/10">
        <h4 className="text-white/70 text-sm font-medium mb-2">Analyse Methodes</h4>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faFilePdf} className="w-3 h-3 text-blue-400" />
            </div>
            <span className="text-white/60">PDF Metadata Analyse</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faImage} className="w-3 h-3 text-purple-400" />
            </div>
            <span className="text-white/60">ELA Beeldmanipulatie</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-amber-500/20 border border-amber-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-400" />
            </div>
            <span className="text-white/60">Unicode Anomalieën</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-green-500/20 border border-green-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faRedo} className="w-3 h-3 text-green-400" />
            </div>
            <span className="text-white/60">Tekst Herhalingspatronen</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faBullseye} className="w-3 h-3 text-cyan-400" />
            </div>
            <span className="text-white/60">Classificatie Confidence</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-pink-500/20 border border-pink-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faCog} className="w-3 h-3 text-pink-400" />
            </div>
            <span className="text-white/60">PDF Creator Software</span>
          </div>
        </div>
      </div>
    </div>
  );
}
