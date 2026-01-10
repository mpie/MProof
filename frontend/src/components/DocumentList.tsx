'use client';

import { useState, useEffect, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faFile, faSpinner, faCheck, faExclamationTriangle,
  faEye, faRedo, faFilePdf, faFileImage, faFileWord, faFileExcel, faTrash, faTimes
} from '@fortawesome/free-solid-svg-icons';
import { Document, listDocuments, subscribeToDocumentEvents, DocumentEvent, analyzeDocument, deleteDocument, getQueueStatus } from '@/lib/api';
import { DocumentDetailModal } from './DocumentDetailModal';

interface DocumentListProps {
  subjectId?: number;
  documents: Document[];
  onDocumentUpdate: (document: Document) => void;
  onDocumentsChange: (documents: Document[]) => void;
}

// Helper function to format document type names
function formatDocumentTypeName(slug: string): string {
  if (slug === 'unknown') {
    return 'Onbekend';
  }
  return slug.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

const statusIcons = {
  pending: faSpinner,
  queued: faSpinner,
  processing: faSpinner,
  done: faCheck,
  error: faExclamationTriangle,
};

const stageDescriptions = {
  sniffing: "Analyzing file type & computing checksum",
  extracting_text: "Extracting text content from document",
  classifying: "Determining document type & classification",
  extracting_metadata: "Extracting structured data & fields",
  risk_signals: "Performing fraud detection & risk analysis",
};

const statusColors = {
  pending: 'text-gray-400',
  queued: 'text-yellow-400 animate-spin',
  processing: 'text-blue-400 animate-spin',
  done: 'text-green-400',
  error: 'text-red-400',
};

const mimeTypeIcons = {
  'application/pdf': faFilePdf,
  'image/jpeg': faFileImage,
  'image/png': faFileImage,
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': faFileWord,
  'application/msword': faFileWord,
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': faFileExcel,
  'application/vnd.ms-excel': faFileExcel,
};

export function DocumentList({ subjectId, documents, onDocumentUpdate, onDocumentsChange }: DocumentListProps) {
  const [activeEventSources, setActiveEventSources] = useState<Map<number, EventSource>>(new Map());
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<{ document: Document; isOpen: boolean } | null>(null);
  const queryClient = useQueryClient();

  // Calculate queue position for queued documents
  const getQueuePosition = (document: Document): number | null => {
    if (document.status !== 'queued') return null;

    const queuedDocs = displayDocuments.filter(doc => doc.status === 'queued');
    const position = queuedDocs.findIndex(doc => doc.id === document.id);
    return position >= 0 ? position + 1 : null;
  };

  const analyzeMutation = useMutation({
    mutationFn: (documentId: number) => analyzeDocument(documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (documentId: number) => deleteDocument(documentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      // Also invalidate subject-specific queries
      queryClient.invalidateQueries({ queryKey: ['documents', subjectId] });
    },
  });

  // Fetch documents
  const { data: fetchedDocuments, isLoading } = useQuery({
    queryKey: ['documents', subjectId],
    queryFn: () => listDocuments(subjectId, undefined, 50, 0),
    enabled: !subjectId || documents.length === 0, // Only fetch if no subject selected or no documents passed
    refetchInterval: 5000, // Poll every 5 seconds to catch status updates (SSE fallback)
  });

  // Use passed documents or fetched documents - merge with passed documents for SSE updates
  const displayDocuments = useMemo(() => {
    const fetched = fetchedDocuments?.documents || [];
    if (documents.length === 0) return fetched;
    
    // Merge: use passed documents but update with fetched data for completeness
    const docMap = new Map(fetched.map(d => [d.id, d]));
    documents.forEach(d => docMap.set(d.id, d));
    return Array.from(docMap.values()).sort((a, b) => 
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  }, [documents, fetchedDocuments]);

  // Get queue status
  const { data: queueStatus } = useQuery({
    queryKey: ['queue-status'],
    queryFn: getQueueStatus,
    refetchInterval: 5000, // Update every 5 seconds
  });

  // Subscribe to real-time updates for processing documents
  useEffect(() => {
    const processingDocs = displayDocuments.filter(doc =>
      doc.status === 'processing' || doc.status === 'queued'
    );

    // Subscribe to new documents
    processingDocs.forEach(doc => {
      if (!activeEventSources.has(doc.id)) {
        const eventSource = subscribeToDocumentEvents(
          doc.id,
          (event: DocumentEvent) => {
            // Update document based on event
            const updatedDoc = { ...doc };

            if (event.type === 'status') {
              updatedDoc.status = event.status as Document['status'];
              updatedDoc.stage = event.stage || undefined;
              updatedDoc.progress = event.progress || 0;
              updatedDoc.updated_at = event.updated_at || new Date().toISOString();
            } else if (event.type === 'result') {
              updatedDoc.doc_type_slug = event.doc_type_slug || undefined;
              updatedDoc.confidence = event.confidence || undefined;
              updatedDoc.metadata = event.metadata || undefined;
              updatedDoc.risk_score = event.risk_score || undefined;
              updatedDoc.status = 'done';
              updatedDoc.progress = 100;
            } else if (event.type === 'error') {
              updatedDoc.status = 'error';
              updatedDoc.error_message = event.error_message || 'Unknown error';
            }

            onDocumentUpdate(updatedDoc);
          },
          () => {
            // Handle connection error - could implement fallback polling here
            console.warn(`SSE connection lost for document ${doc.id}`);
          }
        );

        setActiveEventSources(prev => new Map(prev.set(doc.id, eventSource)));
      }
    });

    // Clean up event sources for completed documents
    const completedDocs = displayDocuments.filter(doc =>
      doc.status === 'done' || doc.status === 'error'
    );

    completedDocs.forEach(doc => {
      const eventSource = activeEventSources.get(doc.id);
      if (eventSource) {
        eventSource.close();
        setActiveEventSources(prev => {
          const newMap = new Map(prev);
          newMap.delete(doc.id);
          return newMap;
        });
      }
    });

    // Cleanup on unmount
    return () => {
      activeEventSources.forEach(eventSource => eventSource.close());
    };
  }, [displayDocuments, activeEventSources, onDocumentUpdate]);

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

  const getMimeTypeIcon = (mimeType: string) => {
    return mimeTypeIcons[mimeType as keyof typeof mimeTypeIcons] || faFile;
  };

  const handleDocumentClick = (document: Document) => {
    setSelectedDocumentId(document.id);
    setIsModalOpen(true);
  };

  const handleReanalyze = (documentId: number, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent modal from opening
    analyzeMutation.mutate(documentId);
  };

  if (isLoading && displayDocuments.length === 0) {
    return (
      <div className="text-center py-12">
        <FontAwesomeIcon icon={faSpinner} className="text-white/40 text-4xl mb-4 animate-spin" />
        <p className="text-white/90">Loading documents...</p>
      </div>
    );
  }

  if (displayDocuments.length === 0) {
    return (
      <div className="text-center py-12">
        <FontAwesomeIcon icon={faFile} className="text-white/40 text-4xl mb-4" />
        <p className="text-white/90">
          {subjectId ? 'No documents found for this subject' : 'No documents uploaded yet'}
        </p>
      </div>
    );
  }

  return (
    <>
      {/* Queue Status Header */}
      {queueStatus && queueStatus.queue_size > 0 && (
        <div className="glass-card p-4">
          <div className="flex items-center space-x-3">
            <FontAwesomeIcon
              icon={faSpinner}
              className={`text-blue-400 w-5 h-5 ${queueStatus.is_running ? 'animate-spin' : ''}`}
            />
            <div>
              <p className="text-white font-medium">
                {queueStatus.queue_size} document{queueStatus.queue_size !== 1 ? 'en' : ''} in queue
              </p>
              <p className="text-white/60 text-sm">
                Processing is {queueStatus.is_running ? 'active' : 'paused'}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-3">
        {displayDocuments.map((document) => (
          <div
            key={document.id}
            className="glass-card p-4 hover:bg-white/20 transition-colors cursor-pointer"
            onClick={() => handleDocumentClick(document)}
          >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3 flex-1 min-w-0">
              <FontAwesomeIcon
                icon={getMimeTypeIcon(document.mime_type)}
                className="text-white/70 w-5 h-5 flex-shrink-0"
              />

              <div className="flex-1 min-w-0">
                <h3 className="text-white font-medium truncate">
                  {document.original_filename}
                </h3>
                <div className="flex flex-wrap items-center gap-3 text-sm text-white/85">
                  <span>{formatFileSize(document.size_bytes)}</span>
                  <span>{formatDate(document.created_at)}</span>
                  {document.subject_name && (
                    <span className="text-white/80 font-medium">
                      {document.subject_name} ({document.subject_context})
                    </span>
                  )}
                  {document.doc_type_slug && (
                    <span className="text-emerald-400 font-medium">
                      {formatDocumentTypeName(document.doc_type_slug)}
                      {document.doc_type_confidence && (
                        <span className="text-white/80">
                          ({Math.round(document.doc_type_confidence * 100)}%)
                        </span>
                      )}
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {/* Status and Progress */}
              <div className="flex items-center space-x-2">
                <FontAwesomeIcon
                  icon={statusIcons[document.status]}
                  className={`w-4 h-4 ${statusColors[document.status]}`}
                />
                <span className="text-white/80 text-sm capitalize">
                  {document.status === 'queued'
                    ? `Queue position: ${getQueuePosition(document)}`
                    : document.status
                  }
                </span>
                {document.stage && document.status === 'processing' && (
                  <span className="text-white/80 text-xs">
                    ({stageDescriptions[document.stage as keyof typeof stageDescriptions] || document.stage.replace('_', ' ')})
                  </span>
                )}
              </div>

              {/* Progress Bar */}
              {document.status === 'processing' && (
                <div className="w-24 bg-white/20 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${document.progress}%` }}
                  />
                </div>
              )}

              {document.status === 'queued' && (
                <div className="w-24 bg-white/20 rounded-full h-2 overflow-hidden">
                  <div className="bg-yellow-500 h-2 rounded-full animate-pulse"
                       style={{ width: '100%', animation: 'pulse 2s infinite' }} />
                </div>
              )}

              {/* Risk Score */}
              {document.risk_score !== null && (
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  document.risk_score >= 70 ? 'bg-red-500/20 text-red-400' :
                  document.risk_score >= 40 ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-green-500/20 text-green-400'
                }`}>
                  Risk: {document.risk_score}
                </div>
              )}

              {/* Actions */}
              <div className="flex space-x-1">
                <button
                  onClick={() => handleDocumentClick(document)}
                  className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                  title="View details"
                >
                  <FontAwesomeIcon icon={faEye} className="w-4 h-4" />
                </button>

                {document.status === 'done' && (
                  <button
                    onClick={(e) => handleReanalyze(document.id, e)}
                    disabled={analyzeMutation.isLoading}
                    className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded-lg transition-colors disabled:opacity-50"
                    title="Re-run analysis"
                  >
                    <FontAwesomeIcon icon={faRedo} className="w-4 h-4" />
                  </button>
                )}

                {/* Delete button for all documents */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setConfirmDelete({ document, isOpen: true });
                  }}
                  disabled={deleteMutation.isLoading}
                  className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50"
                  title="Delete document"
                >
                  <FontAwesomeIcon icon={faTrash} className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Error Message */}
          {document.error_message && (
            <div className="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-red-400 text-sm">{document.error_message}</p>
            </div>
          )}
        </div>
      ))}
      </div>

      {/* Delete Confirmation Modal */}
      {confirmDelete?.isOpen && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-card max-w-md w-full p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-red-500/20 rounded-full flex items-center justify-center">
                  <FontAwesomeIcon icon={faTrash} className="text-red-400 w-5 h-5" />
                </div>
                <h3 className="text-white text-lg font-semibold">Document verwijderen</h3>
              </div>
              <button
                onClick={() => setConfirmDelete(null)}
                className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
              </button>
            </div>

            <div className="mb-6">
              <p className="text-white/80 mb-2">
                Weet je zeker dat je dit document wilt verwijderen?
              </p>
              <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                <p className="text-white font-medium truncate">
                  {confirmDelete.document.original_filename}
                </p>
                <p className="text-white/60 text-sm">
                  {confirmDelete.document.status === 'queued' ? 'Uit wachtrij verwijderen' : 'Kapot document verwijderen'}
                </p>
              </div>
              <p className="text-white/60 text-sm mt-3">
                Deze actie kan niet ongedaan worden gemaakt.
              </p>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={() => setConfirmDelete(null)}
                className="flex-1 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
                disabled={deleteMutation.isLoading}
              >
                Annuleren
              </button>
              <button
                onClick={() => {
                  deleteMutation.mutate(confirmDelete.document.id);
                  setConfirmDelete(null);
                }}
                disabled={deleteMutation.isLoading}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {deleteMutation.isLoading ? (
                  <>
                    <FontAwesomeIcon icon={faSpinner} className="w-4 h-4 animate-spin mr-2" />
                    Verwijderen...
                  </>
                ) : (
                  'Verwijderen'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Document Detail Modal */}
      <DocumentDetailModal
        documentId={selectedDocumentId}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </>
  );
}