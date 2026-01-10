'use client';

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faTimes, faEye, faDownload, faRedo, faFile, faCheck,
  faExclamationTriangle, faInfoCircle, faHighlighter,
  faSearch, faCopy, faChevronDown, faChevronUp, faSpinner, faRobot
} from '@fortawesome/free-solid-svg-icons';
import {
  Document, getDocument, analyzeDocument, getDocumentArtifact, getDocumentArtifactText, getDocumentArtifactJson,
  RiskSignal, subscribeToDocumentEvents
} from '@/lib/api';

function formatDocumentTypeName(slug: string): string {
  if (slug === 'unknown') return 'Onbekend';
  return slug.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

interface DocumentDetailModalProps {
  documentId: number | null;
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'overview' | 'text' | 'metadata' | 'llm';

export function DocumentDetailModal({ documentId, isOpen, onClose }: DocumentDetailModalProps) {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [highlightSources, setHighlightSources] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const queryClient = useQueryClient();

  const { data: document, isLoading, refetch } = useQuery({
    queryKey: ['document', documentId],
    queryFn: () => getDocument(documentId!),
    enabled: !!documentId && isOpen,
  });

  const { data: extractedText, isLoading: textLoading } = useQuery({
    queryKey: ['document-text', documentId],
    queryFn: () => getDocumentArtifactText(documentId!, 'text/extracted.txt'),
    enabled: !!documentId && isOpen && activeTab === 'text' && document?.status === 'done',
  });

  const analyzeMutation = useMutation({
    mutationFn: () => analyzeDocument(documentId!),
    onSuccess: () => {
      refetch();
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.removeQueries({ queryKey: ['document-llm', documentId] });
    },
  });

  useEffect(() => {
    if (!documentId || !isOpen) return;
    const source = subscribeToDocumentEvents(documentId, () => {
      refetch();
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    });
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

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="glass-card max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10 flex-shrink-0">
          <div className="flex items-center space-x-3 min-w-0">
            <FontAwesomeIcon icon={faFile} className="text-white/70 w-5 h-5 flex-shrink-0" />
            <div className="min-w-0">
              <h2 className="text-white text-lg font-semibold truncate">
                {document?.original_filename || 'Loading...'}
              </h2>
              <p className="text-white/60 text-xs">Document #{documentId}</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 flex-shrink-0">
            <button
              onClick={() => analyzeMutation.mutate()}
              disabled={analyzeMutation.isLoading || document?.status === 'processing'}
              className="flex items-center space-x-2 px-3 py-1.5 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              <FontAwesomeIcon icon={faRedo} className="w-3 h-3" />
              <span>{analyzeMutation.isLoading ? 'Analyzing...' : 'Re-run'}</span>
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
            {/* Tabs */}
            <div className="flex border-b border-white/10 flex-shrink-0">
              {[
                { id: 'overview', label: 'Overview', icon: faInfoCircle },
                { id: 'text', label: 'Text', icon: faFile },
                { id: 'metadata', label: 'Data', icon: faSearch },
                { id: 'llm', label: 'LLM', icon: faRobot },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as TabType)}
                  className={`flex items-center space-x-2 px-4 py-2 text-sm font-medium transition-colors ${
                    activeTab === tab.id ? 'text-blue-400 border-b-2 border-blue-400' : 'text-white/60 hover:text-white'
                  }`}
                >
                  <FontAwesomeIcon icon={tab.icon} className="w-3 h-3" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto">
              {activeTab === 'overview' && <OverviewTab document={document} formatFileSize={formatFileSize} formatDate={formatDate} getStatusColor={getStatusColor} getRiskColor={getRiskColor} />}
              {activeTab === 'text' && <TextTab document={document} highlightSources={highlightSources} searchTerm={searchTerm} onHighlightToggle={() => setHighlightSources(!highlightSources)} onSearchChange={setSearchTerm} downloadArtifact={downloadArtifact} extractedText={extractedText} textLoading={textLoading} />}
              {activeTab === 'metadata' && <MetadataTab document={document} copyToClipboard={copyToClipboard} downloadArtifact={downloadArtifact} />}
              {activeTab === 'llm' && <LLMTab documentId={documentId} document={document} downloadArtifact={downloadArtifact} />}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function OverviewTab({ document, formatFileSize, formatDate, getStatusColor, getRiskColor }: {
  document: Document;
  formatFileSize: (bytes: number) => string;
  formatDate: (date: string) => string;
  getStatusColor: (status: string) => string;
  getRiskColor: (score: number) => string;
}) {
  return (
    <div className="p-4 space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* File Info */}
        <div className="bg-blue-500/10 rounded-lg p-4 border border-blue-500/20">
          <h3 className="text-white font-medium mb-3 flex items-center space-x-2">
            <FontAwesomeIcon icon={faFile} className="text-blue-400 w-4 h-4" />
            <span>File Info</span>
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between"><span className="text-white/60">Size:</span><span className="text-white">{formatFileSize(document.size_bytes)}</span></div>
            <div className="flex justify-between"><span className="text-white/60">Type:</span><span className="text-white text-xs">{document.mime_type}</span></div>
            <div className="flex justify-between"><span className="text-white/60">SHA256:</span><span className="text-white font-mono text-xs">{document.sha256?.substring(0, 12)}...</span></div>
            <div className="flex justify-between"><span className="text-white/60">Uploaded:</span><span className="text-white">{formatDate(document.created_at)}</span></div>
          </div>
        </div>

        {/* Status */}
        <div className="bg-green-500/10 rounded-lg p-4 border border-green-500/20">
          <h3 className="text-white font-medium mb-3 flex items-center space-x-2">
            <FontAwesomeIcon icon={faCheck} className="text-green-400 w-4 h-4" />
            <span>Status</span>
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between items-center">
              <span className="text-white/60">Status:</span>
              <span className={`capitalize font-medium px-2 py-0.5 rounded text-xs ${getStatusColor(document.status)}`}>{document.status}</span>
            </div>
            {document.stage && <div className="flex justify-between"><span className="text-white/60">Stage:</span><span className="text-white capitalize">{document.stage.replace('_', ' ')}</span></div>}
            <div className="flex justify-between items-center">
              <span className="text-white/60">Progress:</span>
              <div className="flex items-center space-x-2">
                <div className="w-16 bg-white/20 rounded-full h-1.5">
                  <div className="h-1.5 bg-green-400 rounded-full" style={{ width: `${document.progress}%` }} />
                </div>
                <span className="text-white text-xs">{document.progress}%</span>
              </div>
            </div>
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
                <p className="text-white/70 text-xs mt-2 bg-white/5 rounded p-2">{document.doc_type_rationale}</p>
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
                <span className={`text-2xl font-bold ${document.risk_score >= 70 ? 'text-red-400' : document.risk_score >= 40 ? 'text-yellow-400' : 'text-green-400'}`}>
                  {document.risk_score}
                </span>
                <div className="flex-1 bg-white/20 rounded-full h-2">
                  <div className={`h-2 rounded-full ${getRiskColor(document.risk_score)}`} style={{ width: `${document.risk_score}%` }} />
                </div>
              </div>
              {document.risk_signals_json?.length > 0 && (
                <div className="space-y-1 mt-2">
                  {document.risk_signals_json.slice(0, 2).map((signal: RiskSignal, i: number) => (
                    <div key={i} className="text-xs text-white/70 bg-white/5 rounded p-1.5">
                      <span className={signal.severity === 'high' ? 'text-red-400' : 'text-yellow-400'}>{signal.code}</span>: {signal.message}
                    </div>
                  ))}
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

function MetadataTab({ document, copyToClipboard, downloadArtifact }: {
  document: Document;
  copyToClipboard: (text: string) => void;
  downloadArtifact: (path: string, filename: string) => void;
}) {
  return (
    <div className="p-4 space-y-4">
      <div className="flex space-x-2">
        <button onClick={() => downloadArtifact('metadata/result.json', 'metadata.json')} className="flex items-center space-x-1 px-2 py-1 bg-white/10 text-white text-xs rounded hover:bg-white/20">
          <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
          <span>Metadata</span>
        </button>
        <button onClick={() => downloadArtifact('metadata/evidence.json', 'evidence.json')} className="flex items-center space-x-1 px-2 py-1 bg-white/10 text-white text-xs rounded hover:bg-white/20">
          <FontAwesomeIcon icon={faDownload} className="w-3 h-3" />
          <span>Evidence</span>
        </button>
      </div>

      {document.metadata_json && Object.keys(document.metadata_json).length > 0 ? (
        <div className="grid gap-3">
          {Object.entries(document.metadata_json).map(([key, value]) => (
            <div key={key} className="bg-purple-500/10 rounded-lg p-3 border border-purple-500/20">
              <div className="flex items-center justify-between mb-1">
                <span className="text-white/70 text-xs font-medium uppercase">{key.replace(/_/g, ' ')}</span>
                <button onClick={() => copyToClipboard(String(value))} className="text-purple-400 hover:text-purple-300">
                  <FontAwesomeIcon icon={faCopy} className="w-3 h-3" />
                </button>
              </div>
              <p className="text-white font-mono text-sm break-all">{String(value)}</p>
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
            <p className="text-white/60 text-xs">{document.stage?.replace('_', ' ')}</p>
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
            <CollapsibleBlock id="class-response" title="Response" content={data.classification.response} downloadPath="llm/classification_response.txt" downloadName="classification_response.txt" />
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
            <CollapsibleBlock id="ext-response" title="Response" content={data.extraction.response} downloadPath="llm/extraction_response.txt" downloadName="extraction_response.txt" />
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
