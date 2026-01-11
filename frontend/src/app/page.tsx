'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faCloudUploadAlt,
  faCheckCircle,
  faExclamationTriangle,
  faSpinner,
  faChartLine,
  faFolder,
  faArrowRight,
  faEye,
  faFileAlt,
  faTrash,
  faTimes,
  faRobot
} from '@fortawesome/free-solid-svg-icons';
import { SubjectSelector } from '@/components/SubjectSelector';
import { DocumentUploader } from '@/components/DocumentUploader';
import { DocumentDetailModal } from '@/components/DocumentDetailModal';
import { WizardStepper } from '@/components/WizardStepper';
import { useModel } from '@/context/ModelContext';
import {
  ClassifierStatus,
  Document,
  DocumentEvent,
  DocumentListResponse,
  Subject,
  getClassifierStatus,
  getDocument,
  listDocuments,
  listSubjects,
  subscribeToDocumentEvents,
  deleteDocument,
} from '@/lib/api';

export default function Dashboard() {
  const queryClient = useQueryClient();
  const [selectedSubject, setSelectedSubject] = useState<Subject | null>(null);
  const [activeDocumentId, setActiveDocumentId] = useState<number | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [detailDocumentId, setDetailDocumentId] = useState<number | null>(null);
  const [celebrateDone, setCelebrateDone] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<{ document: Document; isOpen: boolean } | null>(null);
  const [mounted, setMounted] = useState(false);
  
  // Use global model context for classification
  const { selectedModel: classifyModelName } = useModel();
  
  const step1Ref = useRef<HTMLDivElement>(null);

  // Set mounted state after first client render to prevent hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  const activeStep = useMemo(() => {
    if (!selectedSubject) return 1;
    if (!activeDocumentId) return 2;
    return 3;
  }, [selectedSubject, activeDocumentId]);

  const { data: subjects } = useQuery({
    queryKey: ['subjects'],
    queryFn: () => listSubjects(),
    refetchInterval: 10000,
  });

  const { data: recentDocs } = useQuery({
    queryKey: ['documents-recent'],
    queryFn: async () => listDocuments(undefined, undefined, 5, 0),
    refetchInterval: 5000,
  });

  const { data: classifierStatus } = useQuery({
    queryKey: ['classifier-status'],
    queryFn: getClassifierStatus,
    refetchInterval: (query) => {
      const status = query.state.data as ClassifierStatus | undefined;
      return status?.running ? 1500 : 15000;
    },
  });


  const deleteMutation = useMutation({
    mutationFn: (documentId: number) => deleteDocument(documentId),
    onSuccess: (_, documentId) => {
      queryClient.invalidateQueries({ queryKey: ['documents-recent'] });
      queryClient.setQueryData(['documents-recent'], (old: DocumentListResponse | undefined) => {
        if (!old) return old;
        const filteredDocs = old.documents?.filter((d) => d.id !== documentId) || [];
        return {
          ...old,
          documents: filteredDocs,
          total: filteredDocs.length,
        };
      });
    },
  });

  const { data: activeDocument } = useQuery({
    queryKey: ['document', activeDocumentId],
    queryFn: () => getDocument(activeDocumentId!),
    enabled: !!activeDocumentId,
    refetchInterval: (query) => {
      const doc = query.state.data as Document | undefined;
      return doc && (doc.status === 'processing' || doc.status === 'queued') ? 2000 : false;
    },
  });

  const handleSubjectChange = (subject: Subject | null) => {
    setSelectedSubject(subject);
    setActiveDocumentId(null);
    setCelebrateDone(false);
  };

  const handleDocumentUploaded = (document: Document) => {
    setActiveDocumentId(document.id);
    queryClient.setQueryData(['document', document.id], document);
    queryClient.setQueryData(['documents-recent'], (old: DocumentListResponse | undefined) => {
      if (!old) return { documents: [document], total: 1 };
      const exists = old.documents?.some((d) => d.id === document.id);
      if (exists) return old;
      return {
        ...old,
        documents: [document, ...(old.documents || [])].slice(0, 5),
        total: (old.total || old.documents?.length || 0) + 1,
      };
    });
  };

  useEffect(() => {
    if (!activeDocumentId) return;

    const source = subscribeToDocumentEvents(
      activeDocumentId,
      (event: DocumentEvent) => {
        // Update the active document cache
        queryClient.setQueryData(['document', activeDocumentId], (old: Document | undefined) => {
          if (!old) return old;

          const updated = { ...old };

          if (event.type === 'status') {
            updated.status = event.status as Document['status'];
            updated.stage = event.stage || undefined;
            updated.progress = event.progress ?? old.progress;
            updated.updated_at = event.updated_at || old.updated_at;
            // Classification result can come with status update (after classification stage)
            if (event.doc_type_slug != null) updated.doc_type_slug = event.doc_type_slug;
            if (event.confidence != null) updated.doc_type_confidence = event.confidence;
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

        // Also update the recent docs list in-memory for immediate UI feedback
        queryClient.setQueryData(['documents-recent'], (old: DocumentListResponse | undefined) => {
          if (!old?.documents) return old;
          return {
            ...old,
            documents: old.documents.map((doc) => {
              if (doc.id !== activeDocumentId) return doc;
              const updated = { ...doc };
              if (event.type === 'status') {
                updated.status = event.status as Document['status'];
                updated.stage = event.stage || undefined;
                updated.progress = event.progress ?? doc.progress;
                if (event.doc_type_slug != null) updated.doc_type_slug = event.doc_type_slug;
                if (event.confidence != null) updated.doc_type_confidence = event.confidence;
              } else if (event.type === 'result') {
                if (event.doc_type_slug != null) updated.doc_type_slug = event.doc_type_slug;
                if (event.confidence != null) updated.doc_type_confidence = event.confidence;
                updated.status = 'done';
                updated.progress = 100;
              } else if (event.type === 'error') {
                updated.status = 'error';
              }
              return updated;
            }),
          };
        });
        if (event.type === 'result') {
          // Ensure we pull the final DB state (doc_type, metadata) even if SSE is missing fields.
          queryClient.invalidateQueries({ queryKey: ['document', activeDocumentId] });
        }
      },
      () => {
        // SSE may drop; polling keeps the UI reasonably fresh.
      }
    );

    return () => source.close();
  }, [activeDocumentId, queryClient]);

  useEffect(() => {
    if (!activeDocument) return;
    if (activeDocument.status === 'done') {
      setCelebrateDone(true);
      const t = window.setTimeout(() => setCelebrateDone(false), 1200);
      return () => window.clearTimeout(t);
    }
    return;
  }, [activeDocument?.status]);

  const docs = recentDocs?.documents || [];
  const stats = {
    totalSubjects: subjects?.length || 0,
    totalDocuments: (recentDocs?.total ?? docs.length) || 0,
  };

  const openDetail = (documentId: number) => {
    setDetailDocumentId(documentId);
    setIsDetailOpen(true);
  };

  const formatDocumentTypeName = (slug: string): string => {
    if (slug === 'unknown') return 'Onbekend';
    return slug.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
  };

  const stageLabel = (stage?: string, progress?: number): string => {
    if (!stage) return '';
    
    // Handle chunk stages: extracting_metadata_chunk_1_3 → Extractie (chunk 1/3)
    const chunkMatch = stage.match(/^extracting_metadata_chunk_(\d+)_(\d+)$/);
    if (chunkMatch) {
      const [, chunkNum, totalChunks] = chunkMatch;
      return `Extractie (chunk ${chunkNum}/${totalChunks})`;
    }
    
    // Handle merging stage
    if (stage === 'extracting_metadata_merging') {
      return 'Extractie (samenvoegen)';
    }
    
    // Handle chunk done stage: extracting_metadata_chunk_done_3 → Extractie (3 chunks voltooid)
    const chunkDoneMatch = stage.match(/^extracting_metadata_chunk_done_(\d+)$/);
    if (chunkDoneMatch) {
      const [, totalChunks] = chunkDoneMatch;
      return `Extractie (${totalChunks} chunks voltooid)`;
    }
    
    // Handle post-processing stages with chunk info: extracting_metadata_post_processing_chunks_3
    const postProcessingChunkMatch = stage.match(/^extracting_metadata_(post_processing|validating|saving|complete)_chunks_(\d+)$/);
    if (postProcessingChunkMatch) {
      const [, stageType, totalChunks] = postProcessingChunkMatch;
      const stageMap: Record<string, string> = {
        post_processing: 'verwerken',
        validating: 'valideren',
        saving: 'opslaan',
        complete: 'voltooid',
      };
      return `Extractie (${stageMap[stageType] || stageType}, ${totalChunks} chunks)`;
    }
    
    const map: Record<string, string> = {
      sniffing: 'Voorbereiden',
      extracting_text: 'OCR',
      classifying: 'Classificatie',
      extracting_metadata: 'Extractie',
      extracting_metadata_post_processing: 'Extractie (verwerken)',
      extracting_metadata_validating: 'Extractie (valideren)',
      extracting_metadata_saving: 'Extractie (opslaan)',
      extracting_metadata_complete: 'Extractie (voltooid)',
      risk_signals: 'Validatie',
    };
    return map[stage] || stage.replace(/_/g, ' ');
  };

  const statusLabel = (status: Document['status']): string => {
    switch (status) {
      case 'queued':
        return 'wachtrij';
      case 'processing':
        return 'bezig';
      case 'done':
        return 'klaar';
      case 'error':
        return 'fout';
      case 'pending':
      default:
        return 'pending';
    }
  };

  const pipelineIndex = (doc?: Document): number => {
    if (!doc) return 0;
    if (doc.status === 'done') return 4;
    if (doc.status === 'queued' || doc.status === 'pending') return 0;

    switch (doc.stage) {
      case 'sniffing':
      case 'extracting_text':
        return 0;
      case 'classifying':
        return 1;
      case 'extracting_metadata':
        return 2;
      case 'risk_signals':
        return 3;
      default:
        return 0;
    }
  };

  return (
    <div className="space-y-4 sm:space-y-6 lg:space-y-8">
      {/* Hero + Stepper */}
      <div className="step-card p-2.5 sm:p-5 lg:p-8 step-active hero-card">
        <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-2 sm:gap-5">
          <div className="min-w-0 flex-1">
            {/* Mobile: compact single line, Desktop: full title */}
            <h1 className="font-bold text-white whitespace-nowrap mb-0.5 sm:mb-1">
              <span className="sm:hidden text-sm">📄 Analyse in 3 stappen</span>
              <span className="hidden sm:inline text-2xl lg:text-3xl">Document analyse in <span className="font-extrabold drop-shadow-lg">3 stappen</span></span>
            </h1>
            <p className="text-white/60 text-[11px] sm:text-sm lg:text-base max-w-2xl leading-relaxed hidden sm:block">
              Selecteer een subject, upload je document en volg live de AI-analyse.
            </p>

            <div className="mt-2 sm:mt-3 flex flex-wrap gap-1 sm:gap-2">
              <div className="px-1.5 sm:px-3 py-0.5 sm:py-1.5 rounded-full bg-white/8 border border-white/12 text-white/80 text-[9px] sm:text-xs">
                <span className="text-white font-semibold sm:hidden">8</span>
                <span className="text-white font-semibold hidden sm:inline">{stats.totalSubjects}</span> <span className="text-white/60 hidden sm:inline">subjects</span>
                <span className="text-white/60 sm:hidden"> subjects</span>
              </div>
              <div className="px-1.5 sm:px-3 py-0.5 sm:py-1.5 rounded-full bg-white/8 border border-white/12 text-white/80 text-[9px] sm:text-xs">
                <span className="text-white font-semibold sm:hidden">15</span>
                <span className="text-white font-semibold hidden sm:inline">{stats.totalDocuments}</span> <span className="text-white/60 hidden sm:inline">docs</span>
                <span className="text-white/60 sm:hidden"> docs</span>
              </div>
              
              {/* Training Progress Indicator */}
              {classifierStatus?.running && (
                <div className="w-full mt-1.5 p-2 bg-blue-500/15 border border-blue-400/25 rounded-lg">
                  <div className="flex items-center gap-2 mb-1.5">
                    <FontAwesomeIcon icon={faSpinner} className="w-3 h-3 text-blue-400 animate-spin" />
                    <span className="text-blue-300 text-xs font-medium">
                      Training actief...
                    </span>
                  </div>
                  <div className="bg-white/10 rounded-full h-1.5 overflow-hidden">
                    <div className="h-1.5 bg-gradient-to-r from-blue-500 via-purple-500 to-blue-500 rounded-full animate-[shimmer_2s_ease-in-out_infinite]" style={{ width: '100%', backgroundSize: '200% 100%' }} />
                  </div>
                  <div className="mt-1 text-white/50 text-[10px]">
                    {classifierStatus?.started_at 
                      ? `Gestart: ${new Date(classifierStatus.started_at).toLocaleTimeString('nl-NL')}`
                      : 'Bezig met laden van documenten...'}
                  </div>
                </div>
              )}
            </div>

            {(classifierStatus?.last_error || classifierStatus?.last_summary) && !classifierStatus?.running && (
              <div className="mt-2 text-xs text-white/55">
                {classifierStatus?.last_error ? (
                  <span className="text-red-300">Training error: {classifierStatus.last_error}</span>
                ) : (
                  <div className="space-y-1">
                    <div>
                      Laatste training: {classifierStatus?.finished_at 
                        ? new Date(classifierStatus.finished_at).toLocaleString('nl-NL', { 
                            day: '2-digit', 
                            month: '2-digit', 
                            year: 'numeric',
                            hour: '2-digit', 
                            minute: '2-digit' 
                          })
                        : 'onbekend'}
                    </div>
                    {typeof (classifierStatus?.last_summary as any)?.examples === 'number' && (
                      <div>Training samples: {(classifierStatus?.last_summary as any)?.examples}</div>
                    )}
                    {Array.isArray((classifierStatus?.last_summary as any)?.labels) && (
                      <div>Document types: {((classifierStatus?.last_summary as any)?.labels as string[]).length}</div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="w-full lg:max-w-xl">
            <WizardStepper activeStep={activeStep} />
          </div>
        </div>
      </div>

      {/* 3 Step Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
        {/* Step 1 */}
        <div
          ref={step1Ref}
          className={`step-card p-3 sm:p-4 ${
            selectedSubject ? 'step-completed' : 'step-active'
          }`}
        >
          <div className="flex items-start justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg sm:rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shrink-0">
                <FontAwesomeIcon icon={faFolder} className="text-white text-sm sm:text-base" />
              </div>
              <div className="min-w-0">
                <div className="text-white font-semibold text-sm sm:text-base">Stap 1 · Subject</div>
                <div className="text-white/55 text-xs sm:text-sm">Kies context</div>
              </div>
            </div>
            {selectedSubject && (
              <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-[10px] sm:text-xs font-medium shrink-0">
                <FontAwesomeIcon icon={faCheckCircle} className="w-3 h-3" />
                <span className="hidden sm:inline">Geselecteerd</span>
              </span>
            )}
          </div>

          <SubjectSelector
            variant="wizard"
            selectedSubject={selectedSubject}
            onSubjectChange={handleSubjectChange}
          />

          {/* Model indicator - shows which model is active from nav */}
          {mounted && classifyModelName && (
            <div className="mt-3 pt-3 border-t border-white/10">
              <div className="flex items-center gap-2 text-xs">
                <FontAwesomeIcon icon={faRobot} className="w-3 h-3 text-purple-400" />
                <span className="text-white/50">Model:</span>
                <span className="text-purple-300 font-medium">{classifyModelName}</span>
              </div>
            </div>
          )}
        </div>

        {/* Step 2 */}
        <div
          className={`step-card p-3 sm:p-4 ${
            !selectedSubject ? 'step-disabled' : activeStep === 2 ? 'step-active' : 'step-completed'
          }`}
        >
          <div className="flex items-start justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg sm:rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center shrink-0">
                <FontAwesomeIcon icon={faCloudUploadAlt} className="text-white text-sm sm:text-base" />
              </div>
              <div className="min-w-0">
                <div className="text-white font-semibold text-sm sm:text-base">Stap 2 · Upload</div>
                <div className="text-white/55 text-xs sm:text-sm">Sleep of klik</div>
              </div>
            </div>
            {activeStep > 2 && (
              <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-[10px] sm:text-xs font-medium shrink-0">
                <FontAwesomeIcon icon={faCheckCircle} className="w-3 h-3" />
                <span className="hidden sm:inline">Gestart</span>
              </span>
            )}
          </div>

          <DocumentUploader
            variant="wizard"
            selectedSubject={selectedSubject}
            onDocumentUploaded={handleDocumentUploaded}
            disabled={!selectedSubject}
            onDisabledClick={() => step1Ref.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
            selectedModel={classifyModelName}
          />
        </div>

        {/* Step 3 */}
        <div
          className={`step-card p-3 sm:p-4 md:col-span-2 lg:col-span-1 ${
            activeDocumentId ? 'step-active' : 'step-disabled'
          } ${celebrateDone ? 'success-pulse' : ''} ${activeDocument?.status === 'done' ? 'step-completed' : ''}`}
        >
          <div className="flex items-start justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg sm:rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center shrink-0">
                <FontAwesomeIcon icon={faChartLine} className="text-white text-sm sm:text-base" />
              </div>
              <div className="min-w-0">
                <div className="text-white font-semibold text-sm sm:text-base">Stap 3 · Analyse</div>
                <div className="text-white/55 text-xs sm:text-sm">Live resultaat</div>
              </div>
            </div>
            {activeDocument?.status === 'done' && (
              <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs font-medium">
                <FontAwesomeIcon icon={faCheckCircle} className="w-4 h-4" />
                Klaar
              </span>
            )}
            {activeDocument?.status === 'error' && (
              <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-500/10 border border-red-500/20 text-red-300 text-xs font-medium">
                <FontAwesomeIcon icon={faExclamationTriangle} className="w-4 h-4" />
                Fout
              </span>
            )}
          </div>

          {!activeDocumentId && (
            <div className="text-white/60 text-sm">
              Upload een document om hier live de analyse te volgen.
            </div>
          )}

          {activeDocumentId && (
            <div className="space-y-4">
              <div className="bg-white/5 border border-white/10 rounded-xl p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-white font-medium truncate">
                      {activeDocument?.original_filename || `Document #${activeDocumentId}`}
                    </div>
                    <div className="text-white/50 text-xs mt-0.5">
                      {activeDocument?.status === 'queued' ? 'In wachtrij' : activeDocument?.status === 'processing' ? `Bezig: ${stageLabel(activeDocument.stage, activeDocument.progress)}` : activeDocument?.status === 'done' ? 'Analyse afgerond' : activeDocument?.status === 'error' ? 'Analyse mislukt' : ''}
                    </div>
                  </div>

                  <button
                    onClick={() => openDetail(activeDocumentId)}
                    className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 hover:border-white/20 transition-all text-white/80 hover:text-white text-xs sm:text-sm cursor-pointer shrink-0"
                  >
                    <FontAwesomeIcon icon={faEye} className="w-4 h-4" />
                    Bekijk
                  </button>
                </div>

                {(activeDocument?.status === 'processing' || activeDocument?.status === 'queued') && (
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-xs text-white/60 mb-1">
                      <span>Voortgang</span>
                      <span>{activeDocument?.status === 'queued' ? 'Wachten…' : `${Math.round(activeDocument?.progress || 0)}%`}</span>
                    </div>
                    <div className="w-full bg-white/15 rounded-full h-2 overflow-hidden">
                      {activeDocument?.status === 'queued' ? (
                        <div className="h-2 bg-yellow-500/70 rounded-full animate-pulse w-full" />
                      ) : (
                        <div className="h-2 bg-blue-500 rounded-full transition-all duration-300" style={{ width: `${activeDocument?.progress || 0}%` }} />
                      )}
                    </div>
                  </div>
                )}
              </div>

              <PipelineMini currentIndex={pipelineIndex(activeDocument)} />

              {activeDocument?.status === 'done' && (
                <div className="grid gap-3">
                  <div className="bg-white/5 border border-white/10 rounded-xl p-3">
                    <div className="text-white/60 text-xs mb-1">Document type</div>
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-white font-medium truncate">
                        {activeDocument.doc_type_slug ? formatDocumentTypeName(activeDocument.doc_type_slug) : 'Onbekend'}
                      </div>
                      {activeDocument.doc_type_confidence != null && (
                        <div className="px-2 py-1 rounded-lg bg-purple-500/15 border border-purple-500/25 text-purple-200 text-xs font-semibold shrink-0">
                          {Math.round(activeDocument.doc_type_confidence * 100)}%
                        </div>
                      )}
                    </div>
                  </div>

                  {activeDocument.risk_score != null && (
                    <div className="bg-white/5 border border-white/10 rounded-xl p-3">
                      <div className="text-white/60 text-xs mb-1">Risico</div>
                      <div className="flex items-center gap-3">
                        <div className={`text-lg font-bold ${
                          (activeDocument.risk_score ?? 0) >= 70 ? 'text-red-300' :
                          (activeDocument.risk_score ?? 0) >= 40 ? 'text-yellow-300' :
                          'text-emerald-300'
                        }`}>
                          {activeDocument.risk_score}
                        </div>
                        <div className="flex-1 bg-white/15 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-2 rounded-full ${
                              (activeDocument.risk_score ?? 0) >= 70 ? 'bg-red-500' :
                              (activeDocument.risk_score ?? 0) >= 40 ? 'bg-yellow-500' :
                              'bg-emerald-500'
                            }`}
                            style={{ width: `${Math.min(100, Math.max(0, activeDocument.risk_score ?? 0))}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeDocument?.status === 'error' && activeDocument.error_message && (
                <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-red-200 text-sm">
                  {activeDocument.error_message}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Recent Documents */}
      <div className="glass-card p-2.5 sm:p-4 lg:p-6 border-2 border-white/20 bg-gradient-to-br from-white/10 to-white/5">
        <div className="flex items-center justify-between gap-2 mb-2 sm:mb-4">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            <div className="w-7 h-7 sm:w-10 sm:h-10 rounded-lg sm:rounded-xl bg-gradient-to-br from-indigo-500 to-blue-500 flex items-center justify-center shrink-0 shadow-lg">
              <FontAwesomeIcon icon={faFileAlt} className="text-white text-xs sm:text-base" />
            </div>
            <div className="min-w-0">
              <div className="text-white font-semibold text-xs sm:text-base">Recente docs</div>
              <div className="text-white/55 text-[10px] sm:text-sm">
                <span className="sm:hidden">Laatste 3</span>
                <span className="hidden sm:inline">Laatste 5</span>
              </div>
            </div>
          </div>
          <Link
            href="/documents"
            className="inline-flex items-center gap-1 sm:gap-2 text-white/70 hover:text-white text-[10px] sm:text-sm transition-colors shrink-0 cursor-pointer"
          >
            Alles
            <FontAwesomeIcon icon={faArrowRight} className="w-3 h-3" />
          </Link>
        </div>

        {docs.length === 0 ? (
          <div className="text-white/60 text-sm">Nog geen documenten.</div>
        ) : (
          <div className="divide-y divide-white/10">
            {docs.map((doc, index) => (
              <div
                key={doc.id}
                className={`w-full py-1.5 sm:py-3 flex items-center justify-between gap-2 sm:gap-3 hover:bg-white/5 rounded-lg transition-colors group cursor-pointer ${
                  index >= 3 ? 'hidden sm:flex' : ''
                }`}
              >
                <button
                  onClick={() => openDetail(doc.id)}
                  className="flex-1 text-left min-w-0 cursor-pointer flex items-center gap-2"
                >
                  {/* Status icon - mobile only, compact */}
                  <FontAwesomeIcon
                    icon={doc.status === 'done' ? faCheckCircle : (doc.status === 'processing' || doc.status === 'queued') ? faSpinner : faExclamationTriangle}
                    className={`w-3 h-3 sm:hidden shrink-0 ${
                      doc.status === 'done' ? 'text-emerald-400' 
                      : (doc.status === 'processing' || doc.status === 'queued') ? 'text-blue-400 animate-spin'
                      : 'text-red-400'
                    }`}
                  />
                  {/* Filename + inline badges on mobile */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
                      <span className="text-white font-medium truncate text-xs sm:text-base max-w-[120px] sm:max-w-none">{doc.original_filename}</span>
                      {/* Compact type badge with percentage - mobile */}
                      {doc.doc_type_slug && (
                        <span className="sm:hidden text-purple-300 text-[9px] font-medium shrink-0 flex items-center gap-0.5">
                          {formatDocumentTypeName(doc.doc_type_slug).slice(0, 6)}
                          {doc.doc_type_confidence != null && (
                            <span className="text-purple-400">{Math.round(doc.doc_type_confidence * 100)}%</span>
                          )}
                        </span>
                      )}
                    </div>
                    {/* Full badges - desktop only */}
                    <div className="hidden sm:flex flex-wrap items-center gap-2 text-xs mt-1">
                      {doc.doc_type_slug ? (
                        <div className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/30">
                          <span className="text-white font-semibold">{formatDocumentTypeName(doc.doc_type_slug)}</span>
                          {doc.doc_type_confidence != null && (
                            <span className="text-purple-200 font-bold">
                              {Math.round(doc.doc_type_confidence * 100)}%
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-white/50">Geen type</span>
                      )}
                      
                      {doc.risk_score != null && (
                        <div className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border font-semibold ${
                          (doc.risk_score ?? 0) >= 70
                            ? 'bg-red-500/20 border-red-500/30 text-red-200'
                            : (doc.risk_score ?? 0) >= 40
                              ? 'bg-yellow-500/20 border-yellow-500/30 text-yellow-200'
                              : 'bg-emerald-500/20 border-emerald-500/30 text-emerald-200'
                        }`}>
                          <span>Fraude:</span>
                          <span className="font-bold">{doc.risk_score}%</span>
                        </div>
                      )}

                      {doc.subject_name && (
                        <span className="text-white/50">· {doc.subject_name}</span>
                      )}
                    </div>
                  </div>
                </button>

                {/* Right side - compact on mobile */}
                <div className="flex items-center gap-1 sm:gap-2 shrink-0">
                  {/* Risk score mini badge - mobile only */}
                  {doc.risk_score != null && (
                    <span className={`sm:hidden text-[9px] font-bold px-1 py-0.5 rounded ${
                      (doc.risk_score ?? 0) >= 70 ? 'text-red-300 bg-red-500/20'
                      : (doc.risk_score ?? 0) >= 40 ? 'text-yellow-300 bg-yellow-500/20'
                      : 'text-emerald-300 bg-emerald-500/20'
                    }`}>
                      {doc.risk_score}%
                    </span>
                  )}
                  
                  {/* Desktop status badges */}
                  <div className="hidden sm:flex items-center gap-2">
                    {(doc.status === 'processing' || doc.status === 'queued') ? (
                      <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-blue-500/20 border border-blue-500/30 animate-pulse">
                        <FontAwesomeIcon icon={faSpinner} className="w-3 h-3 text-blue-300 animate-spin" />
                        <span className="text-blue-200 text-xs font-medium max-w-[120px] truncate">
                          {doc.stage || (doc.status === 'queued' ? 'Wachtrij' : 'Bezig...')}
                        </span>
                        {doc.progress != null && doc.progress > 0 && (
                          <span className="text-blue-300 text-xs font-bold">{doc.progress}%</span>
                        )}
                      </div>
                    ) : (
                      <>
                        <span className={`px-2 py-1 rounded-lg text-xs font-semibold border ${
                          doc.status === 'done'
                            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-200'
                            : 'bg-red-500/10 border-red-500/20 text-red-200'
                        }`}>
                          {statusLabel(doc.status)}
                        </span>
                        <FontAwesomeIcon
                          icon={doc.status === 'done' ? faCheckCircle : faExclamationTriangle}
                          className={`w-4 h-4 ${
                            doc.status === 'done' ? 'text-emerald-300' : 'text-red-300'
                          }`}
                        />
                      </>
                    )}
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setConfirmDelete({ document: doc, isOpen: true });
                    }}
                    disabled={deleteMutation.isPending}
                    className="p-1 sm:p-1.5 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50 cursor-pointer opacity-0 group-hover:opacity-100 sm:opacity-0"
                    title="Verwijder document"
                  >
                    <FontAwesomeIcon icon={faTrash} className="w-3 h-3 sm:w-4 sm:h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <DocumentDetailModal
        documentId={detailDocumentId}
        isOpen={isDetailOpen}
        onClose={() => setIsDetailOpen(false)}
      />

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
                  {confirmDelete.document.status === 'queued' ? 'Uit wachtrij verwijderen' : 'Document verwijderen'}
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
                disabled={deleteMutation.isPending}
              >
                Annuleren
              </button>
              <button
                onClick={() => {
                  deleteMutation.mutate(confirmDelete.document.id);
                  setConfirmDelete(null);
                }}
                disabled={deleteMutation.isPending}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {deleteMutation.isPending ? (
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
    </div>
  );
}

function PipelineMini({ currentIndex }: { currentIndex: number }) {
  const steps = [
    { title: 'OCR', subtitle: 'Tekst' },
    { title: 'Classificatie', subtitle: 'Type' },
    { title: 'Extractie', subtitle: 'Velden' },
    { title: 'Validatie', subtitle: 'Controle' },
  ] as const;
  const completedThrough = Math.max(0, Math.min(4, currentIndex));
  const activeStep = completedThrough >= 4 ? null : completedThrough + 1;

  return (
    <div className="grid grid-cols-4 gap-1.5 sm:gap-2">
      {steps.map((s, i) => {
        const idx = i + 1;
        const isCompleted = idx <= completedThrough;
        const isActive = activeStep === idx;
        const circleClasses = isCompleted
          ? 'bg-emerald-500/20 border-emerald-500/25 text-emerald-200'
          : isActive
            ? 'bg-blue-500/20 border-blue-500/25 text-blue-200'
            : 'bg-white/5 border-white/10 text-white/55';

        return (
          <div key={s.title} className="flex flex-col items-center gap-1.5 min-w-0">
            <div className={`w-7 h-7 sm:w-8 sm:h-8 rounded-full border flex items-center justify-center text-[11px] sm:text-xs font-bold shrink-0 ${circleClasses}`}>
              {idx}
            </div>
            <div className="text-center min-w-0 w-full">
              <div className={`text-[10px] sm:text-xs font-semibold ${isCompleted ? 'text-white/90' : 'text-white/70'} break-words`}>
                {s.title}
              </div>
              <div className="text-[9px] sm:text-[10px] text-white/45 break-words">
                {s.subtitle}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
