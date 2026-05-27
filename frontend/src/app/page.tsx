'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faCheckCircle, faExclamationTriangle, faSpinner, faFileAlt, faTrash,
  faEye, faShieldAlt, faArrowRight, faArrowLeft, faRobot, faTag, faList, faBolt,
  faCheck, faCloudUploadAlt,
} from '@fortawesome/free-solid-svg-icons';
import { SubjectSelector } from '@/components/SubjectSelector';
import { DocumentUploader } from '@/components/DocumentUploader';
import { DocumentDetailModal } from '@/components/DocumentDetailModal';
import { useModel } from '@/context/ModelContext';
import {
  Document, DocumentEvent, DocumentListResponse, Subject,
  FraudReport, AdviceCard,
  getDocument, listDocuments, subscribeToDocumentEvents,
  deleteDocument, getFraudAnalysis, getClassifierStatus, ClassifierStatus,
} from '@/lib/api';

// ─── Utilities ────────────────────────────────────────────────────────────────

function stageLabel(stage?: string): string {
  if (!stage) return '';
  const m = stage.match(/^extracting_metadata_(?:selecting|chunk)_(\d+)_(?:of_)?(\d+)/);
  if (m) return `Extractie (${m[1]}/${m[2]})`;
  const mChunk = stage.match(/^extracting_metadata_llm_chunk_(\d+)_of_(\d+)$/);
  if (mChunk) return `Extractie (${mChunk[1]}/${mChunk[2]})`;
  const mSel = stage.match(/^extracting_metadata_llm_selected_(\d+)_of_\d+_chunks$/);
  if (mSel) return `Extractie: ${mSel[1]} chunks geselecteerd`;
  const map: Record<string, string> = {
    sniffing: 'Bestand inspecteren',
    extracting_text: 'Tekst extraheren (OCR)',
    classifying: 'Document classificeren',
    extracting_metadata: 'Velden extraheren',
    extracting_metadata_post_processing: 'Velden verwerken',
    extracting_metadata_merging: 'Resultaten samenvoegen',
    extracting_metadata_validating: 'Velden valideren',
    extracting_metadata_saving: 'Opslaan',
    extracting_metadata_complete: 'Extractie voltooid',
    risk_signals: 'Risico analyse',
  };
  return map[stage] || stage.replace(/_/g, ' ');
}

function pipelineStep(doc?: Document): number {
  if (!doc || doc.status === 'queued' || doc.status === 'pending') return 0;
  if (doc.status === 'done') return 5;
  const s = doc.stage || '';
  if (s.startsWith('sniffing') || s.startsWith('extracting_text')) return 1;
  if (s.startsWith('classifying')) return 2;
  if (s.startsWith('extracting_metadata')) return 3;
  if (s.startsWith('risk_signals')) return 4;
  return 1;
}

const fmtType = (slug: string) =>
  slug === 'unknown' ? 'Onbekend' : slug.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

const fmtLabel = (key: string) =>
  key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

const fmtVal = (v: unknown, key?: string): string => {
  if (v == null) return '—';
  if (typeof v === 'boolean') return v ? 'Ja' : 'Nee';
  if (typeof v === 'number') {
    const k = (key ?? '').toLowerCase();
    if (/amount|prijs|waarde|bedrag|kosten|price|value|totaal|hypotheek|lening/.test(k)) {
      return new Intl.NumberFormat('nl-NL', { style: 'currency', currency: 'EUR', maximumFractionDigits: 2 }).format(v);
    }
    return new Intl.NumberFormat('nl-NL').format(v);
  }
  if (typeof v === 'string' && /^\d+(\.\d+)?$/.test(v.trim())) {
    const k = (key ?? '').toLowerCase();
    if (/amount|prijs|waarde|bedrag|kosten|price|value|totaal|hypotheek|lening/.test(k)) {
      return new Intl.NumberFormat('nl-NL', { style: 'currency', currency: 'EUR', maximumFractionDigits: 2 }).format(parseFloat(v));
    }
  }
  return String(v);
};

function riskStyle(score?: number | null) {
  if (score == null) return { text: 'text-slate-400', bg: 'bg-slate-50 border-slate-200', label: '—' };
  if (score >= 70) return { text: 'text-red-600', bg: 'bg-red-50 border-red-200', label: 'Hoog risico' };
  if (score >= 40) return { text: 'text-amber-600', bg: 'bg-amber-50 border-amber-200', label: 'Gemiddeld' };
  return { text: 'text-emerald-600', bg: 'bg-emerald-50 border-emerald-200', label: 'Laag risico' };
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const queryClient = useQueryClient();
  const { selectedModel } = useModel();
  const [selectedSubject, setSelectedSubject] = useState<Subject | null>(null);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [detailId, setDetailId] = useState<number | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [detailInitialTab, setDetailInitialTab] = useState<'overview' | 'text' | 'metadata' | 'llm' | 'forensics' | undefined>(undefined);
  const [confirmDelete, setConfirmDelete] = useState<Document | null>(null);
  const [blinking, setBlinking] = useState<Set<number>>(new Set());
  const prevStatuses = useRef<Map<number, string>>(new Map());

  const { data: recentDocs } = useQuery({
    queryKey: ['documents-recent'],
    queryFn: () => listDocuments(undefined, undefined, 20, 0),
    refetchInterval: (query) => {
      const data = (query.state.data as DocumentListResponse | undefined);
      const hasProcessing = data?.documents.some(d => d.status === 'processing' || d.status === 'queued');
      return hasProcessing ? false : 5000;
    },
    structuralSharing: (old: unknown, n: unknown) => {
      const o = old as DocumentListResponse | undefined;
      const nw = n as DocumentListResponse;
      if (!o) return nw;
      const map = new Map(o.documents.map(d => [d.id, d]));
      return {
        ...nw,
        documents: nw.documents.map(d => {
          const prev = map.get(d.id);
          if (!prev || d.status === 'done' || d.status === 'error') return d;
          const isProcessing = d.status === 'processing' || d.status === 'queued';
          const keepProgress = isProcessing && prev.progress != null && prev.progress > (d.progress ?? 0);
          return {
            ...d,
            doc_type_slug: d.doc_type_slug ?? prev.doc_type_slug,
            doc_type_confidence: d.doc_type_confidence ?? prev.doc_type_confidence,
            progress: keepProgress ? prev.progress : d.progress,
            stage: keepProgress ? prev.stage : d.stage,
          };
        }),
      };
    },
  });

  const { data: activeDoc } = useQuery({
    queryKey: ['document', activeId],
    queryFn: () => getDocument(activeId!),
    enabled: !!activeId,
    // SSE provides real-time updates during processing; DB polling at 2s overwrites SSE progress
    // because per-page progress is only sent via SSE, not written to DB.
    refetchInterval: q => {
      const d = q.state.data as Document | undefined;
      return d && (d.status === 'processing' || d.status === 'queued') ? false : 30000;
    },
    refetchOnWindowFocus: q => {
      const d = (q as any).state.data as Document | undefined;
      return !(d?.status === 'processing' || d?.status === 'queued');
    },
  });

  const { data: fraud } = useQuery({
    queryKey: ['fraud-analysis', activeId],
    queryFn: () => getFraudAnalysis(activeId!),
    enabled: !!activeId && activeDoc?.status === 'done',
    staleTime: Infinity,
    gcTime: Infinity,
  });

  const { data: trainStatus } = useQuery({
    queryKey: ['classifier-status'],
    queryFn: getClassifierStatus,
    refetchInterval: q => ((q.state.data as ClassifierStatus | undefined)?.running ? 1000 : 30000),
  });

  const del = useMutation({
    mutationFn: (id: number) => deleteDocument(id),
    onSuccess: (_, id) => {
      queryClient.setQueryData(['documents-recent'], (o?: DocumentListResponse) => {
        if (!o) return o;
        const docs = o.documents.filter(d => d.id !== id);
        return { ...o, documents: docs, total: docs.length };
      });
      if (activeId === id) setActiveId(null);
    },
  });

  // Subscribe to SSE for ALL processing/queued docs so all update in real-time
  useEffect(() => {
    if (!recentDocs?.documents) return;
    const processing = recentDocs.documents.filter(
      d => d.status === 'processing' || d.status === 'queued'
    );
    if (processing.length === 0) return;

    const sources = processing.map(doc => {
      const docId = doc.id;
      const makeUpdater = (ev: DocumentEvent) => (d: Document): Document => {
        const u = { ...d };
        if (ev.type === 'status') {
          u.status = ev.status as Document['status'];
          u.stage = ev.stage || undefined;
          // On rerun: status flips back to 'processing' while old progress may be 100.
          // Always take the event's progress when transitioning into processing so the
          // ring doesn't flash backwards from a completed run.
          u.progress = (ev.status === 'processing' && d.status !== 'processing')
            ? (ev.progress ?? 0)
            : (ev.progress ?? d.progress);
          if (ev.doc_type_slug != null) u.doc_type_slug = ev.doc_type_slug;
          if (ev.confidence != null) u.doc_type_confidence = ev.confidence;
        } else if (ev.type === 'result') {
          if (ev.doc_type_slug != null) u.doc_type_slug = ev.doc_type_slug;
          if (ev.confidence != null) u.doc_type_confidence = ev.confidence;
          if (ev.metadata != null) u.metadata_json = ev.metadata;
          if (ev.risk_score != null) u.risk_score = ev.risk_score;
          u.status = 'done';
          u.progress = 100;
          u.updated_at = new Date().toISOString();
        } else if (ev.type === 'error') {
          u.status = 'error';
          u.error_message = ev.error_message || 'Onbekende fout';
        }
        return u;
      };

      return subscribeToDocumentEvents(docId, (ev: DocumentEvent) => {
        const patch = makeUpdater(ev);
        queryClient.setQueryData(['document', docId], (o?: Document) => o ? patch(o) : o);
        queryClient.setQueryData(['documents-recent'], (o?: DocumentListResponse) => {
          if (!o?.documents) return o;
          return { ...o, documents: o.documents.map(d => d.id === docId ? patch(d) : d) };
        });
        if (ev.type === 'result') {
          queryClient.invalidateQueries({ queryKey: ['document', docId] });
          queryClient.invalidateQueries({ queryKey: ['fraud-analysis', docId] });
        }
      }, () => {});
    });

    return () => sources.forEach(s => s.close());
  }, [
    // Deps: re-subscribe when the set of processing doc IDs changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
    recentDocs?.documents.filter(d => d.status === 'processing' || d.status === 'queued').map(d => d.id).join(','),
    queryClient,
  ]);

  useEffect(() => {
    if (!recentDocs?.documents) return;
    const curr = new Map(recentDocs.documents.map(d => [d.id, d.status]));
    const newDone = recentDocs.documents
      .filter(d => d.status === 'done' && prevStatuses.current.has(d.id) && prevStatuses.current.get(d.id) !== 'done')
      .map(d => d.id);
    prevStatuses.current = curr;
    if (!newDone.length) return;
    setBlinking(p => { const s = new Set(p); newDone.forEach(id => s.add(id)); return s; });
    const t = setTimeout(() => setBlinking(p => { const s = new Set(p); newDone.forEach(id => s.delete(id)); return s; }), 3000);
    return () => clearTimeout(t);
  }, [recentDocs?.documents]);

  const handleUploaded = (doc: Document) => {
    setActiveId(doc.id);
    queryClient.setQueryData(['document', doc.id], doc);
    queryClient.setQueryData(['documents-recent'], (o?: DocumentListResponse) => {
      if (!o) return { documents: [doc], total: 1 };
      if (o.documents.some(d => d.id === doc.id)) return o;
      return { ...o, documents: [doc, ...o.documents].slice(0, 20), total: o.total + 1 };
    });
  };

  const docs = recentDocs?.documents || [];
  const step = pipelineStep(activeDoc);
  const openDetail = (id: number, tab?: 'overview' | 'text' | 'metadata' | 'llm' | 'forensics') => {
    setDetailId(id);
    setDetailInitialTab(tab);
    setIsDetailOpen(true);
  };

  return (
    <div
      className="-my-4 sm:-my-8 -mx-3 sm:-mx-4 lg:-mx-8 flex overflow-hidden h-[calc(100vh-3.5rem)] sm:h-[calc(100vh-4rem)] min-w-[320px]"
    >
      {/* ── SIDEBAR ──────────────────────────────────────────────────────────── */}
      <aside className="hidden md:flex w-72 lg:w-80 xl:w-[320px] shrink-0 flex-col border-r border-slate-200 overflow-hidden bg-white/70 backdrop-blur-sm">

        {/* Upload block */}
        <div className="p-4 space-y-4 border-b border-slate-200 shrink-0">
          {/* Step 1 */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-[#FFC1F3]/20 border border-[#FFC1F3]/40 flex items-center justify-center text-[#FFC1F3] text-[10px] font-bold shrink-0">1</span>
              <span className="text-slate-500 text-xs font-semibold">Kies referentie</span>
            </div>
            <p className="text-slate-500 text-[10px] pl-7 leading-relaxed -mt-1">
              Een referentie is de persoon, het bedrijf of het dossier waarop je de documenten controleert.
            </p>
            <div className="pl-0">
              <SubjectSelector
                variant="wizard"
                selectedSubject={selectedSubject}
                onSubjectChange={setSelectedSubject}
              />
            </div>
          </div>

          {/* Step 2 */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 transition-all ${
                selectedSubject
                  ? 'bg-[#FFC1F3]/20 border border-[#FFC1F3]/40 text-[#FFC1F3]'
                  : 'bg-slate-50 border border-slate-200 text-slate-500'
              }`}>2</span>
              <span className={`text-xs font-semibold transition-colors ${selectedSubject ? 'text-slate-500' : 'text-slate-500'}`}>
                Upload document
              </span>
            </div>
            <div className="pl-0">
              <DocumentUploader
                variant="wizard"
                selectedSubject={selectedSubject}
                onDocumentUploaded={handleUploaded}
                disabled={!selectedSubject}
                selectedModel={selectedModel}
              />
            </div>
            {!selectedSubject && (
              <p className="text-slate-500 text-[10px] pl-7 flex items-center gap-1.5">
                <FontAwesomeIcon icon={faCloudUploadAlt} className="w-3 h-3" />
                Selecteer eerst een referentie hierboven
              </p>
            )}
          </div>
        </div>

        {/* History list */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="px-4 pt-3 pb-2 flex items-center justify-between sticky top-0 backdrop-blur-sm z-10 border-b border-slate-100">
            <p className="text-slate-500 text-[10px] font-semibold uppercase tracking-widest">Analyses</p>
            <Link href="/documents" className="text-slate-400 hover:text-slate-600 text-[10px] transition-colors">
              Alle →
            </Link>
          </div>

          {docs.length === 0 ? (
            <p className="px-4 py-4 text-slate-400 text-xs">Nog geen documenten.</p>
          ) : (
            docs.map(doc => (
              <div
                key={doc.id}
                role="button"
                tabIndex={0}
                onClick={() => setActiveId(doc.id)}
                onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') setActiveId(doc.id); }}
                className={`
                  w-full text-left px-4 py-2.5 flex items-center gap-2.5 group cursor-pointer
                  hover:bg-slate-50 transition-all duration-150
                  border-l-2 ${activeId === doc.id
                    ? 'border-[#FFC1F3] bg-[#FFC1F3]/5'
                    : 'border-transparent'}
                  ${blinking.has(doc.id) ? 'doc-blink' : ''}
                `}
              >
                <span className="shrink-0 w-4 flex justify-center">
                  {doc.status === 'done' && <FontAwesomeIcon icon={faCheckCircle} className="w-3.5 h-3.5 text-emerald-400" />}
                  {(doc.status === 'processing' || doc.status === 'queued') && <FontAwesomeIcon icon={faSpinner} className="w-3.5 h-3.5 text-blue-400 animate-spin" />}
                  {doc.status === 'error' && <FontAwesomeIcon icon={faExclamationTriangle} className="w-3.5 h-3.5 text-red-400" />}
                  {doc.status === 'pending' && <FontAwesomeIcon icon={faFileAlt} className="w-3.5 h-3.5 text-slate-500" />}
                </span>

                <div className="min-w-0 flex-1">
                  <p className="text-slate-800 text-xs font-medium truncate leading-snug">{doc.original_filename}</p>
                  <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                    {doc.doc_type_slug && (
                      <span className="text-[#22d3d3] text-[10px] truncate max-w-[100px]">
                        {fmtType(doc.doc_type_slug)}
                      </span>
                    )}
                    {doc.risk_score != null && (
                      <span className={`inline-flex items-center gap-0.5 text-[10px] font-bold tabular-nums ${riskStyle(doc.risk_score).text}`} title="Fraude risico score">
                        <FontAwesomeIcon icon={faShieldAlt} className="w-2.5 h-2.5" />
                        {doc.risk_score}%
                      </span>
                    )}
                    {(doc.status === 'processing' || doc.status === 'queued') && doc.progress != null && (
                      <span className="text-blue-500 text-[10px] tabular-nums">{Math.round(doc.progress)}%</span>
                    )}
                  </div>
                </div>

                <button
                  onClick={e => { e.stopPropagation(); setConfirmDelete(doc); }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-all"
                  title="Verwijderen"
                >
                  <FontAwesomeIcon icon={faTrash} className="w-3 h-3" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* Training status footer */}
        {trainStatus?.running && (
          <div className="px-4 py-3 border-t border-slate-200 bg-blue-50 shrink-0">
            <div className="flex items-center gap-2">
              <FontAwesomeIcon icon={faSpinner} className="w-3 h-3 text-blue-400 animate-spin shrink-0" />
              <div className="min-w-0 flex-1">
                <p className="text-blue-700 text-xs font-medium leading-none">Model traint</p>
                {trainStatus.current_label && (
                  <p className="text-blue-500 text-[10px] truncate mt-0.5">{trainStatus.current_label}</p>
                )}
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* ── MAIN PANEL ───────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto min-w-0">

        {/* Mobile: upload strip */}
        <div className="md:hidden p-4 space-y-4 border-b border-slate-200 bg-slate-50">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-4 h-4 rounded-full bg-[#FFC1F3]/20 border border-[#FFC1F3]/40 flex items-center justify-center text-[#FFC1F3] text-[9px] font-bold">1</span>
              <span className="text-slate-400 text-xs font-semibold">Kies referentie</span>
            </div>
            <SubjectSelector variant="wizard" selectedSubject={selectedSubject} onSubjectChange={setSelectedSubject} />
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold ${selectedSubject ? 'bg-[#FFC1F3]/20 border border-[#FFC1F3]/40 text-[#FFC1F3]' : 'bg-slate-50 border border-slate-200 text-slate-500'}`}>2</span>
              <span className={`text-xs font-semibold ${selectedSubject ? 'text-slate-400' : 'text-slate-500'}`}>Upload document</span>
            </div>
            <DocumentUploader
              variant="wizard"
              selectedSubject={selectedSubject}
              onDocumentUploaded={handleUploaded}
              disabled={!selectedSubject}
              selectedModel={selectedModel}
            />
          </div>
        </div>

        {/* Mobile: back to list when a document is active */}
        {activeDoc && (
          <div className="md:hidden flex items-center gap-2 px-4 py-2 border-b border-slate-200 bg-white/80 sticky top-0 z-10">
            <button
              onClick={() => setActiveId(null)}
              className="flex items-center gap-1.5 text-[#22d3d3] text-xs font-medium py-1"
            >
              <FontAwesomeIcon icon={faArrowLeft} className="w-3 h-3" />
              Documenten
            </button>
            <span className="text-slate-300 text-xs">·</span>
            <span className="text-slate-500 text-xs truncate min-w-0">{activeDoc.original_filename}</span>
          </div>
        )}

        {/* Content router */}
        {!activeDoc ? (
          <EmptyState docs={docs} onSelect={setActiveId} />
        ) : activeDoc.status === 'error' ? (
          <ErrorPanel doc={activeDoc} onViewDetail={() => openDetail(activeDoc.id)} />
        ) : activeDoc.status === 'done' ? (
          <ResultsPanel doc={activeDoc} fraud={fraud} onOpenDetail={openDetail} />
        ) : (
          <ProcessingPanel doc={activeDoc} step={step} />
        )}
      </main>

      {/* ── MODALS ───────────────────────────────────────────────────────────── */}
      <DocumentDetailModal documentId={detailId} isOpen={isDetailOpen} onClose={() => setIsDetailOpen(false)} initialTab={detailInitialTab} />

      {confirmDelete && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setConfirmDelete(null)}
        >
          <div className="glass-card max-w-sm w-full p-5" onClick={e => e.stopPropagation()}>
            <h3 className="text-slate-800 font-semibold mb-1">Document verwijderen?</h3>
            <p className="text-slate-400 text-sm mb-4 truncate">{confirmDelete.original_filename}</p>
            <div className="flex gap-2">
              <button
                onClick={() => setConfirmDelete(null)}
                className="flex-1 px-3 py-2 bg-slate-100 text-slate-800 rounded-lg text-sm hover:bg-slate-100 transition-colors"
              >
                Annuleren
              </button>
              <button
                onClick={() => { del.mutate(confirmDelete.id); setConfirmDelete(null); }}
                className="flex-1 px-3 py-2 bg-red-600 text-white rounded-lg text-sm hover:bg-red-700 transition-colors"
              >
                Verwijderen
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── EmptyState ───────────────────────────────────────────────────────────────

function EmptyState({ docs, onSelect }: { docs: Document[]; onSelect: (id: number) => void }) {
  const recent = docs.filter(d => d.status === 'done').slice(0, 6);
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] p-6 sm:p-10">

      {/* How it works — card with depth */}
      <div className="glass-card w-full max-w-lg mb-6 p-8">
        {/* Coloured top accent bar */}
        <div className="h-1 w-24 mx-auto rounded-full bg-gradient-to-r from-[#22d3d3] via-[#FFC1F3] to-[#FCE2CE] mb-6 opacity-70" />
        <p className="text-slate-500 text-xs uppercase tracking-widest font-semibold text-center mb-8">Zo werkt het</p>
        <div className="grid grid-cols-3 gap-6">
          {[
            { step: 1, icon: faTag,            color: 'text-[#d040c8]', bg: 'bg-[#FFC1F3]/15 border-[#FFC1F3]/30', title: 'Referentie kiezen',  desc: 'Persoon, bedrijf of dossier dat je controleert' },
            { step: 2, icon: faCloudUploadAlt, color: 'text-blue-500',  bg: 'bg-blue-50 border-blue-200',           title: 'Document uploaden', desc: 'PDF, afbeelding of Word-bestand' },
            { step: 3, icon: faShieldAlt,      color: 'text-emerald-600',bg: 'bg-emerald-50 border-emerald-200',    title: 'Analyse ontvangen',desc: 'Classificatie, velden en risico automatisch' },
          ].map(s => (
            <div key={s.step} className="flex flex-col items-center text-center gap-3">
              <div className={`icon-glow w-14 h-14 flex items-center justify-center rounded-2xl border ${s.bg}`}>
                <FontAwesomeIcon icon={s.icon} className={`w-6 h-6 ${s.color}`} />
              </div>
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold text-white bg-gradient-to-br from-[#22d3d3] to-[#FFC1F3]`}>{s.step}</div>
              <div>
                <p className="text-slate-800 text-xs font-semibold leading-snug">{s.title}</p>
                <p className="text-slate-500 text-[10px] mt-1 leading-snug">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 pt-6 border-t border-slate-100 flex items-center gap-2 justify-center">
          <FontAwesomeIcon icon={faArrowRight} className="w-3 h-3 text-slate-400 rotate-180" />
          <p className="text-slate-500 text-sm hidden md:block">Selecteer een referentie in de zijbalk en upload een document</p>
          <p className="text-slate-500 text-sm md:hidden">Selecteer een referentie hierboven en upload een document</p>
        </div>
      </div>

      {recent.length > 0 && (
        <div className="w-full max-w-lg">
          <p className="text-slate-500 text-xs uppercase tracking-widest mb-3 text-center">Recente analyses</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {recent.map(doc => (
              <button
                key={doc.id}
                onClick={() => onSelect(doc.id)}
                className="text-left p-3 rounded-xl bg-slate-50 border border-slate-200 hover:bg-slate-100 hover:border-slate-300 transition-all group"
              >
                <div className="flex items-center gap-2 mb-1">
                  <FontAwesomeIcon icon={faCheckCircle} className="w-3 h-3 text-emerald-400 shrink-0" />
                  <span className="text-slate-800 text-xs font-medium truncate">{doc.original_filename}</span>
                </div>
                <div className="flex items-center gap-2 pl-5">
                  {doc.doc_type_slug && (
                    <span className="text-[#22d3d3] text-[10px]">{fmtType(doc.doc_type_slug)}</span>
                  )}
                  {doc.risk_score != null && (
                    <span className={`inline-flex items-center gap-0.5 text-[10px] font-bold ${riskStyle(doc.risk_score).text}`} title="Fraude risico score">
                      <FontAwesomeIcon icon={faShieldAlt} className="w-2.5 h-2.5" />
                      {doc.risk_score}%
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── ProcessingPanel ──────────────────────────────────────────────────────────

const PIPELINE = [
  { label: 'OCR', desc: 'Tekst extraheren', step: 1 },
  { label: 'Classificatie', desc: 'Documenttype bepalen', step: 2 },
  { label: 'Extractie', desc: 'Velden extraheren', step: 3 },
  { label: 'Risico', desc: 'Fraude analyse', step: 4 },
] as const;

function ProcessingPanel({ doc, step }: { doc: Document; step: number }) {
  const isQueued = doc.status === 'queued' || doc.status === 'pending';
  const statusText = isQueued ? 'In wachtrij...' : (stageLabel(doc.stage) || 'Voorbereiden...');
  const pct = isQueued ? 0 : Math.round(doc.progress || 0);

  return (
    <div className="flex items-start sm:items-center justify-center pt-4 px-3 pb-4 sm:p-8">
      <div className="glass-card w-full max-w-sm p-5">

        {/* Orbit animation */}
        <div className="flex justify-center mb-5">
          <div className="relative w-16 h-16 flex items-center justify-center">
            <div className="absolute w-12 h-12 rounded-full border border-[#22d3d3]/30 ai-pulse-ring" />
            <div className="w-3.5 h-3.5 rounded-full bg-[#22d3d3] ai-node-pulse" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="ai-orbit-1"><div className="w-1.5 h-1.5 rounded-full bg-[#22d3d3] shadow-[0_0_5px_#22d3d3]" /></div>
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="ai-orbit-2"><div className="w-1 h-1 rounded-full bg-[#FFC1F3] shadow-[0_0_4px_#FFC1F3]" /></div>
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="ai-orbit-3"><div className="w-1 h-1 rounded-full bg-blue-400 shadow-[0_0_4px_rgba(96,165,250,0.8)]" /></div>
            </div>
          </div>
        </div>

        {/* Filename */}
        <p className="text-slate-800 text-sm font-semibold truncate mb-0.5">{doc.original_filename}</p>
        <p className="text-slate-400 text-[11px] mb-4 truncate">{statusText}</p>

        {/* Progress bar */}
        <div className="mb-5">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Voortgang</span>
            <span className="text-[11px] text-slate-500 tabular-nums font-semibold">{pct}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-slate-100 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3]"
              style={{ width: `${pct}%`, transition: 'width 0.6s cubic-bezier(0.4,0,0.2,1)' }}
            />
          </div>
        </div>

        {/* Pipeline steps */}
        <div className="space-y-1">
          {PIPELINE.map(s => {
            const done = step > s.step;
            const active = step === s.step;
            return (
              <div
                key={s.step}
                className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg transition-all duration-200 ${
                  active ? 'bg-[#22d3d3]/6 border-l-2 border-[#22d3d3]/40' : 'border-l-2 border-transparent'
                }`}
              >
                <div className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center text-[10px] font-bold transition-all duration-200 ${
                  done ? 'bg-emerald-100 text-emerald-600' : active ? 'bg-[#22d3d3]/15 text-[#22d3d3]' : 'bg-slate-50 text-slate-400'
                }`}>
                  {done ? (
                    <FontAwesomeIcon icon={faCheck} className="w-2.5 h-2.5" />
                  ) : active ? (
                    <FontAwesomeIcon icon={faSpinner} className="w-2.5 h-2.5 animate-spin" />
                  ) : (
                    s.step
                  )}
                </div>
                <div className="min-w-0">
                  <p className={`text-[11px] font-semibold leading-tight ${done ? 'text-emerald-600' : active ? 'text-slate-700' : 'text-slate-400'}`}>
                    {s.label}
                  </p>
                  {active && (
                    <p className="text-[10px] text-slate-400 leading-tight truncate">
                      {doc.stage ? stageLabel(doc.stage) : s.desc}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── ErrorPanel ───────────────────────────────────────────────────────────────

function ErrorPanel({ doc, onViewDetail }: { doc: Document; onViewDetail: () => void }) {
  return (
    <div className="p-6 max-w-xl">
      <div className="bg-red-50 border border-red-200 rounded-xl p-5">
        <div className="flex items-start gap-3">
          <FontAwesomeIcon icon={faExclamationTriangle} className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div className="min-w-0">
            <h3 className="text-slate-800 font-semibold mb-0.5">Analyse mislukt</h3>
            <p className="text-slate-400 text-sm truncate">{doc.original_filename}</p>
            {doc.error_message && (
              <p className="text-red-600 text-sm mt-2 leading-relaxed">{doc.error_message}</p>
            )}
          </div>
        </div>
        <button
          onClick={onViewDetail}
          className="mt-4 px-4 py-2 bg-slate-100 text-slate-800 rounded-lg text-sm hover:bg-slate-100 transition-colors"
        >
          Bekijk details
        </button>
      </div>
    </div>
  );
}

// ─── ResultsPanel ─────────────────────────────────────────────────────────────

function ResultsPanel({
  doc,
  fraud,
  onOpenDetail,
}: {
  doc: Document;
  fraud?: FraudReport;
  onOpenDetail: (id: number, tab?: 'overview' | 'text' | 'metadata' | 'llm' | 'forensics') => void;
}) {
  const risk = riskStyle(doc.risk_score);
  const fields = doc.metadata_json
    ? Object.entries(doc.metadata_json)
        .filter(([, v]) => v != null && v !== '' && typeof v !== 'object')
        .slice(0, 24)
    : [];
  const topSignals = fraud?.signals?.filter(s => s.risk_level !== 'low').slice(0, 5) || [];
  const advice = fraud?.advice?.filter(a => a.priority === 'high' || a.priority === 'medium') || [];
  const bert = fraud?.semantic_context;
  const fraudLoading = !fraud && doc.status === 'done';

  return (
    <div className="p-4 sm:p-6 space-y-4 animate-results">

      {/* Header */}
      <div className="flex items-center gap-3 flex-wrap pt-1">
        <FontAwesomeIcon icon={faCheckCircle} className="w-4 h-4 text-emerald-400 shrink-0" />
        <h2 className="text-slate-800 font-semibold text-base sm:text-lg truncate flex-1 min-w-0">
          {doc.original_filename}
        </h2>
        <div className="flex items-center gap-2 shrink-0 flex-wrap">
          {doc.doc_type_slug && (
            <span className="px-2.5 py-1 rounded-lg bg-[#22d3d3]/10 border border-[#22d3d3]/30 text-[#22d3d3] text-xs font-medium">
              {fmtType(doc.doc_type_slug)}
            </span>
          )}
          {doc.risk_score != null && (
            <span className={`px-2.5 py-1 rounded-lg border text-xs font-bold tabular-nums ${risk.bg} ${risk.text}`}>
              {doc.risk_score}%
            </span>
          )}
          <button
            onClick={() => onOpenDetail(doc.id)}
            className="px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-slate-800 text-xs font-medium hover:bg-slate-100 transition-colors flex items-center gap-1.5"
          >
            <FontAwesomeIcon icon={faEye} className="w-3 h-3" />
            Details
          </button>
        </div>
      </div>

      {/* Classification + Risk grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">

        {/* Classification */}
        <div className="glass-card-hover p-4" style={{ animationDelay: '0ms' }}>
          <div className="flex items-center gap-2 mb-3">
            <FontAwesomeIcon icon={faTag} className="w-3 h-3 text-[#22d3d3]" />
            <span className="text-slate-400 text-[10px] font-semibold uppercase tracking-widest">Documenttype</span>
          </div>
          {doc.doc_type_slug ? (
            <>
              <p className="text-slate-800 text-xl font-bold mb-2 leading-tight">
                {fmtType(doc.doc_type_slug)}
              </p>
              {doc.doc_type_confidence != null && (
                <div>
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">Zekerheid</span>
                    <span className="text-[#22d3d3] font-bold tabular-nums">
                      {Math.round(doc.doc_type_confidence * 100)}%
                    </span>
                  </div>
                  <div className="h-1 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] rounded-full"
                      style={{ width: `${doc.doc_type_confidence * 100}%` }}
                    />
                  </div>
                </div>
              )}
              {bert?.top_matches?.length ? (
                <div className="mt-3 pt-3 border-t border-slate-200">
                  <p className="text-slate-500 text-[10px] uppercase tracking-widest mb-2 flex items-center gap-1">
                    <FontAwesomeIcon icon={faRobot} className="w-2.5 h-2.5" />
                    BERT
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {bert.top_matches.slice(0, 3).map(m => (
                      <span
                        key={m.label}
                        className="text-[10px] px-2 py-0.5 rounded-full bg-[#22d3d3]/10 border border-[#22d3d3]/30 text-[#22d3d3]"
                      >
                        {fmtType(m.label)} {Math.round(m.confidence * 100)}%
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <p className="text-slate-500 text-sm">Niet geclassificeerd</p>
          )}
        </div>

        {/* Risk */}
        <div className={`rounded-xl border p-4 ${risk.bg}`} style={{ animationDelay: '60ms' }}>
          <div className="flex items-center gap-2 mb-3">
            <FontAwesomeIcon icon={faShieldAlt} className={`w-3 h-3 ${risk.text}`} />
            <span className="text-slate-400 text-[10px] font-semibold uppercase tracking-widest">Risico</span>
          </div>

          {fraudLoading ? (
            <div className="flex items-center gap-2">
              <FontAwesomeIcon icon={faSpinner} className="w-3 h-3 text-slate-500 animate-spin" />
              <span className="text-slate-500 text-xs">Analyseren...</span>
            </div>
          ) : doc.risk_score != null ? (
            <>
              <div className="flex items-baseline gap-2 mb-3">
                <span className={`text-4xl font-black tabular-nums ${risk.text}`}>{doc.risk_score}</span>
                <span className={`text-sm font-medium ${risk.text}`}>% · {risk.label}</span>
              </div>
              {topSignals.length > 0 ? (
                <div className="space-y-1.5">
                  {topSignals.slice(0, 3).map((s, i) => (
                    <div key={i} className="flex items-start gap-1.5">
                      <FontAwesomeIcon
                        icon={faBolt}
                        className={`w-2.5 h-2.5 mt-0.5 shrink-0 ${
                          s.risk_level === 'critical' || s.risk_level === 'high'
                            ? 'text-red-400'
                            : 'text-amber-400'
                        }`}
                      />
                      <p className="text-slate-500 text-[11px] leading-snug">
                        {s.description.length > 70 ? s.description.slice(0, 70) + '…' : s.description}
                      </p>
                    </div>
                  ))}
                  {topSignals.length > 3 && (
                    <button
                      onClick={() => onOpenDetail(doc.id)}
                      className={`text-[10px] ${risk.text} hover:opacity-80 transition-opacity mt-1`}
                    >
                      +{topSignals.length - 3} meer →
                    </button>
                  )}
                </div>
              ) : (
                <p className="text-slate-400 text-xs">Geen verdachte signalen</p>
              )}
            </>
          ) : (
            <p className="text-slate-500 text-sm">Geen data</p>
          )}
        </div>
      </div>

      {/* Extracted fields */}
      {fields.length > 0 && (
        <div className="glass-card-hover p-4" style={{ animationDelay: '120ms' }}>
          <div className="flex items-center gap-2 mb-3">
            <FontAwesomeIcon icon={faList} className="w-3 h-3 text-amber-400" />
            <span className="text-slate-400 text-[10px] font-semibold uppercase tracking-widest">
              Geëxtraheerde velden
            </span>
            <span className="ml-auto text-slate-500 text-[10px] tabular-nums">{fields.length}</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-1.5">
            {fields.map(([key, val]) => (
              <div key={key} className="flex items-start gap-2 min-w-0">
                <span className="text-slate-500 text-[11px] w-28 shrink-0 truncate pt-px">{fmtLabel(key)}</span>
                <span className="text-slate-800 text-[11px] font-medium break-words min-w-0">{fmtVal(val, key)}</span>
              </div>
            ))}
          </div>
          <button
            onClick={() => onOpenDetail(doc.id, 'text')}
            className="mt-3 text-[11px] text-slate-500 hover:text-slate-400 transition-colors"
          >
            Bekijk met bewijs →
          </button>
        </div>
      )}

      {/* Advice / action points */}
      {advice.length > 0 && (
        <div className="space-y-2" style={{ animationDelay: '180ms' }}>
          <div className="flex items-center gap-2">
            <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-[#FFC1F3]" />
            <span className="text-slate-400 text-[10px] font-semibold uppercase tracking-widest">Actiepunten</span>
          </div>
          {advice.slice(0, 4).map((card: AdviceCard, i: number) => (
            <div
              key={i}
              className={`rounded-xl border p-3.5 ${
                card.priority === 'high'
                  ? 'bg-red-50 border-red-200'
                  : 'bg-amber-50 border-amber-200'
              }`}
            >
              <div className="flex items-start gap-2.5">
                <span className={`shrink-0 text-[9px] font-bold px-1.5 py-0.5 rounded mt-0.5 uppercase ${
                  card.priority === 'high'
                    ? 'bg-red-100 text-red-700'
                    : 'bg-amber-100 text-amber-700'
                }`}>
                  {card.priority}
                </span>
                <div className="min-w-0">
                  <p className="text-slate-800 text-xs font-semibold leading-snug">{card.title}</p>
                  <p className="text-slate-400 text-[11px] mt-0.5 leading-relaxed">{card.action}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Full detail CTA */}
      <div className="pb-8 pt-2">
        <button
          onClick={() => onOpenDetail(doc.id)}
          className="flex items-center gap-2.5 px-4 py-3 rounded-xl bg-slate-50 border border-slate-200 text-slate-500 text-sm hover:bg-slate-100 hover:text-slate-800 transition-all group w-full sm:w-auto"
        >
          <FontAwesomeIcon icon={faEye} className="w-4 h-4 text-slate-400 group-hover:text-slate-500" />
          Volledig rapport openen
          <FontAwesomeIcon icon={faArrowRight} className="w-3 h-3 ml-auto sm:ml-4 text-slate-500 group-hover:text-slate-500 group-hover:translate-x-0.5 transition-transform" />
        </button>
      </div>
    </div>
  );
}
