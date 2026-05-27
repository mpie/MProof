'use client';

import { useState, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCloudUploadAlt, faFile, faTimes, faCheck, faExclamationTriangle } from '@fortawesome/free-solid-svg-icons';
import { Subject, Document, uploadDocument } from '@/lib/api';

interface DocumentUploaderProps {
  selectedSubject: Subject | null;
  onDocumentUploaded: (document: Document) => void;
  disabled?: boolean;
  variant?: 'default' | 'wizard';
  onDisabledClick?: () => void;
  selectedModel?: string;
}

interface UploadProgress {
  file: File;
  progress: number;
  status: 'uploading' | 'completed' | 'error';
  error?: string;
  document?: Document;
  duplicateOf?: number;
}

const VALID_TYPES = [
  'application/pdf',
  'image/jpeg', 'image/png',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'application/vnd.ms-excel',
];

function formatSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

export function DocumentUploader({
  selectedSubject,
  onDocumentUploaded,
  disabled,
  variant = 'default',
  onDisabledClick,
  selectedModel,
}: DocumentUploaderProps) {
  const [uploads, setUploads] = useState<UploadProgress[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isDisabled = Boolean(disabled || !selectedSubject);
  const isWizard = variant === 'wizard';

  const uploadMutation = useMutation({
    mutationFn: ({ subjectId, file, modelName }: { subjectId: number; file: File; modelName?: string }) =>
      uploadDocument(subjectId, file, modelName),
    onSuccess: (response, { file }) => {
      setUploads(prev => prev.map(u => u.file === file ? {
        ...u,
        status: 'completed',
        progress: 100,
        duplicateOf: response.duplicate_of ?? undefined,
      } : u));
      const doc: Document = {
        id: response.document_id,
        subject_id: selectedSubject!.id,
        original_filename: file.name,
        mime_type: file.type || 'application/octet-stream',
        size_bytes: file.size,
        sha256: '',
        status: 'queued',
        progress: 0,
        stage: undefined,
        error_message: undefined,
        doc_type_slug: undefined,
        doc_type_confidence: undefined,
        doc_type_rationale: undefined,
        metadata_json: undefined,
        metadata_validation_json: undefined,
        metadata_evidence_json: undefined,
        risk_score: undefined,
        risk_signals_json: undefined,
        ocr_used: false,
        ocr_quality: undefined,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      onDocumentUploaded(doc);
      setTimeout(() => setUploads(prev => prev.filter(u => u.file !== file)), 3000);
    },
    onError: (error: any, { file }) => {
      const msg = error?.response?.data?.detail || error?.message || 'Upload mislukt';
      setUploads(prev => prev.map(u => u.file === file ? { ...u, status: 'error', error: msg } : u));
    },
  });

  const handleFiles = (files: FileList | null) => {
    if (!files || !selectedSubject) return;
    setValidationError(null);

    const valid = Array.from(files).filter(file => {
      if (!VALID_TYPES.includes(file.type)) {
        setValidationError(`Bestandstype niet ondersteund: ${file.name}`);
        return false;
      }
      if (file.size > 50 * 1024 * 1024) {
        setValidationError(`Bestand te groot: ${file.name} (max 50 MB)`);
        return false;
      }
      return true;
    });

    if (valid.length === 0) return;

    setUploads(prev => [...prev, ...valid.map(file => ({ file, progress: 0, status: 'uploading' as const }))]);
    valid.forEach(file => uploadMutation.mutate({ subjectId: selectedSubject.id, file, modelName: selectedModel }));
  };

  const onDragOver = (e: React.DragEvent) => { if (!isDisabled) { e.preventDefault(); setIsDragOver(true); } };
  const onDragLeave = (e: React.DragEvent) => { if (!isDisabled) { e.preventDefault(); setIsDragOver(false); } };
  const onDrop = (e: React.DragEvent) => {
    if (isDisabled) return;
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  };
  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };
  const triggerClick = () => {
    if (isDisabled) { onDisabledClick?.(); return; }
    fileInputRef.current?.click();
  };

  const input = (
    <input
      ref={fileInputRef}
      type="file"
      multiple
      accept=".pdf,.jpg,.jpeg,.png,.docx,.xlsx,.doc,.xls"
      onChange={onFileChange}
      className="hidden"
    />
  );

  // ── Wizard variant: compact sidebar drop zone ─────────────────────────────
  if (isWizard) {
    return (
      <div className="space-y-2">
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={triggerClick}
          className={`
            relative border border-dashed rounded-xl p-3 flex items-center gap-3
            transition-all duration-150 select-none
            ${isDisabled
              ? 'border-slate-200 opacity-40 cursor-not-allowed'
              : isDragOver
                ? 'border-[#FFC1F3]/60 bg-[#FFC1F3]/8 cursor-copy'
                : 'border-slate-300 hover:border-slate-300 hover:bg-slate-50 cursor-pointer'}
          `}
        >
          <div className={`w-8 h-8 shrink-0 flex items-center justify-center rounded-lg transition-colors ${isDragOver ? 'bg-[#FFC1F3]/15' : 'bg-slate-100'}`}>
            <FontAwesomeIcon
              icon={faCloudUploadAlt}
              className={`w-4 h-4 transition-colors ${isDragOver ? 'text-[#FFC1F3]' : 'text-slate-400'}`}
            />
          </div>
          <div className="min-w-0">
            <p className="text-slate-500 text-xs font-medium leading-tight">
              {isDragOver ? 'Loslaten om te uploaden' : 'Sleep of klik om te uploaden'}
            </p>
            <p className="text-slate-500 text-[10px] mt-0.5">PDF · DOCX · JPG · PNG · max 50 MB</p>
          </div>
          {input}
        </div>

        {/* Validation error */}
        {validationError && (
          <div className="flex items-start gap-2 p-2.5 rounded-lg bg-red-500/10 border border-red-500/20">
            <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-red-400 shrink-0 mt-0.5" />
            <p className="text-red-600 text-xs leading-snug flex-1 min-w-0">{validationError}</p>
            <button onClick={() => setValidationError(null)} className="text-red-400/50 hover:text-red-600 shrink-0">
              <FontAwesomeIcon icon={faTimes} className="w-3 h-3" />
            </button>
          </div>
        )}

        {/* Upload queue */}
        {uploads.length > 0 && (
          <div className="space-y-1.5">
            {uploads.map((u, i) => (
              <div key={i} className="flex items-center gap-2 p-2 rounded-lg bg-slate-50 border border-slate-200">
                <FontAwesomeIcon icon={faFile} className="w-3 h-3 text-slate-400 shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-slate-800 text-[10px] font-medium truncate leading-tight">{u.file.name}</p>
                  {u.status === 'uploading' && (
                    <div className="mt-1 h-0.5 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-400 rounded-full animate-pulse" style={{ width: '60%' }} />
                    </div>
                  )}
                  {u.status === 'error' && <p className="text-red-600 text-[10px] mt-0.5">{u.error}</p>}
                  {u.status === 'completed' && u.duplicateOf && (
                    <p className="text-orange-600 text-[10px] mt-0.5">⚠ Duplicaat van #{u.duplicateOf}</p>
                  )}
                </div>
                <span className="shrink-0">
                  {u.status === 'completed' && <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-emerald-400" />}
                  {u.status === 'error' && <FontAwesomeIcon icon={faTimes} className="w-3 h-3 text-red-400" />}
                  {u.status === 'uploading' && <FontAwesomeIcon icon={faCloudUploadAlt} className="w-3 h-3 text-blue-400 animate-pulse" />}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // ── Default variant: full-size drop zone ──────────────────────────────────
  const hasActiveUploads = uploads.some(u => u.status === 'uploading');
  const zoneClass = isDisabled
    ? 'border-slate-200 bg-slate-50 cursor-not-allowed opacity-80'
    : hasActiveUploads
      ? 'border-[#22d3d3]/40 bg-[#22d3d3]/5 cursor-default'
      : isDragOver
        ? 'border-[#FFC1F3]/60 bg-[#FFC1F3]/8'
        : 'border-slate-300 hover:border-slate-300 hover:bg-slate-50 cursor-pointer';

  return (
    <div className="space-y-3 sm:space-y-4">
      <div
        onDragOver={!hasActiveUploads ? onDragOver : undefined}
        onDragLeave={!hasActiveUploads ? onDragLeave : undefined}
        onDrop={!hasActiveUploads ? onDrop : undefined}
        onClick={!hasActiveUploads ? triggerClick : undefined}
        className={`relative border-2 border-dashed rounded-xl p-8 sm:p-12 text-center transition-all group overflow-hidden ${zoneClass}`}
      >
        {!isDisabled && !hasActiveUploads && (
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/8 via-purple-500/8 to-pink-500/8" />
          </div>
        )}

        {hasActiveUploads ? (
          /* Orbit animation during upload */
          <div className="relative z-10 flex flex-col items-center gap-4">
            <div className="relative w-24 h-24 flex items-center justify-center">
              {/* Pulse ring */}
              <div className="absolute w-16 h-16 rounded-full border border-[#22d3d3]/30 ai-pulse-ring" />
              {/* Center node */}
              <div className="w-5 h-5 rounded-full bg-[#22d3d3] ai-node-pulse" />
              {/* Orbit 1 */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="ai-orbit-1">
                  <div className="w-2 h-2 rounded-full bg-[#22d3d3] shadow-[0_0_6px_#22d3d3]" />
                </div>
              </div>
              {/* Orbit 2 */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="ai-orbit-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#FFC1F3] shadow-[0_0_4px_#FFC1F3]" />
                </div>
              </div>
              {/* Orbit 3 */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="ai-orbit-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-400 shadow-[0_0_4px_rgba(96,165,250,0.8)]" />
                </div>
              </div>
            </div>
            <p className="text-[#22d3d3] text-sm font-medium">Uploaden...</p>
          </div>
        ) : (
          <>
            <div className="relative z-10 mb-4">
              <div className={`inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-slate-100 border border-slate-200 transition-all duration-300 ${!isDisabled && 'group-hover:scale-105 group-hover:bg-slate-100'}`}>
                <FontAwesomeIcon icon={faCloudUploadAlt} className={`text-2xl text-slate-400 transition-colors ${!isDisabled && 'group-hover:text-slate-600'}`} />
              </div>
            </div>
            <div className="relative z-10 space-y-1.5">
              <p className="text-slate-800 font-semibold text-base">
                {isDragOver ? 'Loslaten om te uploaden' : 'Sleep bestanden hierheen of klik'}
              </p>
              <p className="text-slate-400 text-sm">PDF · DOCX · JPG · PNG · XLSX · max 50 MB</p>
            </div>
          </>
        )}

        {input}
      </div>

      {/* Validation error */}
      {validationError && (
        <div className="flex items-start gap-2.5 p-3 rounded-xl bg-red-500/10 border border-red-500/20">
          <FontAwesomeIcon icon={faExclamationTriangle} className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <p className="text-red-600 text-sm flex-1 min-w-0">{validationError}</p>
          <button onClick={() => setValidationError(null)} className="text-red-400/50 hover:text-red-600">
            <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Upload queue */}
      {uploads.length > 0 && (
        <div className="space-y-2">
          {uploads.map((u, i) => (
            <div key={i} className="glass-card p-3">
              <div className="flex items-center gap-2 mb-2">
                <FontAwesomeIcon icon={faFile} className="w-4 h-4 text-slate-400 shrink-0" />
                <span className="text-slate-800 text-sm font-medium truncate flex-1">{u.file.name}</span>
                <span className="text-slate-400 text-xs shrink-0">{formatSize(u.file.size)}</span>
                {u.status === 'completed' && <FontAwesomeIcon icon={faCheck} className="w-4 h-4 text-emerald-400 shrink-0" />}
                {u.status === 'error' && <FontAwesomeIcon icon={faTimes} className="w-4 h-4 text-red-400 shrink-0" />}
                <button onClick={() => setUploads(p => p.filter(x => x.file !== u.file))} className="text-slate-500 hover:text-slate-500 shrink-0">
                  <FontAwesomeIcon icon={faTimes} className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${u.status === 'error' ? 'bg-red-500' : u.status === 'completed' ? 'bg-emerald-500' : 'bg-blue-500'}`}
                  style={{ width: `${u.progress || (u.status === 'completed' ? 100 : 30)}%` }}
                />
              </div>
              {u.status === 'error' && u.error && (
                <p className="text-red-600 text-xs mt-1.5">{u.error}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
