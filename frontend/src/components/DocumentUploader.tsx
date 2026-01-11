'use client';

import { useState, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCloudUploadAlt, faFile, faTimes, faCheck, faInfoCircle } from '@fortawesome/free-solid-svg-icons';
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isDisabled = Boolean(disabled || !selectedSubject);
  const isWizard = variant === 'wizard';

  const uploadMutation = useMutation({
    mutationFn: ({ subjectId, file, modelName }: { subjectId: number; file: File; modelName?: string }) =>
      uploadDocument(subjectId, file, modelName),
    onSuccess: (response, { file }) => {
      // Update progress to completed
      setUploads(prev => prev.map(upload =>
        upload.file === file
          ? { ...upload, status: 'completed', progress: 100 }
          : upload
      ));

      // Fetch the full document details
      // In a real implementation, you'd fetch the document details here
      // For now, we'll create a mock document object
      const mockDocument: Document = {
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

      onDocumentUploaded(mockDocument);

      // Remove from uploads list after a delay
      setTimeout(() => {
        setUploads(prev => prev.filter(upload => upload.file !== file));
      }, 3000);
    },
    onError: (error: any, { file }) => {
      // Extract error message from axios error response
      let errorMessage = 'Upload failed';
      if (error?.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error?.message) {
        errorMessage = error.message;
      }

      setUploads(prev => prev.map(upload =>
        upload.file === file
          ? { ...upload, status: 'error', error: errorMessage }
          : upload
      ));
    },
  });

  const handleFiles = (files: FileList | null) => {
    if (!files || !selectedSubject) return;

    const validFiles = Array.from(files).filter(file => {
      const validTypes = [
        'application/pdf',
        'image/jpeg', 'image/png',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel'
      ];

      if (!validTypes.includes(file.type)) {
        alert(`Unsupported file type: ${file.type}. Please upload PDF, images, or office documents.`);
        return false;
      }

      if (file.size > 50 * 1024 * 1024) { // 50MB
        alert(`File too large: ${file.name}. Maximum size is 50MB.`);
        return false;
      }

      return true;
    });

    // Add to uploads list
    const newUploads: UploadProgress[] = validFiles.map(file => ({
      file,
      progress: 0,
      status: 'uploading' as const,
    }));

    setUploads(prev => [...prev, ...newUploads]);

    // Start uploads
    validFiles.forEach(file => {
      uploadMutation.mutate({
        subjectId: selectedSubject.id,
        file,
        modelName: selectedModel,
      });
    });
  };

  const handleDragOver = (e: React.DragEvent) => {
    if (isDisabled) return;
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    if (isDisabled) return;
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    if (isDisabled) return;
    e.preventDefault();
    setIsDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    // Clear the input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeUpload = (file: File) => {
    setUploads(prev => prev.filter(upload => upload.file !== file));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const dropzoneClasses = isDisabled
    ? 'border-white/15 bg-white/5 cursor-not-allowed'
    : isDragOver
      ? 'border-blue-400 bg-blue-400/10'
      : 'border-white/25 hover:border-white/45 hover:bg-white/5';
  const dropzoneRadius = isWizard ? 'rounded-xl' : 'rounded-lg';

  return (
    <div className={isWizard ? 'space-y-3' : 'space-y-3 sm:space-y-4'}>
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed ${dropzoneRadius} p-6 sm:p-8 lg:p-12 text-center transition-all ${dropzoneClasses} ${isDisabled ? 'opacity-80' : 'cursor-pointer'} overflow-hidden group`}
        onClick={() => {
          if (isDisabled) {
            onDisabledClick?.();
            return;
          }
          fileInputRef.current?.click();
        }}
        role="button"
        aria-disabled={isDisabled}
        tabIndex={isDisabled ? -1 : 0}
      >
        {/* Animated background gradient */}
        {!isDisabled && (
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-purple-500/10 to-pink-500/10 animate-pulse" />
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(96,165,250,0.15),transparent_70%)]" />
          </div>
        )}

        {/* Glowing icon container */}
        <div className="relative z-10 mb-4 sm:mb-6">
          <div className={`inline-flex items-center justify-center w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-white/20 backdrop-blur-sm transition-all duration-300 ${!isDisabled && 'group-hover:scale-110 group-hover:shadow-2xl group-hover:shadow-blue-500/30'}`}>
            <FontAwesomeIcon 
              icon={faCloudUploadAlt} 
              className={`text-white text-2xl sm:text-3xl lg:text-4xl transition-all duration-300 ${!isDisabled && 'group-hover:text-blue-300'}`} 
            />
          </div>
          {!isDisabled && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-blue-500/20 blur-xl animate-pulse" />
            </div>
          )}
        </div>

        <div className="relative z-10 space-y-2">
          <p className="text-white font-semibold text-base sm:text-lg lg:text-xl mb-2">
            {isWizard ? 'Sleep bestanden hierheen of klik om te kiezen' : 'Drop files here or click to browse'}
          </p>
          <p className="text-white/60 text-xs sm:text-sm">
            {isWizard
              ? 'PDF, JPG, PNG, DOCX, XLSX (max 50MB per bestand)'
              : 'Supports PDF, JPG, PNG, DOCX, XLSX (max 50MB each)'}
          </p>
        </div>

        {/* Animated border on hover */}
        {!isDisabled && (
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
            <div className="absolute inset-0 rounded-xl border-2 border-blue-400/50 animate-pulse" />
          </div>
        )}

        {isDisabled && (
          <div className={`absolute inset-0 flex items-center justify-center ${dropzoneRadius} bg-black/40 backdrop-blur-sm z-10`}>
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-blue-600/90 to-purple-600/90 border border-white/30 shadow-xl">
              <FontAwesomeIcon icon={faInfoCircle} className="w-4 h-4 text-white" />
              <span className="text-white font-medium text-xs sm:text-sm">
                {isWizard ? 'Selecteer eerst een subject (stap 1)' : 'Select a subject first'}
              </span>
            </div>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.jpg,.jpeg,.png,.docx,.xlsx,.doc,.xls"
          onChange={handleFileInputChange}
          className="hidden"
        />
      </div>

      {/* Upload Progress */}
      {uploads.length > 0 && (
        <div className="space-y-2">
          {uploads.map((upload, index) => (
            <div key={index} className="glass-card p-2.5 sm:p-3">
              <div className="flex items-center justify-between mb-1.5 sm:mb-2 gap-2">
                <div className="flex items-center space-x-1.5 sm:space-x-2 min-w-0 flex-1">
                  <FontAwesomeIcon icon={faFile} className="text-white/70 w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0" />
                  <span className="text-white text-xs sm:text-sm font-medium truncate">
                    {upload.file.name}
                  </span>
                  <span className="text-white/60 text-[10px] sm:text-xs shrink-0 hidden sm:inline">
                    ({formatFileSize(upload.file.size)})
                  </span>
                </div>
                <div className="flex items-center space-x-1.5 sm:space-x-2 shrink-0">
                  {upload.status === 'completed' && (
                    <FontAwesomeIcon icon={faCheck} className="text-green-400 w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  )}
                  {upload.status === 'error' && (
                    <FontAwesomeIcon icon={faTimes} className="text-red-400 w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  )}
                  <button
                    onClick={() => removeUpload(upload.file)}
                    className="text-white/60 hover:text-white p-0.5"
                  >
                    <FontAwesomeIcon icon={faTimes} className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  </button>
                </div>
              </div>

              <div className="w-full bg-white/20 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    upload.status === 'error' ? 'bg-red-500' :
                    upload.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'
                  }`}
                  style={{ width: `${upload.progress}%` }}
                />
              </div>

              {upload.status === 'error' && upload.error && (
                <p className="text-red-400 text-xs mt-1">{upload.error}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}