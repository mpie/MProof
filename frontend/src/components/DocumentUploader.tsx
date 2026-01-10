'use client';

import { useState, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCloudUploadAlt, faFile, faTimes, faCheck } from '@fortawesome/free-solid-svg-icons';
import { Subject, Document, uploadDocument } from '@/lib/api';

interface DocumentUploaderProps {
  selectedSubject: Subject | null;
  onDocumentUploaded: (document: Document) => void;
  disabled?: boolean;
}

interface UploadProgress {
  file: File;
  progress: number;
  status: 'uploading' | 'completed' | 'error';
  error?: string;
  document?: Document;
}

export function DocumentUploader({ selectedSubject, onDocumentUploaded, disabled }: DocumentUploaderProps) {
  const [uploads, setUploads] = useState<UploadProgress[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadMutation = useMutation({
    mutationFn: ({ subjectId, file }: { subjectId: number; file: File }) =>
      uploadDocument(subjectId, file),
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
        sha256: '', // Will be set by backend
        status: 'queued',
        progress: 0,
        stage: null,
        error_message: null,
        doc_type_slug: null,
        doc_type_confidence: null,
        doc_type_rationale: null,
        metadata_json: null,
        metadata_validation_json: null,
        metadata_evidence_json: null,
        risk_score: null,
        risk_signals_json: null,
        ocr_used: false,
        ocr_quality: null,
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
      });
    });
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
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

  if (disabled || !selectedSubject) {
    return (
      <div className="text-center py-12">
        <FontAwesomeIcon icon={faCloudUploadAlt} className="text-white/40 text-4xl mb-4" />
        <p className="text-white/60">Select a subject first to upload documents</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
          isDragOver
            ? 'border-blue-400 bg-blue-400/10'
            : 'border-white/30 hover:border-white/50'
        }`}
        onClick={() => fileInputRef.current?.click()}
      >
        <FontAwesomeIcon icon={faCloudUploadAlt} className="text-white/60 text-3xl mb-4" />
        <p className="text-white text-lg mb-2">Drop files here or click to browse</p>
        <p className="text-white/60 text-sm">
          Supports PDF, JPG, PNG, DOCX, XLSX (max 50MB each)
        </p>
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
            <div key={index} className="glass-card p-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <FontAwesomeIcon icon={faFile} className="text-white/70 w-4 h-4" />
                  <span className="text-white text-sm font-medium truncate">
                    {upload.file.name}
                  </span>
                  <span className="text-white/60 text-xs">
                    ({formatFileSize(upload.file.size)})
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  {upload.status === 'completed' && (
                    <FontAwesomeIcon icon={faCheck} className="text-green-400 w-4 h-4" />
                  )}
                  {upload.status === 'error' && (
                    <FontAwesomeIcon icon={faTimes} className="text-red-400 w-4 h-4" />
                  )}
                  <button
                    onClick={() => removeUpload(upload.file)}
                    className="text-white/60 hover:text-white"
                  >
                    <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
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