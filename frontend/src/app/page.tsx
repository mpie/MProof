'use client';

import { useState, useEffect } from 'react';
import { SubjectSelector } from '@/components/SubjectSelector';
import { DocumentUploader } from '@/components/DocumentUploader';
import { DocumentList } from '@/components/DocumentList';
import { QuickSearch } from '@/components/QuickSearch';
import { Subject, Document } from '@/lib/api';

export default function Dashboard() {
  const [selectedSubject, setSelectedSubject] = useState<Subject | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubjectChange = (subject: Subject | null) => {
    setSelectedSubject(subject);
  };

  const handleDocumentUploaded = (document: Document) => {
    setDocuments(prev => [document, ...prev]);
  };

  const handleDocumentUpdate = (updatedDocument: Document) => {
    setDocuments(prev =>
      prev.map(doc => doc.id === updatedDocument.id ? updatedDocument : doc)
    );
  };

  return (
    <div className="space-y-8">
      {/* Top Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Subject Selection */}
        <div className="lg:col-span-1 flex">
          <div className="glass-card p-6 w-full flex flex-col">
            <h2 className="text-white text-lg font-semibold mb-4">Subject Selection</h2>
            <div className="flex-1">
              <SubjectSelector
                selectedSubject={selectedSubject}
                onSubjectChange={handleSubjectChange}
              />
            </div>
          </div>
        </div>

        {/* Upload Section */}
        <div className="lg:col-span-2 flex">
          <div className="glass-card p-6 w-full flex flex-col">
            <h2 className="text-white text-lg font-semibold mb-4">Upload Documents</h2>
            <div className="flex-1">
              <DocumentUploader
                selectedSubject={selectedSubject}
                onDocumentUploaded={handleDocumentUploaded}
                disabled={!selectedSubject}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Quick Search */}
      <div className="glass-card p-6">
        <h2 className="text-white text-lg font-semibold mb-4">Quick Search</h2>
        <QuickSearch onSubjectSelect={setSelectedSubject} />
      </div>

      {/* Document List */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-white text-xl font-semibold">
            {selectedSubject
              ? `Documents for ${selectedSubject.name} (${selectedSubject.context})`
              : 'Recent Documents'
            }
          </h2>
        </div>

        <DocumentList
          subjectId={selectedSubject?.id}
          documents={documents}
          onDocumentUpdate={handleDocumentUpdate}
          onDocumentsChange={setDocuments}
        />
      </div>
    </div>
  );
}
