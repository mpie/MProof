'use client';

import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSearch, faUser, faBuilding, faFolder, faQuestion,
  faFilter, faTimes, faFileAlt, faSpinner, faChevronDown
} from '@fortawesome/free-solid-svg-icons';
import { searchSubjectsByName, Document, Subject } from '@/lib/api';
import { DocumentList } from '@/components/DocumentList';

const contextIcons = {
  person: faUser,
  company: faBuilding,
  dossier: faFolder,
  other: faQuestion,
};

const contextLabels = {
  person: 'Persoon',
  company: 'Bedrijf',
  dossier: 'Dossier',
  other: 'Overig',
};

const contextColors = {
  person: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  company: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  dossier: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  other: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
};

export default function DocumentsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [selectedContext, setSelectedContext] = useState<string | null>(null);
  const [selectedSubject, setSelectedSubject] = useState<Subject | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);

  // Debounce search query
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    // Reset selections when search changes
    if (value !== searchQuery) {
      setSelectedContext(null);
      setSelectedSubject(null);
    }
    // Debounce
    const timeoutId = setTimeout(() => {
      setDebouncedQuery(value);
    }, 300);
    return () => clearTimeout(timeoutId);
  };

  // Search subjects by name
  const { data: searchResults, isLoading: isSearching } = useQuery({
    queryKey: ['subjects-by-name', debouncedQuery],
    queryFn: () => searchSubjectsByName(debouncedQuery),
    enabled: debouncedQuery.length >= 2,
  });

  // Get available contexts from search results
  const availableContexts = useMemo(() => {
    if (!searchResults?.groups) return [];
    return Object.keys(searchResults.groups).filter(
      ctx => searchResults.groups[ctx].length > 0
    );
  }, [searchResults]);

  // Filter subjects by selected context
  const filteredSubjects = useMemo(() => {
    if (!searchResults?.groups) return [];
    if (selectedContext) {
      return searchResults.groups[selectedContext] || [];
    }
    // Return all subjects from all contexts
    return Object.values(searchResults.groups).flat();
  }, [searchResults, selectedContext]);

  // Count total subjects
  const totalSubjects = useMemo(() => {
    if (!searchResults?.groups) return 0;
    return Object.values(searchResults.groups).flat().length;
  }, [searchResults]);

  // Handle subject selection
  const handleSubjectSelect = (subject: Subject) => {
    setSelectedSubject(subject);
  };

  // Handle document update
  const handleDocumentUpdate = (updatedDocument: Document) => {
    setDocuments(prev =>
      prev.map(doc => doc.id === updatedDocument.id ? updatedDocument : doc)
    );
  };

  const clearFilters = () => {
    setSearchQuery('');
    setDebouncedQuery('');
    setSelectedContext(null);
    setSelectedSubject(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-white text-3xl font-bold">Documenten</h1>
          <p className="text-white/60 mt-1">Zoek en filter documenten op naam en context</p>
        </div>
      </div>

      {/* Search & Filters */}
      <div className="glass-card p-6 space-y-4">
        {/* Search Input */}
        <div className="relative">
          <FontAwesomeIcon
            icon={faSearch}
            className="absolute left-4 top-1/2 -translate-y-1/2 text-white/40 w-4 h-4"
          />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder="Zoek op naam (bijv. 'Jan Jansen', 'Bakkerij Brouwer')..."
            className="w-full pl-11 pr-10 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/40 focus:ring-2 focus:ring-blue-400 focus:border-transparent text-lg"
          />
          {searchQuery && (
            <button
              onClick={clearFilters}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-white/40 hover:text-white"
            >
              <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Context Filter Pills */}
        {availableContexts.length > 0 && (
          <div className="flex items-center space-x-2 flex-wrap gap-2">
            <span className="text-white/60 text-sm flex items-center">
              <FontAwesomeIcon icon={faFilter} className="w-3 h-3 mr-2" />
              Context:
            </span>
            <button
              onClick={() => setSelectedContext(null)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                selectedContext === null
                  ? 'bg-white/20 text-white'
                  : 'bg-white/5 text-white/60 hover:bg-white/10 hover:text-white'
              }`}
            >
              Alle ({totalSubjects})
            </button>
            {availableContexts.map((ctx) => {
              const count = searchResults?.groups[ctx]?.length || 0;
              const icon = contextIcons[ctx as keyof typeof contextIcons] || faQuestion;
              return (
                <button
                  key={ctx}
                  onClick={() => setSelectedContext(ctx)}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all border ${
                    selectedContext === ctx
                      ? contextColors[ctx as keyof typeof contextColors]
                      : 'bg-white/5 text-white/60 border-transparent hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <FontAwesomeIcon icon={icon} className="w-3 h-3" />
                  <span>{contextLabels[ctx as keyof typeof contextLabels] || ctx}</span>
                  <span className="text-xs opacity-60">({count})</span>
                </button>
              );
            })}
          </div>
        )}

        {/* Loading state */}
        {isSearching && (
          <div className="flex items-center justify-center py-4">
            <FontAwesomeIcon icon={faSpinner} className="w-5 h-5 text-blue-400 animate-spin mr-2" />
            <span className="text-white/60">Zoeken...</span>
          </div>
        )}

        {/* No results */}
        {debouncedQuery.length >= 2 && !isSearching && totalSubjects === 0 && (
          <div className="text-center py-8">
            <FontAwesomeIcon icon={faSearch} className="w-12 h-12 text-white/20 mb-3" />
            <p className="text-white/60">Geen resultaten gevonden voor "{debouncedQuery}"</p>
          </div>
        )}

        {/* Subjects Grid */}
        {filteredSubjects.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
            {filteredSubjects.map((subject) => {
              const isSelected = selectedSubject?.id === subject.id;
              const icon = contextIcons[subject.context] || faQuestion;
              const colorClass = contextColors[subject.context] || contextColors.other;
              
              return (
                <button
                  key={subject.id}
                  onClick={() => handleSubjectSelect(subject)}
                  className={`flex items-center space-x-3 p-4 rounded-xl border transition-all text-left ${
                    isSelected
                      ? 'bg-blue-500/20 border-blue-500/50 ring-2 ring-blue-500/30'
                      : 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20'
                  }`}
                >
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${colorClass.split(' ').slice(0, 2).join(' ')}`}>
                    <FontAwesomeIcon icon={icon} className={colorClass.split(' ')[1]} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-white font-medium truncate">{subject.name}</p>
                    <p className="text-white/50 text-sm">
                      {contextLabels[subject.context]}
                    </p>
                  </div>
                  {isSelected && (
                    <FontAwesomeIcon icon={faChevronDown} className="w-4 h-4 text-blue-400" />
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Selected Subject Documents */}
      {selectedSubject && (
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                contextColors[selectedSubject.context].split(' ').slice(0, 2).join(' ')
              }`}>
                <FontAwesomeIcon
                  icon={contextIcons[selectedSubject.context]}
                  className={contextColors[selectedSubject.context].split(' ')[1]}
                />
              </div>
              <div>
                <h2 className="text-white text-xl font-semibold">{selectedSubject.name}</h2>
                <p className="text-white/50 text-sm">{contextLabels[selectedSubject.context]}</p>
              </div>
            </div>
            <button
              onClick={() => setSelectedSubject(null)}
              className="text-white/40 hover:text-white p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
            </button>
          </div>

          <DocumentList
            subjectId={selectedSubject.id}
            documents={documents}
            onDocumentUpdate={handleDocumentUpdate}
            onDocumentsChange={setDocuments}
          />
        </div>
      )}

      {/* Empty State */}
      {!selectedSubject && debouncedQuery.length < 2 && (
        <div className="glass-card p-12 text-center">
          <FontAwesomeIcon icon={faFileAlt} className="w-16 h-16 text-white/20 mb-4" />
          <h3 className="text-white text-lg font-medium mb-2">Zoek naar documenten</h3>
          <p className="text-white/50 max-w-md mx-auto">
            Typ een naam in het zoekveld om te zoeken. Je kunt daarna filteren op context
            (persoon, bedrijf, dossier) en een specifiek subject selecteren om de documenten te bekijken.
          </p>
        </div>
      )}
    </div>
  );
}
