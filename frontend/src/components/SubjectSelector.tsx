'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUser, faBuilding, faFolder, faQuestion, faPlus, faSearch } from '@fortawesome/free-solid-svg-icons';
import { Subject, searchSubjects, createSubject } from '@/lib/api';

interface SubjectSelectorProps {
  selectedSubject: Subject | null;
  onSubjectChange: (subject: Subject | null) => void;
  variant?: 'default' | 'wizard';
}

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

export function SubjectSelector({ selectedSubject, onSubjectChange, variant = 'default' }: SubjectSelectorProps) {
  const [context, setContext] = useState<'person' | 'company' | 'dossier' | 'other'>('person');
  const [searchQuery, setSearchQuery] = useState('');
  const queryClient = useQueryClient();

  // Search subjects
  const { data: searchResults, isLoading } = useQuery({
    queryKey: ['subjects', searchQuery, context],
    queryFn: () => searchSubjects(searchQuery || undefined, context, 20),
    enabled: searchQuery.length > 0,
  });

  // Create subject mutation
  const createMutation = useMutation({
    mutationFn: (data: { name: string; context: 'person' | 'company' | 'dossier' | 'other' }) =>
      createSubject(data),
    onSuccess: (newSubject) => {
      onSubjectChange(newSubject);
      setSearchQuery('');
      // Invalidate all subject-related queries
      queryClient.invalidateQueries({ queryKey: ['subjects'] });
      queryClient.invalidateQueries({ predicate: (query) => 
        query.queryKey[0] === 'subjects' || 
        (Array.isArray(query.queryKey) && query.queryKey[0] === 'subjects')
      });
    },
  });

  const handleSubjectSelect = (subject: Subject) => {
    onSubjectChange(subject);
    setSearchQuery('');
  };

  const isWizard = variant === 'wizard';
  const contextButtonBase = isWizard
    ? 'flex items-center justify-center gap-2 px-3 py-2 rounded-full text-xs sm:text-sm font-medium transition-all border'
    : 'flex items-center justify-center space-x-1.5 sm:space-x-2 px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg text-xs sm:text-sm font-medium transition-all border';

  const contextButtonActive = isWizard
    ? 'bg-gradient-to-r from-[#22d3d3]/20 to-[#FFC1F3]/20 border-[#22d3d3]/40 text-slate-800 shadow-sm'
    : 'bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white border-transparent';

  const contextButtonInactive = isWizard
    ? 'bg-slate-50 border-slate-200 text-slate-600 hover:text-slate-800 hover:bg-slate-100 hover:border-slate-300'
    : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100 hover:border-slate-300';

  const inputClasses = isWizard
    ? 'w-full px-4 py-2.5 bg-white/80 border border-slate-200 rounded-xl text-slate-800 placeholder-slate-400 focus:border-transparent text-sm sm:text-base'
    : 'w-full px-3 sm:px-4 py-1.5 sm:py-2 bg-white/80 border border-slate-300 rounded-lg text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-[#22d3d3] focus:border-transparent text-sm';

  return (
    <div className={isWizard ? 'space-y-3' : 'space-y-3 sm:space-y-4'}>
      {/* Context Selection */}
      <div>
        {!isWizard && (
          <label className="block text-slate-800 text-xs sm:text-sm font-medium mb-1.5 sm:mb-2">Type</label>
        )}
        <div className={isWizard ? 'grid grid-cols-2 gap-2' : 'grid grid-cols-2 gap-1.5 sm:gap-2'}>
          {Object.entries(contextLabels).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setContext(key as typeof context)}
              className={`${contextButtonBase} cursor-pointer ${
                context === key ? contextButtonActive : contextButtonInactive
              }`}
            >
              <FontAwesomeIcon icon={contextIcons[key as keyof typeof contextIcons]} className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span className="truncate">{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Subject Search/Input */}
      <div>
        {!isWizard && (
          <label className="block text-slate-800 text-xs sm:text-sm font-medium mb-1.5 sm:mb-2">Naam referentie</label>
        )}
        <div className={`relative ${isWizard ? 'bling-input' : ''}`}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={isWizard ? 'Zoek of maak een referentie...' : 'Zoek of typ naam referentie...'}
            className={inputClasses}
          />
          <FontAwesomeIcon
            icon={faSearch}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 w-4 h-4"
          />
        </div>
      </div>

      {/* Selected Subject Display */}
      {selectedSubject && (
        <div className={isWizard ? 'glass-card p-3' : 'glass-card p-2.5 sm:p-3'}>
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center space-x-1.5 sm:space-x-2 min-w-0 flex-1">
              <FontAwesomeIcon
                icon={contextIcons[selectedSubject.context]}
                className="text-blue-400 w-3.5 h-3.5 sm:w-4 sm:h-4 shrink-0"
              />
              <span className="text-slate-800 font-medium text-sm sm:text-base truncate">{selectedSubject.name}</span>
              <span className="text-slate-400 text-xs sm:text-sm hidden sm:inline">({contextLabels[selectedSubject.context]})</span>
            </div>
            <button
              onClick={() => onSubjectChange(null)}
              className="text-slate-500 hover:text-slate-800 text-xs sm:text-sm shrink-0 cursor-pointer"
            >
              Wijzig
            </button>
          </div>
        </div>
      )}

      {/* Search Results */}
      {searchQuery && searchResults && (
        <div className={isWizard ? 'max-h-56 overflow-y-auto' : 'max-h-48 overflow-y-auto'}>
          {searchResults.subjects.length > 0 ? (
            <div className={isWizard ? 'space-y-1.5' : 'space-y-1'}>
              {searchResults.subjects.map((subject) => (
                <button
                  key={subject.id}
                  onClick={() => handleSubjectSelect(subject)}
                  className={isWizard
                    ? 'w-full text-left px-3 py-2 rounded-xl bg-slate-50 hover:bg-slate-100 border border-slate-200 hover:border-slate-300 transition-all cursor-pointer'
                    : 'w-full text-left px-3 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 transition-colors cursor-pointer'
                  }
                >
                  <div className="flex items-center space-x-2">
                    <FontAwesomeIcon
                      icon={contextIcons[subject.context]}
                      className="text-blue-400 w-4 h-4"
                    />
                    <span className="text-slate-800">{subject.name}</span>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className={isWizard ? 'text-center py-3' : 'text-center py-4'}>
              <p className="text-slate-500 text-sm mb-3">{isWizard ? 'Geen referenties gevonden' : 'Geen referenties gevonden'}</p>
              <button
                onClick={() => {
                  // Directly create subject with current search query
                  createMutation.mutate({
                    name: searchQuery.trim(),
                    context,
                  });
                }}
                disabled={createMutation.isPending || !searchQuery.trim()}
                className={isWizard
                  ? 'flex items-center gap-2 mx-auto px-4 py-2.5 bg-gradient-to-r from-blue-600 to-cyan-600 text-slate-800 rounded-xl hover:from-blue-500 hover:to-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all cursor-pointer shadow-lg'
                  : 'flex items-center space-x-2 mx-auto px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer'
                }
              >
                <FontAwesomeIcon icon={faPlus} className="w-4 h-4" />
                <span>
                  {createMutation.isPending
                    ? 'Aanmaken...'
                    : `"${searchQuery}" aanmaken`
                  }
                </span>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Loading state */}
      {searchQuery && isLoading && (
        <div className={isWizard ? 'text-slate-400 text-sm' : 'text-slate-500 text-sm'}>
          Zoeken...
        </div>
      )}

    </div>
  );
}