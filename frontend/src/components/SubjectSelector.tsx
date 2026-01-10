'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUser, faBuilding, faFolder, faQuestion, faPlus, faSearch } from '@fortawesome/free-solid-svg-icons';
import { Subject, searchSubjects, createSubject } from '@/lib/api';

interface SubjectSelectorProps {
  selectedSubject: Subject | null;
  onSubjectChange: (subject: Subject | null) => void;
}

const contextIcons = {
  person: faUser,
  company: faBuilding,
  dossier: faFolder,
  other: faQuestion,
};

const contextLabels = {
  person: 'Person',
  company: 'Company',
  dossier: 'Dossier',
  other: 'Other',
};

export function SubjectSelector({ selectedSubject, onSubjectChange }: SubjectSelectorProps) {
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
      queryClient.invalidateQueries({ queryKey: ['subjects'] });
    },
  });

  const handleSubjectSelect = (subject: Subject) => {
    onSubjectChange(subject);
    setSearchQuery('');
  };

  return (
    <div className="space-y-4">
      {/* Context Selection */}
      <div>
        <label className="block text-white text-sm font-medium mb-2">Context</label>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(contextLabels).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setContext(key as typeof context)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                context === key
                  ? 'bg-blue-600 text-white'
                  : 'bg-white/10 text-white hover:bg-white/20'
              }`}
            >
              <FontAwesomeIcon icon={contextIcons[key as keyof typeof contextIcons]} className="w-4 h-4" />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Subject Search/Input */}
      <div>
        <label className="block text-white text-sm font-medium mb-2">Subject Name</label>
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search or type subject name..."
            className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/60 focus:ring-2 focus:ring-blue-400 focus:border-transparent"
          />
          <FontAwesomeIcon
            icon={faSearch}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/60 w-4 h-4"
          />
        </div>
      </div>

      {/* Selected Subject Display */}
      {selectedSubject && (
          <div className="glass-card p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <FontAwesomeIcon
                icon={contextIcons[selectedSubject.context]}
                className="text-blue-400 w-4 h-4"
              />
              <span className="text-white font-medium">{selectedSubject.name}</span>
              <span className="text-white/60 text-sm">({selectedSubject.context})</span>
            </div>
            <button
              onClick={() => onSubjectChange(null)}
              className="text-white/60 hover:text-white text-sm"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Search Results */}
      {searchQuery && searchResults && (
        <div className="max-h-48 overflow-y-auto">
          {searchResults.subjects.length > 0 ? (
            <div className="space-y-1">
              {searchResults.subjects.map((subject) => (
                <button
                  key={subject.id}
                  onClick={() => handleSubjectSelect(subject)}
                  className="w-full text-left px-3 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                >
                  <div className="flex items-center space-x-2">
                    <FontAwesomeIcon
                      icon={contextIcons[subject.context]}
                      className="text-blue-400 w-4 h-4"
                    />
                    <span className="text-white">{subject.name}</span>
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-white/60 text-sm mb-3">No subjects found</p>
              <button
                onClick={() => {
                  // Directly create subject with current search query
                  createMutation.mutate({
                    name: searchQuery.trim(),
                    context,
                  });
                }}
                disabled={createMutation.isPending || !searchQuery.trim()}
                className="flex items-center space-x-2 mx-auto px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <FontAwesomeIcon icon={faPlus} className="w-4 h-4" />
                <span>{createMutation.isPending ? 'Creating...' : `Create "${searchQuery}"`}</span>
              </button>
            </div>
          )}
        </div>
      )}

    </div>
  );
}