'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUser, faBuilding, faFolder, faQuestion, faSearch } from '@fortawesome/free-solid-svg-icons';
import { Subject, searchSubjectsByName } from '@/lib/api';

interface QuickSearchProps {
  onSubjectSelect: (subject: Subject) => void;
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

export function QuickSearch({ onSubjectSelect }: QuickSearchProps) {
  const [query, setQuery] = useState('');
  const [showResults, setShowResults] = useState(false);

  const { data: results, isLoading } = useQuery({
    queryKey: ['subjects-by-name', query],
    queryFn: () => searchSubjectsByName(query),
    enabled: query.length >= 2,
  });

  const handleInputChange = (value: string) => {
    setQuery(value);
    setShowResults(value.length >= 2);
  };

  const handleSubjectSelect = (subject: Subject) => {
    onSubjectSelect(subject);
    setQuery('');
    setShowResults(false);
  };

  return (
    <div className="relative">
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => handleInputChange(e.target.value)}
          placeholder="Search subjects by name..."
          className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/60 focus:ring-2 focus:ring-blue-400 focus:border-transparent"
        />
        <FontAwesomeIcon
          icon={faSearch}
          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/60 w-5 h-5"
        />
      </div>

      {/* Search Results */}
      {showResults && (
        <div className="absolute top-full left-0 right-0 mt-2 glass-card max-h-96 overflow-y-auto z-10">
          {isLoading ? (
            <div className="p-4 text-center text-white/60">
              Searching...
            </div>
          ) : results && Object.keys(results.groups).length > 0 ? (
            <div className="p-2">
              {Object.entries(results.groups).map(([context, subjects]) => (
                <div key={context} className="mb-4 last:mb-0">
                  <div className="flex items-center space-x-2 px-3 py-2 bg-white/5 rounded-lg mb-2">
                    <FontAwesomeIcon
                      icon={contextIcons[context as keyof typeof contextIcons]}
                      className="text-blue-400 w-4 h-4"
                    />
                    <span className="text-white font-medium">
                      {contextLabels[context as keyof typeof contextLabels]}
                    </span>
                    <span className="text-white/60 text-sm">
                      ({subjects.length})
                    </span>
                  </div>

                  <div className="space-y-1">
                    {subjects.slice(0, 5).map((subject) => (
                      <button
                        key={subject.id}
                        onClick={() => handleSubjectSelect(subject)}
                        className="w-full text-left px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                      >
                        <span className="text-white">{subject.name}</span>
                      </button>
                    ))}

                    {subjects.length > 5 && (
                      <div className="px-4 py-2 text-white/60 text-sm">
                        ... and {subjects.length - 5} more
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : query.length >= 2 ? (
            <div className="p-4 text-center text-white/60">
              No subjects found for "{query}"
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}