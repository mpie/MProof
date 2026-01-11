'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface ModelContextType {
  selectedModel: string | undefined;
  setSelectedModel: (model: string | undefined) => void;
}

const ModelContext = createContext<ModelContextType | undefined>(undefined);

export function ModelProvider({ children }: { children: ReactNode }) {
  const [selectedModel, setSelectedModel] = useState<string | undefined>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('mproof_classify_model');
      return saved || undefined;
    }
    return undefined;
  });

  // Persist to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      if (selectedModel) {
        localStorage.setItem('mproof_classify_model', selectedModel);
      } else {
        localStorage.removeItem('mproof_classify_model');
      }
    }
  }, [selectedModel]);

  return (
    <ModelContext.Provider value={{ selectedModel, setSelectedModel }}>
      {children}
    </ModelContext.Provider>
  );
}

export function useModel() {
  const context = useContext(ModelContext);
  if (context === undefined) {
    throw new Error('useModel must be used within a ModelProvider');
  }
  return context;
}
