'use client';

import { useQuery } from '@tanstack/react-query';
import { getLLMHealth } from '@/lib/api';

export function LLMStatus() {
  const { data: health, isLoading } = useQuery({
    queryKey: ['llm-health'],
    queryFn: getLLMHealth,
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 10000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center space-x-2 text-white/50 text-sm">
        <span className="w-2 h-2 bg-white/30 rounded-full animate-pulse"></span>
        <span>LLM...</span>
      </div>
    );
  }

  const activeProvider = health?.active_provider || 'ollama';
  const providerHealth = health?.providers?.[activeProvider as 'ollama' | 'vllm'];
  const isOnline = providerHealth?.reachable || false;
  const providerName = activeProvider === 'vllm' ? 'vLLM' : 'Ollama';

  return (
    <div className="flex items-center space-x-2 text-white text-sm">
      <span 
        className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-400' : 'bg-red-400'}`}
        title={isOnline ? 'Online' : 'Offline'}
      ></span>
      <span>{providerName}</span>
    </div>
  );
}
