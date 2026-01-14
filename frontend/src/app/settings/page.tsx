'use client';

import { useState, useMemo, useEffect, useRef } from 'react';
import { useQuery, useQueries, useMutation, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faCog, faKey, faRobot, faPlug, faPlus, faTrash, faCopy, faCheck,
  faEye, faEyeSlash, faRefresh, faExclamationTriangle, faGraduationCap,
  faFileAlt, faCode, faDatabase, faFilter, faToggleOn, faToggleOff, faTimes, faSpinner,
  faFolder, faChevronDown, faBan, faHandPointer, faSearch, faBullseye, faBolt, faBrain,
  faLightbulb, faGlobe, faCommentDots, faChartBar, faRocket, faShieldAlt, faImage, faInfoCircle
} from '@fortawesome/free-solid-svg-icons';
import {
  getTrainingDetails,
  TrainingDetails,
  listApiKeys,
  createApiKey,
  updateApiKey,
  deleteApiKey,
  regenerateApiKey,
  ApiKey,
  ApiKeyCreated,
  getClassifierStatus,
  trainClassifier,
  ClassifierStatus,
  listSkipMarkers,
  createSkipMarker,
  updateSkipMarker,
  deleteSkipMarker,
  SkipMarker,
  getAvailableModels,
  getBertClassifierStatus,
  trainBertClassifier,
  BertClassifierStatus,
  getLLMSettings,
  getLLMHealth,
  switchLLMProvider,
  LLMSettingsResponse,
  LLMHealthResponse,
  getAppSettings,
  updateAppSetting,
  listDocumentTypes,
  generateDocumentTypePrefill,
} from '@/lib/api';
import { useModel } from '@/context/ModelContext';

type TabType = 'model' | 'llm' | 'api-keys' | 'skip-markers' | 'fraud-detection' | 'mcp';

interface TokenInfo {
  token: string;
  count: number;
}

interface TrainedLabelsGridProps {
  labels: string[];
  docCounts: Record<string, number>;
  trainingFilesByLabel: Record<string, Array<{ path: string; sha256: string; updated_at: string; }>>;
  tokensByLabel: Record<string, TokenInfo[]>;
  modelName?: string; // Which model this belongs to
  allModelsData?: Array<{ name: string; document_types: Array<{ slug: string }> }>;
}

// Scalable grid for trained document types
function TrainedLabelsGrid({ labels, docCounts, trainingFilesByLabel, tokensByLabel, modelName, allModelsData }: TrainedLabelsGridProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  const [showTokensModal, setShowTokensModal] = useState(false);
  const { selectedModel } = useModel();
  const router = useRouter();
  
  // Fetch existing document types to check which ones don't exist
  const { data: existingDocumentTypes } = useQuery({
    queryKey: ['document-types'],
    queryFn: listDocumentTypes,
  });
  
  // Create a set of existing document type slugs for quick lookup
  const existingSlugs = useMemo(() => {
    if (!existingDocumentTypes) return new Set<string>();
    return new Set(existingDocumentTypes.map(dt => dt.slug.toLowerCase()));
  }, [existingDocumentTypes]);
  
  // Helper function to generate slug from label
  const generateSlug = (label: string): string => {
    return label
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '');
  };
  
  // Helper function to capitalize first letter
  const capitalizeFirst = (str: string): string => {
    if (!str) return str;
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  };
  
  // Helper function to create classification hints from top tokens
  const createClassificationHints = (label: string): string => {
    const tokens = tokensByLabel[label] || [];
    const topTokens = tokens.slice(0, 10).map(t => t.token);
    // Format as "kw:keyword" per line
    return topTokens.map(token => `kw:${token}`).join('\n');
  };
  
  // Navigate to document types page with prefill
  const handleCreateDocumentType = (label: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const slug = generateSlug(label);
    const hints = createClassificationHints(label);
    const capitalizedName = capitalizeFirst(label);
    
    // Navigate immediately, LLM will be called on the document types page
    const params = new URLSearchParams({
      create: 'true',
      name: capitalizedName,
      slug: slug,
      classification_hints: hints,
      generate_llm: 'true', // Flag to trigger LLM generation on the page
    });
    
    router.push(`/document-types?${params.toString()}`);
  };

  // Build a map of label -> model name for "Standaard" view
  const labelToModel = useMemo(() => {
    const map: Record<string, string> = {};
    if (allModelsData) {
      allModelsData.forEach(model => {
        model.document_types?.forEach(dt => {
          if (!map[dt.slug]) {
            map[dt.slug] = model.name;
          }
        });
      });
    }
    return map;
  }, [allModelsData]);

  const filteredLabels = labels.filter(label => 
    label.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Get tokens for selected label - try multiple sources
  const selectedTokens = useMemo(() => {
    if (!selectedLabel) return [];
    
    // First try direct lookup
    if (tokensByLabel[selectedLabel]?.length > 0) {
      return tokensByLabel[selectedLabel];
    }
    
    // Try case-insensitive match
    const lowerLabel = selectedLabel.toLowerCase();
    for (const [key, tokens] of Object.entries(tokensByLabel)) {
      if (key.toLowerCase() === lowerLabel && tokens?.length > 0) {
        return tokens;
      }
    }
    
    return [];
  }, [selectedLabel, tokensByLabel]);
  
  const selectedLabelModel = selectedLabel ? labelToModel[selectedLabel] : null;
  const selectedLabelDocCount = selectedLabel ? (docCounts[selectedLabel] || 0) : 0;

  return (
    <div className="glass-card p-4 sm:p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
            <FontAwesomeIcon icon={faDatabase} className="text-blue-400 w-4 h-4" />
          </div>
          {modelName ? `Document Types (${labels.length})` : `Getrainde Modellen (${labels.length})`}
          {modelName && (
            <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded-full font-normal">
              {modelName}
            </span>
          )}
        </h3>
        
        {/* Search */}
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Zoek type..."
            className="w-full sm:w-48 px-3 py-1.5 pl-8 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-blue-500/50 focus:outline-none"
          />
          <FontAwesomeIcon icon={faSearch} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-white/40 w-3 h-3" />
        </div>
      </div>

      {/* Compact Grid - scrollable for many items */}
      <div className="max-h-96 overflow-y-auto pr-1 scrollbar-thin">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          {filteredLabels.map((label) => {
            const docCount = docCounts[label] || 0;
            const topTokens = tokensByLabel[label] || [];
            const hasTokens = topTokens.length > 0;
            const belongsToModel = labelToModel[label];
            const labelSlug = generateSlug(label);
            const documentTypeExists = existingSlugs.has(labelSlug.toLowerCase());

            return (
              <div 
                key={label} 
                onClick={() => {
                  setSelectedLabel(label);
                  setShowTokensModal(true);
                }}
                className="bg-white/5 rounded-lg p-4 border border-white/10 hover:border-white/30 hover:bg-white/10 transition-all cursor-pointer group relative min-h-[120px]"
              >
                <div className="flex items-start justify-between gap-2 mb-2 pr-16">
                  <h4 className="text-white font-semibold text-sm flex-1 break-words" title={label}>
                    {label}
                  </h4>
                </div>
                
                {/* Create Document Type button if it doesn't exist */}
                {!documentTypeExists && (
                  <button
                    onClick={(e) => handleCreateDocumentType(label, e)}
                    className="absolute top-2.5 right-2.5 z-10 flex items-center gap-1 px-1.5 py-0.5 bg-green-500/10 hover:bg-green-500/20 border border-green-500/20 text-green-400 text-[10px] rounded transition-colors cursor-pointer opacity-70 hover:opacity-100"
                    title="Maak document type aan"
                  >
                    <FontAwesomeIcon icon={faPlus} className="w-2.5 h-2.5" />
                    <span>Maak aan</span>
                  </button>
                )}
                
                {/* Show model name when viewing all (Standaard) */}
                {!selectedModel && belongsToModel && (
                  <div className="mb-2">
                    <span className="text-xs bg-purple-500/15 text-purple-200 px-2 py-0.5 rounded font-medium inline-flex items-center gap-1">
                      <FontAwesomeIcon icon={faFolder} className="w-3 h-3" />
                      {belongsToModel}
                    </span>
                  </div>
                )}
                
                {/* Top 3 tokens preview */}
                {hasTokens && (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {topTokens.slice(0, 3).map((t, i) => (
                      <span 
                        key={i}
                        className="text-xs bg-purple-500/15 text-white/90 px-1.5 py-0.5 rounded font-mono truncate max-w-[80px]"
                        title={t.token}
                      >
                        {t.token}
                      </span>
                    ))}
                    {topTokens.length > 3 && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedLabel(label);
                          setShowTokensModal(true);
                        }}
                        className="text-xs text-purple-300 hover:text-purple-100 px-1.5 py-0.5 hover:bg-purple-500/30 rounded transition-colors cursor-pointer font-medium underline decoration-purple-400/50 hover:decoration-purple-300"
                        title={`Bekijk alle ${topTokens.length} woorden`}
                      >
                        +{topTokens.length - 3} meer
                      </button>
                    )}
                  </div>
                )}

                {/* Click hint with document count */}
                <div className="flex items-center justify-between gap-2 text-xs text-white/60 group-hover:text-white/80 transition-colors">
                  <div className="flex items-center gap-1.5">
                    <FontAwesomeIcon icon={faHandPointer} className="w-3 h-3" />
                    <span>Bekijk details</span>
                  </div>
                  <span className="text-xs bg-blue-500/20 text-blue-200 px-2 py-0.5 rounded font-medium">
                    {docCount}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {filteredLabels.length === 0 && searchQuery && (
        <div className="text-center py-4 text-white/50 text-sm">
          Geen types gevonden voor "{searchQuery}"
        </div>
      )}

      {/* Tokens Modal */}
      {showTokensModal && selectedLabel && (
        <div 
          className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setShowTokensModal(false);
              setSelectedLabel(null);
            }
          }}
        >
          <div className="bg-[#1a1a2e] rounded-xl border border-white/10 max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl">
            <div className="p-5 border-b border-white/10 flex items-center justify-between">
              <div>
                <h3 className="text-white font-bold text-lg">Herkende woorden: {selectedLabel}</h3>
                <p className="text-white/50 text-sm mt-1">
                      {selectedTokens.length > 0 ? (
                    <>
                      {selectedTokens.length} unieke tokens • Gesorteerd op frequentie
                      {!selectedModel && selectedLabelModel && (
                        <span className="ml-2 text-purple-200">• Model: {selectedLabelModel}</span>
                      )}
                    </>
                  ) : (
                    "Geen training data beschikbaar voor dit document type"
                  )}
                </p>
              </div>
              <button
                onClick={() => {
                  setShowTokensModal(false);
                  setSelectedLabel(null);
                }}
                className="text-white/60 hover:text-white px-3 py-1 hover:bg-white/10 rounded transition-colors"
                title="Sluiten"
              >
                <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
              </button>
            </div>
            
            <div className="p-5 overflow-y-auto flex-1">
              {selectedTokens.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {selectedTokens.map((t, i) => (
                    <span 
                      key={i} 
                      className="inline-flex items-center gap-2 text-sm bg-purple-500/20 text-white px-3 py-1.5 rounded-lg font-mono border border-purple-500/30 hover:bg-purple-500/30 transition-colors"
                      title={`${t.token} komt ${t.count}x voor`}
                    >
                      <span className="truncate max-w-[200px]">{t.token}</span>
                      <span className="text-white/70 text-xs font-semibold">×{t.count}</span>
                    </span>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 mx-auto mb-3 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                    <FontAwesomeIcon icon={faFileAlt} className="w-8 h-8 text-white/40" />
                  </div>
                  <p className="text-white/80 text-base font-medium">
                    Geen training data beschikbaar
                  </p>
                  <p className="text-white/50 text-sm mt-2">
                    {selectedLabelDocCount > 0 ? (
                      <>
                        Dit document type heeft {selectedLabelDocCount} documenten, maar er zijn geen tokens beschikbaar.
                        <br />
                        Train het model opnieuw om woorden voor dit document type te zien.
                      </>
                    ) : (
                      "Train het model om woorden voor dit document type te zien"
                    )}
                  </p>
                  {selectedLabelModel && (
                    <p className="text-white/40 text-xs mt-3">
                      Model: {selectedLabelModel}
                    </p>
                  )}
                </div>
              )}
            </div>

            <div className="p-5 border-t border-white/10 flex justify-between items-center bg-white/5">
              <p className="text-white/50 text-sm flex items-center gap-1.5">
                <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-400" />
                Deze woorden helpen het model om "{selectedLabel}" documenten te herkennen
              </p>
              <button
                onClick={() => {
                  setShowTokensModal(false);
                  setSelectedLabel(null);
                }}
                className="px-5 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm font-medium transition-colors"
              >
                Sluiten
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<TabType>('model');

  const tabs = [
    { id: 'model' as TabType, label: 'Model', icon: faRobot },
    { id: 'llm' as TabType, label: 'LLM', icon: faBrain },
    { id: 'skip-markers' as TabType, label: 'Skip Markers', icon: faFilter },
    { id: 'fraud-detection' as TabType, label: 'Fraud Detection', icon: faShieldAlt },
    { id: 'api-keys' as TabType, label: 'API Keys', icon: faKey },
    { id: 'mcp' as TabType, label: 'MCP', icon: faPlug },
  ];

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-white text-xl sm:text-2xl lg:text-3xl font-bold flex items-center gap-2 sm:gap-3">
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
            <FontAwesomeIcon icon={faCog} className="text-blue-400 w-5 h-5 sm:w-6 sm:h-6" />
          </div>
          Instellingen
        </h1>
        <p className="text-white/60 mt-1 text-sm sm:text-base">Beheer training, API keys en integraties</p>
      </div>

      {/* Tabs - scrollable on mobile */}
      <div className="glass-card p-1 flex rounded-lg sm:rounded-xl overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 sm:py-2.5 rounded-lg text-xs sm:text-sm font-medium transition-all whitespace-nowrap cursor-pointer ${
              activeTab === tab.id
                ? 'bg-purple-500/25 text-purple-300 border border-purple-400/30'
                : 'text-white/60 hover:text-white hover:bg-white/5'
            }`}
          >
            <FontAwesomeIcon icon={tab.icon} className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'model' && <ModelTab />}
      {activeTab === 'llm' && <LLMTab />}
      {activeTab === 'skip-markers' && <SkipMarkersTab />}
      {activeTab === 'fraud-detection' && <FraudDetectionTab />}
      {activeTab === 'api-keys' && <ApiKeysTab />}
      {activeTab === 'mcp' && <McpTab />}
    </div>
  );
}

// Model Tab (replaces Training Tab)
function ModelTab() {
  const queryClient = useQueryClient();
  const { selectedModel, setSelectedModel } = useModel();
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [showStopwordsModal, setShowStopwordsModal] = useState(false);
  const [activeSection, setActiveSection] = useState<string>('naive-bayes');
  const isProgrammaticScrollRef = useRef(false);

  // Scroll spy to update active section
  useEffect(() => {
    let scrollTimeout: NodeJS.Timeout | undefined;
    let rafId: number | undefined;

    const handleScroll = () => {
      // Skip scroll spy updates during programmatic scrolling
      if (isProgrammaticScrollRef.current) {
        return;
      }

      // Cancel any pending RAF
      if (rafId) {
        cancelAnimationFrame(rafId);
      }

      // Use requestAnimationFrame for smoother updates
      rafId = requestAnimationFrame(() => {
        const sectionIds = [
          'naive-bayes',
          'bert', 
          'how-extraction-works',
          'training-data',
          'classification-priority',
          'stopwords',
        ];

        // Get all sections with their positions
        const sectionsWithPositions = sectionIds
          .map(id => {
            const element = document.getElementById(id);
            if (!element) return null;
            const rect = element.getBoundingClientRect();
            return {
              id,
              top: rect.top,
              bottom: rect.bottom,
            };
          })
          .filter((s): s is { id: string; top: number; bottom: number } => s !== null);

        if (sectionsWithPositions.length === 0) return;

        // Find the section that's currently most visible
        // A section is "active" if its top is above the middle of the viewport
        const viewportMiddle = window.innerHeight / 3; // Use top third as trigger point
        
        let activeId = sectionsWithPositions[0].id; // Default to first
        
        for (let i = sectionsWithPositions.length - 1; i >= 0; i--) {
          const section = sectionsWithPositions[i];
          // If the section's top is above the trigger point, it's the active one
          if (section.top <= viewportMiddle) {
            activeId = section.id;
            break;
          }
        }

        // Only update if it's different to avoid unnecessary re-renders
        setActiveSection((prev) => {
          if (prev !== activeId) {
            return activeId;
          }
          return prev;
        });
      });

      // Clear any pending timeout
      if (scrollTimeout) {
        clearTimeout(scrollTimeout);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    // Also listen for resize as it can affect positions
    window.addEventListener('resize', handleScroll, { passive: true });
    
    // Initial check after a small delay to ensure DOM is ready
    scrollTimeout = setTimeout(handleScroll, 100);
    
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      if (scrollTimeout) clearTimeout(scrollTimeout);
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleScroll);
    };
  }, []);
  
  // App settings for ELA/EXIF
  const { data: appSettings } = useQuery({
    queryKey: ['app-settings'],
    queryFn: () => getAppSettings(),
  });
  
  const elaEnabled = appSettings?.find(s => s.key === 'ela_enabled')?.value === 'true' || false;
  const exifEnabled = appSettings?.find(s => s.key === 'exif_enabled')?.value === 'true' || false;
  
  const updateSettingMutation = useMutation({
    mutationFn: ({ key, value }: { key: string; value: string }) => updateAppSetting(key, value),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['app-settings'] });
    },
  });
  
  // Dutch stopwords - complete list from backend
  const dutchStopwords = {
    'Lidwoorden & Voornaamwoorden': ['de', 'het', 'een', 'die', 'dat', 'deze', 'dit', 'hij', 'zij', 'wij', 'jullie', 'hun', 'haar', 'hem', 'mij', 'jou', 'ons', 'hen', 'wie', 'wat', 'welke', 'welk'],
    'Voorzetsels': ['van', 'in', 'op', 'te', 'aan', 'met', 'voor', 'door', 'over', 'bij', 'naar', 'uit', 'tot', 'om', 'onder', 'tegen', 'tussen', 'zonder', 'binnen', 'buiten'],
    'Voegwoorden': ['en', 'of', 'maar', 'want', 'dus', 'omdat', 'als', 'dan', 'toen', 'terwijl', 'hoewel', 'indien', 'tenzij', 'zodra', 'voordat', 'nadat', 'zodat'],
    'Veelvoorkomende Werkwoorden': ['is', 'zijn', 'was', 'waren', 'ben', 'bent', 'heeft', 'hebben', 'had', 'hadden', 'wordt', 'worden', 'werd', 'werden', 'kan', 'kunnen', 'kon', 'konden', 'zal', 'zullen', 'zou', 'zouden', 'moet', 'moeten', 'moest', 'moesten', 'mag', 'mogen', 'wil', 'willen', 'wilde', 'wilden', 'gaat', 'gaan', 'ging', 'gingen', 'komt', 'komen', 'kwam', 'kwamen', 'doet', 'doen', 'deed', 'deden', 'zegt', 'zeggen'],
    'Bijwoorden & Overig': ['niet', 'ook', 'nog', 'wel', 'al', 'er', 'hier', 'daar', 'waar', 'hoe', 'nu', 'dan', 'toen', 'zo', 'toch', 'heel', 'erg', 'zeer', 'meer', 'veel', 'weinig', 'alle', 'alles', 'iets', 'niets', 'iemand', 'niemand', 'elke', 'elk', 'ander', 'andere', 'eigen', 'zelf', 'alleen', 'samen', 'verder', 'eerst', 'laatste'],
    'Getallen (als woorden)': ['een', 'twee', 'drie', 'vier', 'vijf', 'zes', 'zeven', 'acht', 'negen', 'tien'],
    'Veelvoorkomende Documentwoorden': ['pagina', 'bladzijde', 'datum', 'naam', 'adres', 'www', 'http', 'https', 'com', 'org', 'net'],
  };
  
  const totalStopwords = Object.values(dutchStopwords).flat().length;

  const { data: availableModels } = useQuery({
    queryKey: ['available-models'],
    queryFn: getAvailableModels,
  });

  const { data: trainingDetails, isLoading: detailsLoading } = useQuery({
    queryKey: ['training-details', selectedModel],
    queryFn: () => getTrainingDetails(selectedModel),
  });

  // Fetch training details for all models when "Standaard" is selected
  const allModelTrainingQueries = useQueries({
    queries: !selectedModel && availableModels?.models 
      ? availableModels.models.map(model => ({
          queryKey: ['training-details', model.name],
          queryFn: () => getTrainingDetails(model.name),
        }))
      : [],
  });

  // Aggregate all model training details for "Standaard" view
  const aggregatedTrainingDetails = useMemo(() => {
    if (selectedModel || !availableModels?.models || availableModels.models.length === 0) {
      return null;
    }

    const allTokens: Record<string, Array<{ token: string; count: number }>> = {};
    const allDocCounts: Record<string, number> = {};

    allModelTrainingQueries.forEach((query, index) => {
      const modelName = availableModels.models[index].name;
      const details = query.data;
      
      if (details?.model_exists && details.model && details.important_tokens_by_label) {
        Object.entries(details.important_tokens_by_label).forEach(([label, tokens]) => {
          // Create unique key with model prefix if needed
          if (!allTokens[label]) {
            allTokens[label] = tokens as Array<{ token: string; count: number }>;
            allDocCounts[label] = details.model?.class_doc_counts?.[label] || 0;
          }
        });
      }
    });

    return { tokens: allTokens, docCounts: allDocCounts };
  }, [selectedModel, availableModels, allModelTrainingQueries]);

  const { data: classifierStatus } = useQuery({
    queryKey: ['classifier-status'],
    queryFn: getClassifierStatus,
    refetchInterval: (query) => {
      const status = query.state.data as ClassifierStatus | undefined;
      return status?.running ? 1000 : false;
    },
  });

  const trainMutation = useMutation({
    mutationFn: async ({ modelName, incremental }: { modelName?: string; incremental?: boolean } = {}) => {
      if (!modelName && availableModels?.models && availableModels.models.length > 0) {
        // Train all available models when "Standaard" is selected
        // We DON'T train the default model because with model folders,
        // it would only see model names as labels (not useful)
        const results = [];
        for (const model of availableModels.models) {
          try {
            const result = await trainClassifier(model.name, incremental || false);
            results.push({ model: model.name, result });
          } catch (error) {
            results.push({ model: model.name, error: String(error) });
          }
        }
        return results;
      }
      // Train specific model or default (when no model folders exist)
      return trainClassifier(modelName, incremental || false);
    },
    onMutate: async () => {
      // Immediately invalidate and refetch status when training starts
      await queryClient.invalidateQueries({ queryKey: ['classifier-status'] });
      await queryClient.refetchQueries({ queryKey: ['classifier-status'] });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['classifier-status'] });
      queryClient.invalidateQueries({ queryKey: ['training-details'] });
      queryClient.invalidateQueries({ queryKey: ['available-models'] });
    },
  });

  // Timer for training progress
  const [trainingElapsed, setTrainingElapsed] = useState(0);
  
  useEffect(() => {
    if (!(trainMutation.isPending || classifierStatus?.running)) {
      setTrainingElapsed(0);
      return;
    }
    
    const startTime = classifierStatus?.started_at ? new Date(classifierStatus.started_at).getTime() : Date.now();
    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      setTrainingElapsed(elapsed);
    }, 1000);
    
    return () => clearInterval(interval);
  }, [trainMutation.isPending, classifierStatus?.running, classifierStatus?.started_at]);

  // Get selected model details
  const selectedModelDetails = useMemo(() => {
    if (!selectedModel || !availableModels?.models) return null;
    return availableModels.models.find(m => m.name === selectedModel);
  }, [selectedModel, availableModels]);

  if (detailsLoading) {
    return (
      <div className="glass-card p-8 text-center">
        <FontAwesomeIcon icon={faRobot} className="text-white/40 text-4xl animate-pulse" />
        <p className="text-white/60 mt-2">Laden...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Model Selector */}
      <div className="glass-card p-4 sm:p-6 bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-white text-lg font-semibold flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
                <FontAwesomeIcon icon={faRobot} className="text-purple-400 w-4 h-4" />
              </div>
              Actief Model
            </h2>
            <p className="text-white/60 text-sm mt-1">
              Dit model wordt gebruikt voor document classificatie
            </p>
          </div>
          
          <div className="relative">
            <button
              onClick={() => setShowModelDropdown(!showModelDropdown)}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 border border-purple-400/30 text-white font-medium transition-all min-w-[160px] justify-between"
            >
              <span className="flex items-center gap-2">
                <FontAwesomeIcon icon={faRobot} className="w-4 h-4 text-purple-400" />
                {selectedModel || 'Standaard'}
              </span>
              <FontAwesomeIcon icon={faChevronDown} className="w-3 h-3 opacity-60" />
            </button>

            {showModelDropdown && (
              <>
                <div
                  className="fixed inset-0 z-[100]"
                  onClick={() => setShowModelDropdown(false)}
                />
                <div
                  className="absolute top-full right-0 mt-1 z-[101] min-w-[200px] rounded-xl border border-white/20 shadow-2xl overflow-hidden"
                  style={{ backgroundColor: '#111827' }}
                >
                  <div className="px-3 py-2 border-b border-white/10 bg-white/5">
                    <span className="text-white/50 text-[10px] uppercase tracking-wide font-medium">Kies Model</span>
                  </div>
                  <button
                    onClick={() => {
                      setSelectedModel(undefined);
                      setShowModelDropdown(false);
                    }}
                    className={`w-full px-4 py-3 text-left text-sm hover:bg-white/10 transition-colors flex items-center justify-between cursor-pointer ${
                      !selectedModel ? 'text-white bg-purple-500/20' : 'text-white/70'
                    }`}
                  >
                    <span>Standaard (alle types)</span>
                    {!selectedModel && <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-purple-400" />}
                  </button>
                  {availableModels?.models?.map((model) => (
                    <button
                      key={model.name}
                      onClick={() => {
                        setSelectedModel(model.name);
                        setShowModelDropdown(false);
                      }}
                      className={`w-full px-4 py-3 text-left text-sm hover:bg-white/10 transition-colors flex items-center justify-between cursor-pointer ${
                        selectedModel === model.name ? 'text-white bg-purple-500/20' : 'text-white/70'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span>{model.name}</span>
                        {model.is_trained && (
                          <span className="text-[10px] bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">trained</span>
                        )}
                      </div>
                      {selectedModel === model.name && <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-purple-400" />}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Selected model details */}
        {selectedModelDetails && (
          <div className="mt-4 pt-4 border-t border-purple-500/20">
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <div className="flex items-center gap-2 text-white/60">
                <FontAwesomeIcon icon={faFolder} className="w-3 h-3" />
                <span className="font-mono text-xs truncate max-w-[300px]" title={selectedModelDetails.path}>
                  {selectedModelDetails.path ? (() => {
                    // Convert absolute path to relative (e.g., /var/www/.../data/backoffice -> data/backoffice)
                    const dataIndex = selectedModelDetails.path.indexOf('/data/');
                    return dataIndex >= 0 ? selectedModelDetails.path.substring(dataIndex + 1) : selectedModelDetails.path;
                  })() : 'N/A'}
                </span>
              </div>
              <span className="text-white/40">•</span>
              <span className="text-purple-300">{selectedModelDetails.document_types?.length || 0} types</span>
              <span className="text-white/40">•</span>
              <span className="text-purple-300">{selectedModelDetails.total_files || 0} bestanden</span>
            </div>
          </div>
        )}
      </div>

      {/* Settings Layout with Sidebar */}
      <div className="flex gap-6">
        {/* Sidebar Navigation */}
        <aside className="hidden lg:block w-64 shrink-0">
          <div className="glass-card p-4 sticky top-4">
            <h3 className="text-white/80 font-semibold text-xs uppercase tracking-wider mb-4">Navigatie</h3>
            <nav className="space-y-1">
              <a
                href="#naive-bayes"
                onClick={(e) => {
                  e.preventDefault();
                  const element = document.getElementById('naive-bayes');
                  if (element) {
                    isProgrammaticScrollRef.current = true;
                    setActiveSection('naive-bayes');
                    const elementTop = element.getBoundingClientRect().top + window.scrollY;
                    const offsetPosition = elementTop - 20; // Small offset for sticky header
                    window.scrollTo({
                      top: offsetPosition,
                      behavior: 'smooth'
                    });
                    setTimeout(() => {
                      isProgrammaticScrollRef.current = false;
                    }, 800);
                  }
                }}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  activeSection === 'naive-bayes'
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon icon={faGraduationCap} className="w-4 h-4" />
                Naive Bayes
              </a>
              <a
                href="#bert"
                onClick={(e) => {
                  e.preventDefault();
                  const element = document.getElementById('bert');
                  if (element) {
                    isProgrammaticScrollRef.current = true;
                    setActiveSection('bert');
                    const elementTop = element.getBoundingClientRect().top + window.scrollY;
                    const offsetPosition = elementTop - 20;
                    window.scrollTo({
                      top: offsetPosition,
                      behavior: 'smooth'
                    });
                    setTimeout(() => {
                      isProgrammaticScrollRef.current = false;
                    }, 800);
                  }
                }}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  activeSection === 'bert'
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon icon={faBrain} className="w-4 h-4" />
                BERT Classifier
              </a>
              <a
                href="#how-extraction-works"
                onClick={(e) => {
                  e.preventDefault();
                  const element = document.getElementById('how-extraction-works');
                  if (element) {
                    isProgrammaticScrollRef.current = true;
                    setActiveSection('how-extraction-works');
                    const elementTop = element.getBoundingClientRect().top + window.scrollY;
                    const offsetPosition = elementTop - 20;
                    window.scrollTo({
                      top: offsetPosition,
                      behavior: 'smooth'
                    });
                    setTimeout(() => {
                      isProgrammaticScrollRef.current = false;
                    }, 800);
                  }
                }}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  activeSection === 'how-extraction-works'
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon icon={faLightbulb} className="w-4 h-4" />
                Hoe werkt extractie?
              </a>
              <a
                href="#training-data"
                onClick={(e) => {
                  e.preventDefault();
                  const element = document.getElementById('training-data');
                  if (element) {
                    isProgrammaticScrollRef.current = true;
                    setActiveSection('training-data');
                    const elementTop = element.getBoundingClientRect().top + window.scrollY;
                    const offsetPosition = elementTop - 20;
                    window.scrollTo({
                      top: offsetPosition,
                      behavior: 'smooth'
                    });
                    setTimeout(() => {
                      isProgrammaticScrollRef.current = false;
                    }, 800);
                  }
                }}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  activeSection === 'training-data'
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon icon={faFileAlt} className="w-4 h-4" />
                Training Data
              </a>
              <a
                href="#classification-priority"
                onClick={(e) => {
                  e.preventDefault();
                  const element = document.getElementById('classification-priority');
                  if (element) {
                    isProgrammaticScrollRef.current = true;
                    setActiveSection('classification-priority');
                    const elementTop = element.getBoundingClientRect().top + window.scrollY;
                    const offsetPosition = elementTop - 20;
                    window.scrollTo({
                      top: offsetPosition,
                      behavior: 'smooth'
                    });
                    setTimeout(() => {
                      isProgrammaticScrollRef.current = false;
                    }, 800);
                  }
                }}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  activeSection === 'classification-priority'
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon icon={faCode} className="w-4 h-4" />
                Classificatie Prioriteit
              </a>
              <a
                href="#stopwords"
                onClick={(e) => {
                  e.preventDefault();
                  const element = document.getElementById('stopwords');
                  if (element) {
                    isProgrammaticScrollRef.current = true;
                    setActiveSection('stopwords');
                    const elementTop = element.getBoundingClientRect().top + window.scrollY;
                    const offsetPosition = elementTop - 20;
                    window.scrollTo({
                      top: offsetPosition,
                      behavior: 'smooth'
                    });
                    setTimeout(() => {
                      isProgrammaticScrollRef.current = false;
                    }, 800);
                  }
                }}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  activeSection === 'stopwords'
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <FontAwesomeIcon icon={faBan} className="w-4 h-4" />
                Stopwoorden
              </a>
            </nav>
          </div>
        </aside>

        {/* Main Content */}
        <div className="flex-1 space-y-6">
      {/* Model Status */}
      <div id="naive-bayes" className="glass-card p-6 scroll-mt-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-white text-xl font-semibold flex items-center gap-2">
            <div className="w-9 h-9 rounded-lg bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faGraduationCap} className="text-purple-400 w-4 h-4" />
            </div>
            {selectedModel ? `Model: ${selectedModel}` : 'Classificatie Model'}
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={() => trainMutation.mutate({ modelName: selectedModel, incremental: false })}
              disabled={trainMutation.isPending || classifierStatus?.running}
              className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
              title={!selectedModel ? "Traint alle modellen (Standaard + alle beschikbare modellen)" : undefined}
            >
              <FontAwesomeIcon 
                icon={faRefresh} 
                className={`w-4 h-4 ${(trainMutation.isPending || classifierStatus?.running) ? 'animate-spin' : ''}`} 
              />
              {classifierStatus?.running ? (
                'Training...'
              ) : !selectedModel ? (
                'Train Alle Modellen'
              ) : (
                `Train ${selectedModel}`
              )}
            </button>
            <button
              onClick={() => trainMutation.mutate({ modelName: selectedModel, incremental: true })}
              disabled={trainMutation.isPending || classifierStatus?.running}
              className="flex items-center gap-2 px-4 py-2 bg-purple-500/50 text-white rounded-lg hover:bg-purple-500/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
              title="Incrementeel trainen (alleen nieuwe/gewijzigde bestanden)"
            >
              <FontAwesomeIcon 
                icon={faRefresh} 
                className={`w-4 h-4 ${(trainMutation.isPending || classifierStatus?.running) ? 'animate-spin' : ''}`} 
              />
              Incrementeel
            </button>
          </div>
        </div>

        {/* Training Progress */}
        {(trainMutation.isPending || classifierStatus?.running) ? (
          <div className="mt-4 p-4 bg-purple-500/10 border border-purple-500/20 rounded-lg">
            <div className="flex items-center gap-3">
              <FontAwesomeIcon icon={faSpinner} className="w-5 h-5 text-purple-400 animate-spin" />
              <div className="flex-1">
                <div className="text-purple-400 font-medium text-sm mb-1">
                  {!selectedModel ? 'Training alle modellen...' : `Training model "${selectedModel}"...`}
                </div>
                <div className="text-white/60 text-xs space-y-1">
                  <div>
                    {classifierStatus?.started_at && (
                      <>Gestart: {new Date(classifierStatus.started_at).toLocaleTimeString('nl-NL')} • </>
                    )}
                    {(() => {
                      const hours = Math.floor(trainingElapsed / 3600);
                      const minutes = Math.floor((trainingElapsed % 3600) / 60);
                      const seconds = trainingElapsed % 60;
                      const parts = [];
                      if (hours > 0) parts.push(`${hours}u`);
                      if (minutes > 0) parts.push(`${minutes}m`);
                      parts.push(`${seconds}s`);
                      return `Looptijd: ${parts.join(' ')}`;
                    })()}
                  </div>
                  {/* Show active files being processed */}
                  {classifierStatus?.active_files && classifierStatus.active_files.length > 0 ? (
                    <div className="space-y-2">
                      <div className="text-purple-300 font-medium text-sm">
                        Actieve bestanden ({classifierStatus.active_files.length}):
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {classifierStatus.active_files.map((fileInfo: any, idx: number) => (
                          <div
                            key={idx}
                            className="px-2 py-1 bg-purple-500/20 border border-purple-500/30 rounded text-xs"
                          >
                            <div className="text-purple-200 font-medium">{fileInfo.label || 'Unknown'}</div>
                            <div className="text-white/70 truncate max-w-[200px]" title={fileInfo.path || fileInfo.file}>
                              {(() => {
                                const path = fileInfo.path || fileInfo.file;
                                if (!path) return fileInfo.file;
                                // Convert absolute path to relative (e.g., /var/www/.../data/backoffice/file.pdf -> data/backoffice/file.pdf)
                                const dataIndex = path.indexOf('/data/');
                                return dataIndex >= 0 ? path.substring(dataIndex + 1) : path.split('/').pop() || path;
                              })()}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (classifierStatus?.current_file || classifierStatus?.current_label) && (
                    <div className="text-purple-300 font-medium">
                      {classifierStatus.current_label && (
                        <span className="mr-2">Type: {classifierStatus.current_label}</span>
                      )}
                      {classifierStatus.current_file && (
                        <span className="text-white/70">Bestand: {(() => {
                          const path = classifierStatus.current_file;
                          if (!path) return '';
                          // Convert absolute path to relative
                          const dataIndex = path.indexOf('/data/');
                          return dataIndex >= 0 ? path.substring(dataIndex + 1) : path.split('/').pop() || path;
                        })()}</span>
                      )}
                      {classifierStatus.ocr_rotation != null && classifierStatus.ocr_rotation !== 0 && (
                        <span className="ml-2 text-blue-300">OCR: {classifierStatus.ocr_rotation}°</span>
                      )}
                    </div>
                  )}
                  <div className="text-white/50">
                    <span>Training stappen:</span>
                    <ul className="list-disc list-inside ml-2 mt-1 space-y-0.5">
                      <li>Scannen van training data</li>
                      <li>Tekst extractie uit documenten</li>
                      <li>Tokenisatie en vocabulaire opbouw</li>
                      <li>Model training (Naive Bayes)</li>
                    </ul>
                  </div>
                  {classifierStatus?.last_error && (
                    <div className="text-red-400 text-xs mt-2 flex items-center gap-1.5">
                      <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                      Fout: {classifierStatus.last_error}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : trainingDetails?.model_exists && trainingDetails.model ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="text-white/60 text-xs mb-1">Status</div>
                <div className="text-green-400 font-semibold flex items-center gap-2">
                  <FontAwesomeIcon icon={faCheck} className="w-3 h-3" />
                  Getraind
                </div>
              </div>
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="text-white/60 text-xs mb-1">Laatste Training</div>
                <div className="text-white font-semibold text-sm">
                  {new Date(trainingDetails.model.updated_at).toLocaleString('nl-NL', {
                    day: '2-digit',
                    month: '2-digit',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </div>
              </div>
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="text-white/60 text-xs mb-1">Vocabulaire</div>
                <div className="text-white font-semibold">
                  {trainingDetails.model.vocab_size.toLocaleString()} tokens
                </div>
              </div>
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="text-white/60 text-xs mb-1">Document Types</div>
                <div className="text-white font-semibold">
                  {trainingDetails.model.labels.length}
                </div>
              </div>
              <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
                <div className="text-purple-300 text-xs mb-1">Threshold</div>
                <div className="text-white font-semibold">
                  {(trainingDetails.model.threshold * 100).toFixed(0)}%
                </div>
                <div className="text-purple-300/60 text-[10px] mt-1">Min. zekerheid</div>
              </div>
              <div className="bg-cyan-500/10 rounded-lg p-4 border border-cyan-500/20">
                <div className="text-cyan-300 text-xs mb-1">Alpha (Smoothing)</div>
                <div className="text-white font-semibold">
                  {trainingDetails.model.alpha}
                </div>
                <div className="text-cyan-300/60 text-[10px] mt-1">Onbekende woorden</div>
              </div>
            </div>

            {/* Model Parameters Explained */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Threshold Explanation */}
              <div className="bg-purple-500/10 rounded-lg p-4 border border-purple-500/20">
                <h4 className="text-purple-300 font-medium mb-2 flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
                    <FontAwesomeIcon icon={faBullseye} className="w-3.5 h-3.5 text-purple-400" />
                  </div>
                  Threshold (Drempel)
                </h4>
                <p className="text-white/70 text-sm mb-2">
                  Hoe zeker moet het model zijn voordat het een classificatie accepteert?
                </p>
                
                {/* Example scenario */}
                <div className="bg-black/20 rounded p-2.5 mb-3 text-xs">
                  <div className="text-purple-300 font-medium mb-1">Voorbeeld:</div>
                  <div className="text-white/70">
                    Model zegt: &quot;Dit is 72% een factuur&quot;
                  </div>
                  <div className="text-white/50 mt-1">
                    <span className="text-green-400">Threshold 70%</span> → Geaccepteerd als factuur<br/>
                    <span className="text-red-400">Threshold 85%</span> → Afgewezen, valt terug op LLM
                  </div>
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-12 text-purple-300 font-mono">30%</span>
                    <span className="text-white/60">→ Soepel: accepteert bijna alles</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-12 text-purple-300 font-mono">50%</span>
                    <span className="text-white/60">→ Gemiddeld: goede balans</span>
                  </div>
                  <div className="flex items-center gap-2 bg-purple-500/20 rounded px-2 py-1">
                    <span className="w-12 text-purple-300 font-mono font-bold">85%</span>
                    <span className="text-white">→ Hoog: alleen zekere matches</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-12 text-purple-300 font-mono">95%</span>
                    <span className="text-white/60">→ Zeer streng: perfecte match</span>
                  </div>
                </div>
                <p className="text-white/50 text-[10px] mt-2 flex items-center gap-1">
                  <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-400" />
                  85% voorkomt foute classificaties, LLM vangt twijfelgevallen op
                </p>
              </div>

              {/* Alpha Explanation */}
              <div className="bg-cyan-500/10 rounded-lg p-4 border border-cyan-500/20">
                <h4 className="text-cyan-300 font-medium mb-2 flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
                    <FontAwesomeIcon icon={faCog} className="w-3.5 h-3.5 text-cyan-400" />
                  </div>
                  Alpha (Smoothing)
                </h4>
                <p className="text-white/70 text-sm mb-2">
                  Hoe gaat het model om met woorden die niet in de training data voorkomen?
                </p>
                
                {/* Example scenario */}
                <div className="bg-black/20 rounded p-2.5 mb-3 text-xs">
                  <div className="text-cyan-300 font-medium mb-1">Voorbeeld:</div>
                  <div className="text-white/70">
                    Factuur bevat &quot;cryptocurrency&quot; maar dat woord zat niet in training
                  </div>
                  <div className="text-white/50 mt-1">
                    <span className="text-red-400">Alpha 0.01</span> → Classificatie faalt volledig<br/>
                    <span className="text-green-400">Alpha 1.0</span> → Woord genegeerd, rest bepaalt
                  </div>
                </div>

                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-12 text-cyan-300 font-mono">0.01</span>
                    <span className="text-white/60">→ Streng: onbekend = grote straf</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-12 text-cyan-300 font-mono">0.5</span>
                    <span className="text-white/60">→ Gemiddeld: kleine straf</span>
                  </div>
                  <div className="flex items-center gap-2 bg-cyan-500/20 rounded px-2 py-1">
                    <span className="w-12 text-cyan-300 font-mono font-bold">1.0</span>
                    <span className="text-white">→ Standaard: neutraal effect</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-12 text-cyan-300 font-mono">2.0+</span>
                    <span className="text-white/60">→ Soepel: woord genegeerd</span>
                  </div>
                </div>
                <p className="text-white/50 text-[10px] mt-2 flex items-center gap-1">
                  <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-400" />
                  1.0 zorgt dat nieuwe woorden de classificatie niet breken
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4 text-center">
            <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faExclamationTriangle} className="text-amber-400 w-6 h-6" />
            </div>
            <p className="text-white">Geen getraind model beschikbaar</p>
            <p className="text-white/60 text-sm mt-1">
              Klik op "Train Model" om het classificatie model te trainen
            </p>
          </div>
        )}
      </div>

      {/* BERT Classifier Section - Under Naive Bayes */}
      <div id="bert" className="mt-6 scroll-mt-4">
        <BertClassifierSection selectedModel={selectedModel} />
      </div>

      {/* How Extraction Works - Explanation */}
      <div id="how-extraction-works" className="scroll-mt-4">
        <div className="glass-card p-6 mb-6">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faLightbulb} className="text-blue-400 w-4 h-4" />
            </div>
            Hoe werkt data extractie?
          </h3>
          
          <div className="space-y-4 text-sm">
            {/* Step 1: Classification */}
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 bg-purple-500/30 rounded-full flex items-center justify-center text-purple-300 text-xs font-bold">1</span>
                <span className="text-white font-medium">Document Classificatie</span>
              </div>
              <p className="text-white/70 ml-8">
                Het systeem herkent eerst het <strong className="text-white">documenttype</strong> (bijv. &quot;factuur&quot;, &quot;loonstrook&quot;, &quot;commitment-agreement&quot;). 
                Dit gebeurt via getrainde AI-modellen (Naive Bayes en BERT) die leren van voorbeelddocumenten.
              </p>
            </div>

            {/* Step 2: Field Extraction */}
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 bg-blue-500/30 rounded-full flex items-center justify-center text-blue-300 text-xs font-bold">2</span>
                <span className="text-white font-medium">Metadata Extractie (LLM)</span>
              </div>
              <p className="text-white/70 ml-8 mb-2">
                Op basis van het documenttype vraagt het systeem aan de LLM om specifieke velden te extraheren. 
                Elk documenttype heeft zijn eigen velden gedefinieerd in <strong className="text-white">Document Types</strong>.
              </p>
              <div className="ml-8 bg-black/20 rounded p-3 font-mono text-xs text-white/60">
                <div>Voorbeeld &quot;commitment-agreement&quot;:</div>
                <div className="text-blue-300 mt-1">→ participant, fondsmanager, commitment (bedrag), datum, adres</div>
              </div>
            </div>

            {/* Step 3: Evidence Finding */}
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 bg-emerald-500/30 rounded-full flex items-center justify-center text-emerald-300 text-xs font-bold">3</span>
                <span className="text-white font-medium">Evidence Zoeken (Alle Pagina&apos;s)</span>
              </div>
              <p className="text-white/70 ml-8 mb-2">
                Voor elke geëxtraheerde waarde doorzoekt het systeem <strong className="text-white">ALLE pagina&apos;s</strong> van het document om te vinden waar de waarde voorkomt:
              </p>
              <ul className="ml-8 text-white/60 space-y-1">
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">•</span>
                  <span>Exacte tekstmatch (bijv. &quot;P.C.M. Vastgoed Holding B.V.&quot;)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">•</span>
                  <span>Genormaliseerde match (witruimte-verschillen door OCR)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">•</span>
                  <span>Numerieke formaten (100000 → &quot;100.000&quot; → &quot;€ 100.000,-&quot;)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">•</span>
                  <span>Case-insensitive matching</span>
                </li>
              </ul>
            </div>

            {/* Step 4: PDF Highlighting */}
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 bg-yellow-500/30 rounded-full flex items-center justify-center text-yellow-300 text-xs font-bold">4</span>
                <span className="text-white font-medium">PDF Highlighting</span>
              </div>
              <p className="text-white/70 ml-8">
                In de PDF viewer worden alle gevonden evidence-locaties <strong className="text-blue-300">blauw gemarkeerd</strong>. 
                Onderaan zie je knoppen om snel naar pagina&apos;s met evidence te navigeren.
              </p>
            </div>

            {/* Tip */}
            <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3 flex items-start gap-3">
              <FontAwesomeIcon icon={faLightbulb} className="text-yellow-400 w-4 h-4 mt-0.5" />
              <div className="text-white/80 text-xs">
                <strong className="text-yellow-300">Tip:</strong> Als een veld &quot;geen bewijs&quot; toont maar de waarde WEL in het document staat,
                kan dit komen door OCR-fouten of afwijkende opmaak. Het systeem probeert automatisch verschillende formaten te matchen.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Data per Label - Scalable */}
      <div id="training-data" className="scroll-mt-4">
      {(() => {
        // When "Standaard" is selected, show all document types from all models
        if (!selectedModel && availableModels?.models) {
          // Combine all document types from all models
          const allLabels: string[] = [];
          const allDocCounts: Record<string, number> = {};
          const allTrainingFiles: Record<string, Array<{ path: string; sha256: string; updated_at: string; }>> = {};
          const allTokens: Record<string, TokenInfo[]> = {};
          
          // Get all model names to filter them out
          const modelNames = new Set(availableModels.models.map(m => m.name.toLowerCase()));
          
          availableModels.models.forEach(model => {
            model.document_types?.forEach(dt => {
              // Filter out model names that appear as document types
              const slugLower = dt.slug.toLowerCase();
              if (!modelNames.has(slugLower) && !allLabels.includes(dt.slug)) {
                allLabels.push(dt.slug);
                // We don't have detailed counts per model, so use file_count as approximation
                allDocCounts[dt.slug] = dt.file_count || 0;
              }
            });
          });
          
          // Use aggregated training details from all model-specific files
          // This gives us the actual document type tokens, not model folder tokens
          if (aggregatedTrainingDetails) {
            Object.entries(aggregatedTrainingDetails.tokens).forEach(([label, tokens]) => {
              const labelLower = label.toLowerCase();
              // Skip if this label is a model name
              if (modelNames.has(labelLower)) {
                return;
              }
              
              allTokens[label] = tokens;
              if (!allLabels.includes(label)) {
                allLabels.push(label);
              }
              allDocCounts[label] = aggregatedTrainingDetails.docCounts[label] || allDocCounts[label] || 0;
            });
          }
          
          return (
            <TrainedLabelsGrid
              labels={allLabels}
              docCounts={allDocCounts}
              trainingFilesByLabel={allTrainingFiles}
              tokensByLabel={allTokens}
              modelName={undefined}
              allModelsData={availableModels.models}
            />
          );
        }
        
        // When a specific model is selected, show only that model's data
        if (trainingDetails?.model_exists && trainingDetails.model) {
          // Filter out model names from labels
          const modelNames = new Set((availableModels?.models || []).map(m => m.name.toLowerCase()));
          const filteredLabels = trainingDetails.model.labels.filter(
            label => !modelNames.has(label.toLowerCase())
          );
          
          return (
            <TrainedLabelsGrid
              labels={filteredLabels}
              docCounts={trainingDetails.model.class_doc_counts}
              trainingFilesByLabel={trainingDetails.training_files_by_label}
              tokensByLabel={trainingDetails.important_tokens_by_label}
              modelName={selectedModel}
              allModelsData={availableModels?.models}
            />
          );
        }
        
        return null;
      })()}
      </div>

      {/* Classification Priority Info */}
      <div id="classification-priority" className="glass-card p-6 scroll-mt-4">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
            <FontAwesomeIcon icon={faCode} className="text-cyan-400 w-4 h-4" />
          </div>
          Classificatie Methodes & Prioriteit
        </h3>
        <p className="text-white/60 text-sm mb-4">
          Het systeem gebruikt meerdere methodes om documenten te classificeren. De volgorde bepaalt welke methode voorrang krijgt.
        </p>
        <div className="space-y-3">
          <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
            <div className="w-8 h-8 bg-red-500/20 rounded-full flex items-center justify-center text-red-400 text-sm font-bold flex-shrink-0">1</div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-white font-medium">Deterministic Matching (STRONG)</span>
                <span className="text-[10px] bg-red-500/30 text-red-300 px-1.5 py-0.5 rounded uppercase font-semibold">Hoogste Prioriteit</span>
              </div>
              <div className="text-white/70 text-sm mb-2">
                <strong>Keywords & Regex</strong> - Als <strong>alle</strong> kw: regels matchen, heeft dit voorrang boven alle andere methodes.
              </div>
              <div className="text-white/50 text-xs flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400" /> Snel & voorspelbaar</span>
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400" /> 100% match</span>
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-400" /> Handmatig</span>
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
            <div className="w-8 h-8 bg-green-500/20 rounded-full flex items-center justify-center text-green-400 text-sm font-bold flex-shrink-0">2</div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-white font-medium">Trained Models</span>
                <span className="text-[10px] bg-green-500/30 text-green-300 px-1.5 py-0.5 rounded uppercase font-semibold">Primair</span>
              </div>
              <div className="text-white/70 text-sm mb-2">
                <strong>Naive Bayes & BERT</strong> - Beide worden uitgevoerd, de beste wordt gekozen. BERT wint als het significant beter is (+0.1 confidence).
              </div>
              <div className="text-white/50 text-xs flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400" /> Leert automatisch</span>
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400" /> Confidence score</span>
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400" /> Verbetert met data</span>
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <div className="w-8 h-8 bg-blue-500/20 rounded-full flex items-center justify-center text-blue-400 text-sm font-bold flex-shrink-0">3</div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-white font-medium">LLM Classificatie</span>
                <span className="text-[10px] bg-blue-500/30 text-blue-300 px-1.5 py-0.5 rounded uppercase font-semibold">Laatste Resort</span>
              </div>
              <div className="text-white/70 text-sm mb-2">
                <strong>AI Language Model</strong> - Gebruikt een groot taalmodel om document type te bepalen.
              </div>
              <div className="text-white/50 text-xs flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400" /> Begrijpt context</span>
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400" /> Geen training nodig</span>
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-400" /> Langzamer</span>
                <span className="inline-flex items-center gap-1"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-400" /> Hogere kosten</span>
              </div>
            </div>
          </div>
        </div>

        {/* Future Training Methods */}
        <div className="mt-6 p-4 bg-white/5 border border-white/10 rounded-lg">
          <h4 className="text-white/80 font-medium mb-2 flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faBrain} className="w-3 h-3 text-cyan-400" />
            </div>
            Toekomstige Training Methodes
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            <div className="text-white/50">
              <span className="text-white/70">TF-IDF + SVM:</span> Betere prestaties bij grote datasets
            </div>
            <div className="text-white/50">
              <span className="text-white/70">Ensemble:</span> Combinatie van meerdere modellen
            </div>
            <div className="text-white/50">
              <span className="text-white/70">Fine-tuned LLM:</span> Aangepast taalmodel voor jouw data
            </div>
          </div>
        </div>
      </div>

      {/* Stopwords & Training Settings */}
      <div id="stopwords" className="glass-card p-6 scroll-mt-4">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-amber-500/20 border border-amber-500/30 flex items-center justify-center">
            <FontAwesomeIcon icon={faBan} className="text-amber-400 w-4 h-4" />
          </div>
          Stopwoorden & Training Instellingen
        </h3>
        
        <div className="space-y-4">
          <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
            <p className="text-white/70 text-sm mb-3">
              Stopwoorden zijn veelvoorkomende woorden die geen betekenis toevoegen aan de classificatie.
              Deze worden automatisch genegeerd tijdens training.
            </p>
            <div className="bg-black/20 rounded-lg p-3">
              <p className="text-white/50 text-xs font-mono mb-2">Nederlandse stopwoorden (standaard actief):</p>
              <div className="flex flex-wrap gap-1">
                {['de', 'het', 'een', 'en', 'van', 'in', 'is', 'op', 'te', 'dat', 'die', 'voor', 'met', 'zijn', 'aan', 'niet', 'ook', 'als', 'maar', 'om'].map(word => (
                  <span key={word} className="text-xs bg-amber-500/20 text-amber-300 px-1.5 py-0.5 rounded font-mono">
                    {word}
                  </span>
                ))}
                <button
                  onClick={() => setShowStopwordsModal(true)}
                  className="text-xs text-amber-300 hover:text-amber-100 px-1.5 py-0.5 hover:bg-amber-500/30 rounded transition-colors cursor-pointer font-medium underline decoration-amber-400/50 hover:decoration-amber-300"
                  title={`Bekijk alle ${totalStopwords} stopwoorden`}
                >
                  +{totalStopwords - 20} meer...
                </button>
              </div>
            </div>
            <p className="text-amber-300/60 text-xs mt-3 flex items-center gap-1.5">
              <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-400" />
              Stopwoorden configuratie wordt binnenkort configureerbaar via een bestand of API.
            </p>
          </div>

        </div>
      </div>

      {/* Stopwords Modal */}
      {showStopwordsModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 border border-white/20 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            <div className="p-6 border-b border-white/10 flex items-center justify-between">
              <div>
                <h3 className="text-white font-bold text-lg">Nederlandse Stopwoorden</h3>
                <p className="text-white/60 text-sm mt-1">
                  Totaal {totalStopwords} woorden die automatisch worden genegeerd tijdens training
                </p>
              </div>
              <button
                onClick={() => setShowStopwordsModal(false)}
                className="text-white/60 hover:text-white transition-colors"
              >
                <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6">
              <div className="space-y-6">
                {Object.entries(dutchStopwords).map(([category, words]) => (
                  <div key={category}>
                    <h4 className="text-amber-400 font-semibold text-sm mb-3 flex items-center gap-2">
                      <FontAwesomeIcon icon={faBan} className="w-3 h-3" />
                      {category} ({words.length})
                    </h4>
                    <div className="flex flex-wrap gap-1.5">
                      {words.map(word => (
                        <span
                          key={word}
                          className="text-xs bg-amber-500/20 text-amber-300 px-2 py-1 rounded font-mono border border-amber-500/30"
                        >
                          {word}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="p-4 border-t border-white/10 bg-white/5">
              <button
                onClick={() => setShowStopwordsModal(false)}
                className="w-full px-4 py-2 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 rounded-lg transition-colors font-medium"
              >
                Sluiten
              </button>
            </div>
          </div>
        </div>
      )}
        </div>
      </div>
    </div>
  );
}

// BERT Classifier Section Component
function BertClassifierSection({ selectedModel }: { selectedModel?: string }) {
  const queryClient = useQueryClient();
  const [bertThreshold, setBertThreshold] = useState(0.7);
  const [selectedBertLabel, setSelectedBertLabel] = useState<string | null>(null);

  const { data: bertStatus } = useQuery({
    queryKey: ['bert-classifier-status', selectedModel],
    queryFn: () => getBertClassifierStatus(selectedModel),
    refetchInterval: (query) => {
      const status = query.state.data as BertClassifierStatus | undefined;
      return status?.running ? 1000 : false;
    },
  });

  const { data: availableModels } = useQuery({
    queryKey: ['available-models'],
    queryFn: getAvailableModels,
  });

  const trainBertMutation = useMutation({
    mutationFn: async ({ incremental }: { incremental?: boolean } = {}) => {
      if (!selectedModel && availableModels?.models && availableModels.models.length > 0) {
        // Train all available models when "Standaard" is selected
        // We DON'T train the default model because with model folders,
        // it would only see model names as labels (not useful)
        const results = [];
        for (const model of availableModels.models) {
          try {
            const result = await trainBertClassifier(model.name, bertThreshold, incremental || false);
            results.push({ model: model.name, result });
          } catch (error) {
            results.push({ model: model.name, error: String(error) });
          }
        }
        return results;
      }
      // Train specific model or default (when no model folders exist)
      return trainBertClassifier(selectedModel, bertThreshold, incremental || false);
    },
    onSuccess: () => {
      // Invalidate both with and without model name
      queryClient.invalidateQueries({ queryKey: ['bert-classifier-status'] });
      queryClient.invalidateQueries({ queryKey: ['bert-classifier-status', selectedModel] });
      queryClient.invalidateQueries({ queryKey: ['available-models'] });
    },
    onError: (error: any) => {
      console.error('BERT training error:', error);
    },
  });

  // Timer for BERT training progress
  const [bertTrainingElapsed, setBertTrainingElapsed] = useState(0);
  
  useEffect(() => {
    if (!(trainBertMutation.isPending || bertStatus?.running)) {
      setBertTrainingElapsed(0);
      return;
    }
    
    const startTime = bertStatus?.started_at ? new Date(bertStatus.started_at).getTime() : Date.now();
    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTime) / 1000);
      setBertTrainingElapsed(elapsed);
    }, 1000);
    
    return () => clearInterval(interval);
  }, [trainBertMutation.isPending, bertStatus?.running, bertStatus?.started_at]);

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
            <FontAwesomeIcon icon={faBrain} className="text-blue-400 w-4 h-4" />
          </div>
          BERT Embeddings Classifier
          <span className="text-xs bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded-full font-normal">
            Experimenteel
          </span>
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => trainBertMutation.mutate({ incremental: false })}
            disabled={trainBertMutation.isPending || bertStatus?.running}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            <FontAwesomeIcon
              icon={faRefresh}
              className={`w-4 h-4 ${(trainBertMutation.isPending || bertStatus?.running) ? 'animate-spin' : ''}`}
            />
            {bertStatus?.running ? (
              'Training...'
            ) : !selectedModel ? (
              'Train BERT (Alle Modellen)'
            ) : (
              'Train BERT'
            )}
          </button>
          <button
            onClick={() => trainBertMutation.mutate({ incremental: true })}
            disabled={trainBertMutation.isPending || bertStatus?.running}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500/50 text-white rounded-lg hover:bg-blue-500/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
            title="Incrementeel trainen (alleen nieuwe/gewijzigde bestanden)"
          >
            <FontAwesomeIcon
              icon={faRefresh}
              className={`w-4 h-4 ${(trainBertMutation.isPending || bertStatus?.running) ? 'animate-spin' : ''}`}
            />
            Incrementeel
          </button>
        </div>
      </div>

      <p className="text-white/60 text-sm mb-4">
        BERT gebruikt deep learning voor semantisch tekstbegrip. Het begrijpt de <em>betekenis</em> van woorden
        in context, niet alleen hun frequentie.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Status */}
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-white/60 text-xs mb-1">Status</div>
          {bertStatus?.model_exists ? (
            <div className="text-green-400 font-semibold flex items-center gap-2">
              <FontAwesomeIcon icon={faCheck} className="w-3 h-3" />
              Getraind
            </div>
          ) : (
            <div className="text-yellow-400 font-semibold flex items-center gap-2">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
              Niet getraind
            </div>
          )}
        </div>

        {/* Model */}
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-white/60 text-xs mb-1">BERT Model</div>
          <div className="text-white font-mono text-xs truncate" title={bertStatus?.bert_model}>
            {bertStatus?.bert_model || 'multilingual-MiniLM'}
          </div>
          <p className="text-white/40 text-[10px] mt-2 leading-relaxed">
            {bertStatus?.bert_model?.includes('robbert') || bertStatus?.bert_model?.includes('NetherlandsForensicInstitute')
              ? 'Ontwikkeld door het Nederlands Forensisch Instituut (NFI). Gespecialiseerd in Nederlandse juridische en zakelijke documenten. Herkent o.a. contracten, facturen, ID-documenten, bankafschriften en officiële correspondentie.'
              : 'Dit model zet tekst om naar numerieke vectoren die de betekenis vastleggen, waardoor documenten vergeleken kunnen worden op inhoud.'}
          </p>
        </div>

        {/* Labels */}
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-white/60 text-xs mb-1">Document Types</div>
          <div className="text-white font-semibold">
            {bertStatus?.labels?.length || 0}
          </div>
        </div>

        {/* Threshold */}
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-white/60 text-xs mb-1">Threshold</div>
          <div className="text-white font-semibold">
            {Math.round((bertStatus?.threshold || bertThreshold) * 100)}%
          </div>
        </div>
      </div>

      {/* Training Info */}
      {bertStatus?.last_summary && (
        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
          <div className="text-green-400 text-sm font-medium mb-2">Laatste Training</div>
          {bertStatus.finished_at && (
            <div className="text-white/60 text-xs mb-2">
              {new Date(bertStatus.finished_at).toLocaleString('nl-NL', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              })}
            </div>
          )}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
            <div title="Totaal aantal documenten gebruikt voor training">
              <span className="text-white/50">Documenten:</span>
              <span className="text-white ml-1">{bertStatus.last_summary.total_documents}</span>
            </div>
            <div title="Aantal document types dat BERT kan herkennen">
              <span className="text-white/50">Types:</span>
              <span className="text-white ml-1">{bertStatus.last_summary.labels?.length || 0}</span>
            </div>
            <div 
              title="Dimensie van de BERT vector: elk document wordt omgezet naar 768 getallen die de 'betekenis' van het document representeren. Dit is een vast kenmerk van het BERT model."
              className="cursor-help"
            >
              <span className="text-white/50">Vector grootte:</span>
              <span className="text-white ml-1">{bertStatus.last_summary.embedding_dim}</span>
            </div>
            <div title="Minimale zekerheid voordat BERT een classificatie accepteert">
              <span className="text-white/50">Threshold:</span>
              <span className="text-white ml-1">{Math.round(bertStatus.last_summary.threshold * 100)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* BERT Model Summary - Compact */}
      {bertStatus?.model_exists && bertStatus?.last_summary && (
        <div className="mt-4 p-4 bg-gradient-to-br from-blue-500/15 to-purple-500/10 border border-blue-500/30 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
                <FontAwesomeIcon icon={faBrain} className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <h3 className="text-blue-300 font-bold text-sm">BERT Model</h3>
                {bertStatus.finished_at && (
                  <p className="text-white/50 text-xs">
                    Getraind: {new Date(bertStatus.finished_at).toLocaleDateString('nl-NL', {
                      day: '2-digit',
                      month: '2-digit',
                      year: 'numeric'
                    })}
                  </p>
                )}
              </div>
            </div>
            <div className="flex gap-2 text-xs">
              <div className="bg-white/5 rounded px-2 py-1 border border-white/10">
                <span className="text-white/50">Types:</span>
                <span className="text-white font-bold ml-1">{bertStatus.last_summary.labels?.length || 0}</span>
              </div>
              <div className="bg-white/5 rounded px-2 py-1 border border-white/10">
                <span className="text-white/50">Docs:</span>
                <span className="text-white font-bold ml-1">{bertStatus.last_summary.total_documents || 0}</span>
              </div>
              <div className="bg-white/5 rounded px-2 py-1 border border-white/10">
                <span className="text-white/50">Threshold:</span>
                <span className="text-white font-bold ml-1">{Math.round((bertStatus.last_summary.threshold || 0.7) * 100)}%</span>
              </div>
            </div>
          </div>
          
          {/* How BERT works explanation */}
          <details className="mt-2">
            <summary className="text-white/70 text-xs cursor-pointer hover:text-white transition-colors flex items-center gap-1">
              <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-400" />
              Hoe werkt BERT classificatie?
            </summary>
            <div className="mt-2 pt-2 border-t border-white/10 text-xs space-y-2">
              <div className="bg-black/20 rounded p-3">
                <div className="text-blue-300 font-medium mb-2">BERT zet tekst om naar vectoren</div>
                <p className="text-white/60 mb-2">
                  Elk document wordt omgezet naar een &quot;vector&quot; van <strong className="text-white">{bertStatus.last_summary.embedding_dim || 768} getallen</strong>. 
                  Deze getallen representeren de &quot;betekenis&quot; van het document.
                </p>
                <div className="bg-black/30 rounded p-2 font-mono text-[10px] text-white/50">
                  &quot;Factuur voor levering...&quot; → [0.23, -0.87, 0.45, ... 768 getallen]
                </div>
              </div>
              <div className="bg-black/20 rounded p-3">
                <div className="text-blue-300 font-medium mb-2">Vergelijking via afstand</div>
                <p className="text-white/60">
                  BERT vergelijkt de vector van een nieuw document met de vectoren van bekende types. 
                  Het type met de kleinste &quot;afstand&quot; wint (als boven threshold).
                </p>
              </div>
              <div className="text-white/40 text-[10px] flex items-center gap-1">
                <FontAwesomeIcon icon={faInfoCircle} className="w-3 h-3" />
                768 dimensies is standaard voor het &quot;all-MiniLM-L6-v2&quot; BERT model
              </div>
            </div>
          </details>

          {bertStatus.last_summary.samples_per_label && Object.keys(bertStatus.last_summary.samples_per_label).length > 0 && (
            <details className="mt-2">
              <summary className="text-white/70 text-xs cursor-pointer hover:text-white transition-colors">
                Document types ({Object.keys(bertStatus.last_summary.samples_per_label).length}) •
                Semantisch begrip • Beter dan NB bij variaties
              </summary>
              <div className="mt-2 pt-2 border-t border-white/10">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                  {Object.entries(bertStatus.last_summary.samples_per_label)
                    .sort(([, a], [, b]) => (b as number) - (a as number))
                    .map(([label, count]) => {
                      const numCount = count as number;
                      const quality = numCount >= 10 ? 'good' : numCount >= 5 ? 'medium' : 'low';
                      return (
                        <button 
                          key={label} 
                          onClick={() => setSelectedBertLabel(label)}
                          className="bg-white/5 rounded px-2 py-1.5 border border-white/10 flex items-center justify-between hover:bg-white/10 hover:border-cyan-500/30 transition-colors cursor-pointer text-left"
                        >
                          <span className="text-white/80 truncate">{label}</span>
                          <span className={`font-semibold shrink-0 ml-1 ${
                            quality === 'good' ? 'text-green-400' : 
                            quality === 'medium' ? 'text-yellow-400' : 
                            'text-red-400'
                          }`}>{count}</span>
                        </button>
                      );
                    })}
                </div>
              </div>
            </details>
          )}

          {/* BERT Label Detail Modal */}
          {selectedBertLabel && bertStatus?.last_summary?.samples_per_label && (
            <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setSelectedBertLabel(null)}>
              <div className="bg-gray-900 border border-white/20 rounded-xl p-6 max-w-lg w-full shadow-2xl" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-white font-semibold flex items-center gap-2">
                    <FontAwesomeIcon icon={faBrain} className="text-cyan-400 w-5 h-5" />
                    Model: {selectedBertLabel}
                  </h3>
                  <button onClick={() => setSelectedBertLabel(null)} className="text-white/60 hover:text-white p-1">
                    <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
                  </button>
                </div>

                {(() => {
                  const count = bertStatus.last_summary.samples_per_label[selectedBertLabel] as number;
                  const quality = count >= 10 ? 'good' : count >= 5 ? 'medium' : 'low';
                  const modelName = selectedModel || 'backoffice';
                  return (
                    <div className="space-y-4">
                      {/* Count display */}
                      <div className="bg-white/5 rounded-lg p-4 border border-white/10 text-center">
                        <div className={`text-4xl font-bold ${
                          quality === 'good' ? 'text-green-400' : 
                          quality === 'medium' ? 'text-yellow-400' : 
                          'text-red-400'
                        }`}>{count}</div>
                        <div className="text-white/60 text-sm mt-1">training documenten</div>
                      </div>

                      {/* What does this mean */}
                      <div className="space-y-3">
                        <h4 className="text-white/80 font-medium text-sm">Wat betekent dit getal?</h4>
                        <p className="text-white/60 text-sm">
                          In de map <code className="bg-black/30 px-1 py-0.5 rounded text-cyan-300 text-xs">data/{selectedBertLabel}/</code> staan 
                          <strong className="text-white"> {count} PDF&apos;s</strong> verdeeld over verschillende document type mappen 
                          (bijv. <code className="bg-black/30 px-1 py-0.5 rounded text-white/60 text-xs">offerte/</code>, 
                          <code className="bg-black/30 px-1 py-0.5 rounded text-white/60 text-xs">factuur/</code>, etc.).
                        </p>
                        <p className="text-white/60 text-sm">
                          BERT leert hiermee welke documenten bij het model <strong className="text-cyan-300">{selectedBertLabel}</strong> horen 
                          en kan nieuwe documenten automatisch classificeren naar het juiste document type.
                        </p>
                      </div>

                      {/* Quality indicator */}
                      <div className={`rounded-lg p-3 border ${
                        quality === 'good' ? 'bg-green-500/10 border-green-500/30' : 
                        quality === 'medium' ? 'bg-yellow-500/10 border-yellow-500/30' : 
                        'bg-red-500/10 border-red-500/30'
                      }`}>
                        <div className="flex items-center gap-2">
                          <FontAwesomeIcon 
                            icon={quality === 'good' ? faCheck : quality === 'medium' ? faExclamationTriangle : faExclamationTriangle} 
                            className={`w-4 h-4 ${
                              quality === 'good' ? 'text-green-400' : 
                              quality === 'medium' ? 'text-yellow-400' : 
                              'text-red-400'
                            }`} 
                          />
                          <span className={`font-medium text-sm ${
                            quality === 'good' ? 'text-green-300' : 
                            quality === 'medium' ? 'text-yellow-300' : 
                            'text-red-300'
                          }`}>
                            {quality === 'good' ? 'Voldoende training data' : 
                             quality === 'medium' ? 'Matig - meer voorbeelden aanbevolen' : 
                             'Onvoldoende - voeg meer voorbeelden toe'}
                          </span>
                        </div>
                        <p className="text-white/50 text-xs mt-2">
                          {quality === 'good'
                            ? 'Met 10+ documenten per document type kan BERT betrouwbaar classificeren.'
                            : quality === 'medium'
                            ? 'Met 5-9 documenten werkt BERT redelijk, maar meer voorbeelden per type verbeteren nauwkeurigheid.'
                            : 'Met minder dan 5 documenten per type is BERT onbetrouwbaar. Voeg meer voorbeelden toe.'}
                        </p>
                      </div>

                      {/* Folder structure explanation */}
                      <div className="bg-cyan-500/10 border border-cyan-500/20 rounded-lg p-3">
                        <div className="text-cyan-300 text-xs font-medium mb-2">Folder structuur:</div>
                        <div className="bg-black/30 rounded p-2 font-mono text-[10px] text-white/70 space-y-0.5">
                          <div className="text-purple-300">data/</div>
                          <div className="ml-3 text-blue-300">{selectedBertLabel}/ <span className="text-white/40">← model ({count} PDF&apos;s totaal)</span></div>
                          <div className="ml-6 text-cyan-300">offerte/ <span className="text-white/40">← document type</span></div>
                          <div className="ml-9 text-white/50">offerte1.pdf, offerte2.pdf, ...</div>
                          <div className="ml-6 text-cyan-300">factuur/ <span className="text-white/40">← document type</span></div>
                          <div className="ml-9 text-white/50">factuur1.pdf, factuur2.pdf, ...</div>
                          <div className="ml-6 text-white/40">... meer document types</div>
                        </div>
                        <p className="text-white/50 text-[10px] mt-2">
                          Het totaal van {count} PDF&apos;s is de som van alle bestanden in alle document type mappen.
                        </p>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {(bertStatus?.last_error || trainBertMutation.error) && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
          <div className="text-red-400 text-sm font-medium mb-1">Error</div>
          <div className="text-red-300/70 text-xs">
            {trainBertMutation.error?.message || bertStatus?.last_error || 'Onbekende fout'}
          </div>
          {trainBertMutation.error && (
            <button
              onClick={() => trainBertMutation.reset()}
              className="mt-2 text-xs text-red-400 hover:text-red-300 underline"
            >
              Reset error
            </button>
          )}
        </div>
      )}

      {/* Training Progress */}
      {(trainBertMutation.isPending || bertStatus?.running) ? (
        <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <div className="flex items-center gap-3">
            <FontAwesomeIcon icon={faSpinner} className="w-5 h-5 text-blue-400 animate-spin" />
            <div className="flex-1">
              <div className="text-blue-400 font-medium text-sm mb-1">
                {!selectedModel ? 'Training alle BERT modellen...' : `Training BERT model "${selectedModel}"...`}
              </div>
              <div className="text-white/60 text-xs space-y-1">
                <div>
                  {bertStatus?.started_at && (
                    <>Gestart: {new Date(bertStatus.started_at).toLocaleTimeString('nl-NL')} • </>
                  )}
                  {(() => {
                    const hours = Math.floor(bertTrainingElapsed / 3600);
                    const minutes = Math.floor((bertTrainingElapsed % 3600) / 60);
                    const seconds = bertTrainingElapsed % 60;
                    const parts = [];
                    if (hours > 0) parts.push(`${hours}u`);
                    if (minutes > 0) parts.push(`${minutes}m`);
                    parts.push(`${seconds}s`);
                    return `Looptijd: ${parts.join(' ')}`;
                  })()}
                </div>
                <div className="text-white/50">
                  <span>Training stappen:</span>
                  <ul className="list-none ml-2 mt-1 space-y-0.5">
                    <li className="flex items-center gap-1.5">
                      {bertStatus?.model_downloaded ? (
                        <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400 shrink-0" />
                      ) : (
                        <span className="w-3 h-3 shrink-0" />
                      )}
                      <span>BERT model {bertStatus?.model_downloaded ? 'al gedownload' : 'download (~30-60s bij eerste keer)'}</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="w-3 h-3 shrink-0" />
                      <span>Tekst extractie uit documenten</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="w-3 h-3 shrink-0" />
                      <span>BERT embeddings berekenen (kan lang duren bij veel documenten)</span>
                    </li>
                    <li className="flex items-center gap-1.5">
                      <span className="w-3 h-3 shrink-0" />
                      <span>Cosine similarity berekenen per document type</span>
                    </li>
                  </ul>
                </div>
                {bertTrainingElapsed > 120 && (
                  <div className="text-amber-400 text-xs mt-2 flex items-center gap-1.5">
                    <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                    Training duurt langer dan verwacht. Check de backend logs voor details.
                  </div>
                )}
                {bertStatus?.last_error && (
                  <div className="text-red-400 text-xs mt-2 flex items-center gap-1.5">
                    <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                    Fout: {bertStatus.last_error}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* Threshold Slider */}
      <div className="mt-4 p-4 bg-white/5 border border-white/10 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="text-white/70 text-sm">Similarity Threshold</span>
          <span className="text-blue-400 font-mono text-sm">{Math.round(bertThreshold * 100)}%</span>
        </div>
        <input
          type="range"
          min="0.5"
          max="0.95"
          step="0.05"
          value={bertThreshold}
          onChange={(e) => setBertThreshold(parseFloat(e.target.value))}
          className="w-full accent-blue-500"
        />
        <div className="flex justify-between text-[10px] text-white/40 mt-1">
          <span>50% - Meer matches</span>
          <span>95% - Alleen zekere matches</span>
        </div>
      </div>

      {/* Comparison with Naive Bayes - Compact */}
      <details className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <summary className="text-white/80 font-medium text-sm cursor-pointer hover:text-white transition-colors">
          BERT vs Naive Bayes
        </summary>
        <div className="mt-3">
          <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-blue-400 font-medium mb-2 flex items-center gap-2">
              <div className="w-5 h-5 rounded bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
                <FontAwesomeIcon icon={faBrain} className="w-2.5 h-2.5" />
              </div>
              BERT
            </div>
            <ul className="text-white/60 space-y-1.5">
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400 shrink-0" /> Begrijpt context</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400 shrink-0" /> Synoniemen</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400 shrink-0" /> Weinig data</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-400 shrink-0" /> ~100ms</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-400 shrink-0" /> ~500MB RAM</li>
            </ul>
          </div>
          <div>
            <div className="text-purple-400 font-medium mb-2 flex items-center gap-2">
              <div className="w-5 h-5 rounded bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
                <FontAwesomeIcon icon={faBolt} className="w-2.5 h-2.5" />
              </div>
              Naive Bayes
            </div>
            <ul className="text-white/60 space-y-1.5">
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400 shrink-0" /> Zeer snel (~1ms)</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400 shrink-0" /> Weinig geheugen</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400 shrink-0" /> Veel data</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-400 shrink-0" /> Mist context</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-400 shrink-0" /> Woordfrequentie</li>
            </ul>
          </div>
        </div>
        </div>
      </details>
    </div>
  );
}

// Fraud Detection Tab
function FraudDetectionTab() {
  const queryClient = useQueryClient();
  
  // App settings for ELA/EXIF
  const { data: appSettings } = useQuery({
    queryKey: ['app-settings'],
    queryFn: () => getAppSettings(),
  });
  
  const elaEnabled = appSettings?.find(s => s.key === 'ela_enabled')?.value === 'true' || false;
  const exifEnabled = appSettings?.find(s => s.key === 'exif_enabled')?.value === 'true' || false;
  
  const updateSettingMutation = useMutation({
    mutationFn: ({ key, value }: { key: string; value: string }) => updateAppSetting(key, value),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['app-settings'] });
    },
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-red-500/20 border border-red-500/30 flex items-center justify-center">
            <FontAwesomeIcon icon={faShieldAlt} className="text-red-400 w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Fraud Detection Instellingen</h2>
            <p className="text-white/60 text-sm">Configureer fraud detection analyse opties</p>
          </div>
        </div>
      </div>

      {/* Settings */}
      <div className="glass-card p-6">
        <h3 className="text-white font-semibold text-sm mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faShieldAlt} className="w-4 h-4 text-red-400" />
          Analyse Opties
        </h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/10">
            <div className="flex items-center gap-4 flex-1">
              <div className="w-12 h-12 rounded-lg bg-red-500/20 border border-red-500/30 flex items-center justify-center">
                <FontAwesomeIcon icon={faImage} className="w-6 h-6 text-red-400" />
              </div>
              <div className="flex-1">
                <div className="text-white font-medium text-base mb-1">Error Level Analysis (ELA)</div>
                <div className="text-white/60 text-sm">
                  Detecteert JPEG manipulatie via compressie inconsistenties. Standaard uit (vaak ruis bij scans).
                </div>
              </div>
            </div>
            <button
              onClick={() => updateSettingMutation.mutate({ key: 'ela_enabled', value: elaEnabled ? 'false' : 'true' })}
              disabled={updateSettingMutation.isPending}
              className={`relative w-14 h-7 rounded-full transition-colors cursor-pointer ${
                elaEnabled ? 'bg-green-500' : 'bg-white/20'
              }`}
            >
              <div className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform ${
                elaEnabled ? 'translate-x-7' : 'translate-x-0'
              }`} />
            </button>
          </div>
          <div className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/10">
            <div className="flex items-center gap-4 flex-1">
              <div className="w-12 h-12 rounded-lg bg-orange-500/20 border border-orange-500/30 flex items-center justify-center">
                <FontAwesomeIcon icon={faImage} className="w-6 h-6 text-orange-400" />
              </div>
              <div className="flex-1">
                <div className="text-white font-medium text-base mb-1">EXIF Analyse</div>
                <div className="text-white/60 text-sm">
                  Detecteert foto editing software (Photoshop, GIMP). Standaard uit (vaak ruis bij scans).
                </div>
              </div>
            </div>
            <button
              onClick={() => updateSettingMutation.mutate({ key: 'exif_enabled', value: exifEnabled ? 'false' : 'true' })}
              disabled={updateSettingMutation.isPending}
              className={`relative w-14 h-7 rounded-full transition-colors cursor-pointer ${
                exifEnabled ? 'bg-green-500' : 'bg-white/20'
              }`}
            >
              <div className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform ${
                exifEnabled ? 'translate-x-7' : 'translate-x-0'
              }`} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// LLM Tab - Switch between Ollama and vLLM
function LLMTab() {
  const queryClient = useQueryClient();
  const [switching, setSwitching] = useState(false);

  // Fetch LLM settings
  const { data: settings, isLoading: settingsLoading, refetch: refetchSettings } = useQuery({
    queryKey: ['llm-settings'],
    queryFn: getLLMSettings,
  });

  // Fetch LLM health status - check both providers on settings page
  const { data: health, isLoading: healthLoading, refetch: refetchHealth, isRefetching: isRefetchingHealth } = useQuery({
    queryKey: ['llm-health', 'all'],
    queryFn: () => getLLMHealth(true), // Check both providers on settings page
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const switchMutation = useMutation({
    mutationFn: switchLLMProvider,
    onMutate: () => setSwitching(true),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-settings'] });
      queryClient.invalidateQueries({ queryKey: ['llm-health'] });
    },
    onSettled: () => setSwitching(false),
  });

  const handleSwitch = (provider: 'ollama' | 'vllm') => {
    if (settings?.active_provider !== provider) {
      switchMutation.mutate(provider);
    }
  };

  const isLoading = settingsLoading || healthLoading;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
            <FontAwesomeIcon icon={faBrain} className="text-purple-400 w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">LLM Provider</h2>
            <p className="text-white/60 text-sm">Kies tussen Ollama en vLLM voor AI-gebaseerde classificatie</p>
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <FontAwesomeIcon icon={faSpinner} className="w-6 h-6 text-purple-400 animate-spin" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Ollama Card */}
            <div
              onClick={() => handleSwitch('ollama')}
              className={`relative p-5 rounded-xl border-2 transition-all cursor-pointer ${
                settings?.active_provider === 'ollama'
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-white/5 border-white/10 hover:border-white/30 hover:bg-white/10'
              }`}
            >
              {switching && settings?.active_provider !== 'ollama' && (
                <div className="absolute inset-0 bg-black/50 rounded-xl flex items-center justify-center">
                  <FontAwesomeIcon icon={faSpinner} className="w-6 h-6 text-white animate-spin" />
                </div>
              )}
              
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-orange-500/20 border border-orange-500/30 flex items-center justify-center">
                    <span className="text-orange-400 font-bold text-lg">🦙</span>
                  </div>
                  <div>
                    <h3 className="text-white font-semibold">Ollama</h3>
                    <p className="text-white/50 text-xs">Lokale LLM server</p>
                  </div>
                </div>
                {settings?.active_provider === 'ollama' && (
                  <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full font-medium">
                    Actief
                  </span>
                )}
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-white/50">URL:</span>
                  <span className="text-white/80 font-mono text-xs">{settings?.providers.ollama.base_url}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Model:</span>
                  <span className="text-white/80">{settings?.providers.ollama.model}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Max tokens:</span>
                  <span className="text-white/80">{settings?.providers.ollama.max_tokens?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/50">Status:</span>
                  {health?.providers.ollama.reachable ? (
                    <span className="flex items-center gap-1 text-green-400">
                      <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                      Online
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-red-400">
                      <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                      Offline
                    </span>
                  )}
                </div>
                {health?.providers.ollama.reachable && (
                  <div className="flex justify-between items-center">
                    <span className="text-white/50">Model beschikbaar:</span>
                    {health.providers.ollama.model_available ? (
                      <FontAwesomeIcon icon={faCheck} className="text-green-400 w-4 h-4" />
                    ) : (
                      <FontAwesomeIcon icon={faTimes} className="text-red-400 w-4 h-4" />
                    )}
                  </div>
                )}
              </div>

              <div className="mt-4 pt-3 border-t border-white/10">
                <p className="text-white/40 text-xs">
                  Sequentiële verwerking • Ideaal voor ontwikkeling
                </p>
              </div>
            </div>

            {/* vLLM Card */}
            <div
              onClick={() => handleSwitch('vllm')}
              className={`relative p-5 rounded-xl border-2 transition-all cursor-pointer ${
                settings?.active_provider === 'vllm'
                  ? 'bg-green-500/10 border-green-500/50'
                  : 'bg-white/5 border-white/10 hover:border-white/30 hover:bg-white/10'
              }`}
            >
              {switching && settings?.active_provider !== 'vllm' && (
                <div className="absolute inset-0 bg-black/50 rounded-xl flex items-center justify-center">
                  <FontAwesomeIcon icon={faSpinner} className="w-6 h-6 text-white animate-spin" />
                </div>
              )}

              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
                    <FontAwesomeIcon icon={faRocket} className="text-blue-400 w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-white font-semibold">vLLM</h3>
                    <p className="text-white/50 text-xs">High-performance inference</p>
                  </div>
                </div>
                {settings?.active_provider === 'vllm' && (
                  <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full font-medium">
                    Actief
                  </span>
                )}
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-white/50">URL:</span>
                  <span className="text-white/80 font-mono text-xs">{settings?.providers.vllm.base_url}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Model:</span>
                  <span className="text-white/80">{settings?.providers.vllm.model}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Max tokens:</span>
                  <span className="text-white/80">{settings?.providers.vllm.max_tokens?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-white/50">Status:</span>
                  {health?.providers.vllm.reachable ? (
                    <span className="flex items-center gap-1 text-green-400">
                      <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                      Online
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-red-400">
                      <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                      Offline
                    </span>
                  )}
                </div>
                {health?.providers.vllm.reachable && (
                  <div className="flex justify-between items-center">
                    <span className="text-white/50">Model beschikbaar:</span>
                    {health.providers.vllm.model_available ? (
                      <FontAwesomeIcon icon={faCheck} className="text-green-400 w-4 h-4" />
                    ) : (
                      <FontAwesomeIcon icon={faTimes} className="text-red-400 w-4 h-4" />
                    )}
                  </div>
                )}
              </div>

              <div className="mt-4 pt-3 border-t border-white/10">
                <p className="text-white/40 text-xs">
                  Parallelle verwerking • OpenAI-compatibel • Productie
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Refresh button */}
        <div className="mt-4 flex justify-end">
          <button
            type="button"
            onClick={async () => {
              await Promise.all([refetchHealth(), refetchSettings()]);
            }}
            disabled={isRefetchingHealth || settingsLoading}
            className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg transition-all ${
              isRefetchingHealth || settingsLoading
                ? 'text-white/40 cursor-not-allowed'
                : 'text-white/60 hover:text-white hover:bg-white/10 cursor-pointer'
            }`}
          >
            <FontAwesomeIcon 
              icon={faRefresh} 
              className={`w-3 h-3 ${isRefetchingHealth || settingsLoading ? 'animate-spin' : ''}`} 
            />
            Status vernieuwen
          </button>
        </div>
      </div>

      {/* Info Section */}
      <div className="glass-card p-6">
        <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faLightbulb} className="text-yellow-400 w-4 h-4" />
          Configuratie
        </h3>
        <div className="text-white/60 text-sm space-y-3">
          <p>
            De LLM provider kan worden geconfigureerd via de <code className="bg-white/10 px-1.5 py-0.5 rounded text-white/80">.env</code> file in de backend folder:
          </p>
          <div className="bg-black/30 rounded-lg p-4 font-mono text-xs overflow-x-auto">
            <div className="text-green-400"># Actieve provider</div>
            <div>LLM_PROVIDER=ollama</div>
            <div className="mt-2 text-green-400"># Ollama instellingen</div>
            <div>OLLAMA_BASE_URL=http://localhost:11434</div>
            <div>OLLAMA_MODEL=llama3.2:3b</div>
            <div className="mt-2 text-green-400"># vLLM instellingen</div>
            <div>VLLM_BASE_URL=http://localhost:8000</div>
            <div>VLLM_MODEL=llama3.2:3b</div>
          </div>
          <p className="text-white/40 text-xs mt-2">
            Wijzigingen in de .env file vereisen een herstart van de backend. Runtime wisselen via deze pagina werkt direct.
          </p>
        </div>
      </div>
    </div>
  );
}

// Skip Markers Tab
function SkipMarkersTab() {
  const queryClient = useQueryClient();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newPattern, setNewPattern] = useState('');
  const [newDescription, setNewDescription] = useState('');
  const [newIsRegex, setNewIsRegex] = useState(false);

  const { data: markers, isLoading } = useQuery({
    queryKey: ['skip-markers'],
    queryFn: () => listSkipMarkers(),
  });

  const createMutation = useMutation({
    mutationFn: (data: { pattern: string; description?: string; is_regex?: boolean; is_active?: boolean }) => createSkipMarker(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['skip-markers'] });
      setShowCreateForm(false);
      setNewPattern('');
      setNewDescription('');
      setNewIsRegex(false);
    },
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, is_active }: { id: number; is_active: boolean }) =>
      updateSkipMarker(id, { is_active }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['skip-markers'] }),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteSkipMarker(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['skip-markers'] }),
  });

  const handleCreate = () => {
    if (!newPattern.trim()) return;
    createMutation.mutate({
      pattern: newPattern.trim(),
      description: newDescription.trim() || undefined,
      is_regex: newIsRegex,
      is_active: true,
    });
  };

  return (
    <div className="space-y-6">
      {/* Info Card */}
      <div className="glass-card p-4 sm:p-6 border border-blue-500/20 bg-blue-500/5">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-500/20 border border-blue-500/30 flex items-center justify-center shrink-0">
            <FontAwesomeIcon icon={faFilter} className="text-blue-400 w-5 h-5" />
          </div>
          <div>
            <h3 className="text-white font-semibold mb-1">Wat zijn Skip Markers?</h3>
            <p className="text-white/60 text-sm leading-relaxed">
              Skip markers zijn tekstpatronen die aangeven waar de documentverwerking moet stoppen.
              Als een skip marker wordt gevonden, wordt alle tekst daarna genegeerd. Dit is handig voor:
            </p>
            <ul className="text-white/60 text-sm mt-2 space-y-1 list-disc list-inside">
              <li>Algemene voorwaarden onderaan documenten</li>
              <li>Repeterende headers/footers</li>
              <li>Automatisch gegenereerde content</li>
            </ul>
            <p className="text-white/50 text-xs mt-3 flex items-center gap-1.5">
              <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-400" />
              Dit bespaart LLM tokens en vermindert verwerkingstijd voor grote documenten.
            </p>
          </div>
        </div>
      </div>

      {/* Create Form */}
      {!showCreateForm ? (
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 text-blue-300 text-sm font-medium transition-all"
        >
          <FontAwesomeIcon icon={faPlus} className="w-3 h-3" />
          Nieuwe Skip Marker
        </button>
      ) : (
        <div className="glass-card p-4 sm:p-6 space-y-4">
          <h3 className="text-white font-semibold">Nieuwe Skip Marker</h3>
          
          <div>
            <label className="block text-white/60 text-xs mb-1">Patroon *</label>
            <input
              type="text"
              value={newPattern}
              onChange={(e) => setNewPattern(e.target.value)}
              placeholder="bijv. Algemene Voorwaarden"
              className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-blue-500/50 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-white/60 text-xs mb-1">Beschrijving</label>
            <input
              type="text"
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              placeholder="Optionele uitleg"
              className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-sm focus:border-blue-500/50 focus:outline-none"
            />
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={newIsRegex}
              onChange={(e) => setNewIsRegex(e.target.checked)}
              className="w-4 h-4 rounded border-white/20 bg-white/5 text-blue-500 focus:ring-blue-500/50"
            />
            <span className="text-white/70 text-sm">Regex patroon</span>
            {newIsRegex && (
              <span className="text-amber-400/80 text-xs">(bijv. Pagina \d+ van \d+)</span>
            )}
          </label>

          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              disabled={!newPattern.trim() || createMutation.isPending}
              className="px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {createMutation.isPending ? 'Opslaan...' : 'Opslaan'}
            </button>
            <button
              onClick={() => {
                setShowCreateForm(false);
                setNewPattern('');
                setNewDescription('');
                setNewIsRegex(false);
              }}
              className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white/70 text-sm"
            >
              Annuleren
            </button>
          </div>
        </div>
      )}

      {/* Markers List */}
      <div className="glass-card p-4 sm:p-6">
        <h3 className="text-white font-semibold mb-4">
          Skip Markers ({markers?.length || 0})
        </h3>

        {isLoading ? (
          <div className="text-white/60 text-sm">Laden...</div>
        ) : !markers?.length ? (
          <div className="text-white/60 text-sm">Geen skip markers geconfigureerd.</div>
        ) : (
          <div className="space-y-2">
            {markers.map((marker) => (
              <div
                key={marker.id}
                className={`flex items-center justify-between gap-3 p-3 rounded-lg border transition-all ${
                  marker.is_active
                    ? 'bg-white/5 border-white/10'
                    : 'bg-white/2 border-white/5 opacity-60'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <code className="text-white text-sm font-mono truncate">
                      {marker.pattern}
                    </code>
                    {marker.is_regex && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-300 border border-purple-500/30">
                        regex
                      </span>
                    )}
                  </div>
                  {marker.description && (
                    <div className="text-white/50 text-xs mt-0.5 truncate">
                      {marker.description}
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => toggleMutation.mutate({ id: marker.id, is_active: !marker.is_active })}
                    className={`p-2 rounded-lg transition-all ${
                      marker.is_active
                        ? 'text-emerald-400 hover:bg-emerald-500/20'
                        : 'text-white/40 hover:bg-white/10'
                    }`}
                    title={marker.is_active ? 'Deactiveren' : 'Activeren'}
                  >
                    <FontAwesomeIcon
                      icon={marker.is_active ? faToggleOn : faToggleOff}
                      className="w-5 h-5"
                    />
                  </button>
                  <button
                    onClick={() => {
                      if (confirm('Weet je zeker dat je deze skip marker wilt verwijderen?')) {
                        deleteMutation.mutate(marker.id);
                      }
                    }}
                    className="p-2 rounded-lg text-red-400/70 hover:text-red-400 hover:bg-red-500/20 transition-all"
                    title="Verwijderen"
                  >
                    <FontAwesomeIcon icon={faTrash} className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// API Keys Tab
function ApiKeysTab() {
  const queryClient = useQueryClient();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [createdKey, setCreatedKey] = useState<ApiKeyCreated | null>(null);
  const [showSecret, setShowSecret] = useState(false);
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<{ key: ApiKey } | null>(null);

  const { data: apiKeys, isLoading } = useQuery({
    queryKey: ['api-keys'],
    queryFn: listApiKeys,
  });

  const createMutation = useMutation({
    mutationFn: createApiKey,
    onSuccess: (data) => {
      setCreatedKey(data);
      setShowCreateForm(false);
      setNewKeyName('');
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteApiKey,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, is_active }: { id: number; is_active: boolean }) =>
      updateApiKey(id, { is_active }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });

  const copyToClipboard = (text: string, field: string) => {
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Created Key Modal */}
      {createdKey && (
        <div className="glass-card p-4 sm:p-6 border-2 border-green-500/30 bg-green-500/5">
          <div className="flex items-center gap-2 mb-4">
            <FontAwesomeIcon icon={faCheck} className="text-green-400 w-4 h-4" />
            <h3 className="text-white font-semibold text-sm sm:text-base">API Key Aangemaakt!</h3>
          </div>
          <div className="bg-amber-500/20 border border-amber-500/40 rounded-lg p-2 sm:p-3 mb-4">
            <p className="text-amber-200 text-xs sm:text-sm flex items-start sm:items-center gap-2 font-medium">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 sm:w-4 sm:h-4 shrink-0 mt-0.5 sm:mt-0" />
              <span>Kopieer de client secret nu! Deze wordt niet meer getoond.</span>
            </p>
          </div>
          <div className="space-y-3">
            <div>
              <label className="text-white/60 text-xs block mb-1">Client ID</label>
              <div className="flex items-center gap-1 sm:gap-2">
                <code className="flex-1 bg-black/30 px-2 sm:px-3 py-2 rounded text-white font-mono text-[10px] sm:text-sm overflow-x-auto">
                  {createdKey.client_id}
                </code>
                <button
                  onClick={() => copyToClipboard(createdKey.client_id, 'client_id')}
                  className="p-1.5 sm:p-2 bg-white/10 rounded hover:bg-white/20 transition-colors shrink-0"
                >
                  <FontAwesomeIcon 
                    icon={copiedField === 'client_id' ? faCheck : faCopy} 
                    className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${copiedField === 'client_id' ? 'text-green-400' : 'text-white/60'}`} 
                  />
                </button>
              </div>
            </div>
            <div>
              <label className="text-white/60 text-xs block mb-1">Client Secret</label>
              <div className="flex items-center gap-1 sm:gap-2">
                <code className="flex-1 bg-black/30 px-2 sm:px-3 py-2 rounded text-white font-mono text-[10px] sm:text-sm overflow-x-auto">
                  {showSecret ? createdKey.client_secret : '•'.repeat(20)}
                </code>
                <button
                  onClick={() => setShowSecret(!showSecret)}
                  className="p-1.5 sm:p-2 bg-white/10 rounded hover:bg-white/20 transition-colors shrink-0"
                >
                  <FontAwesomeIcon icon={showSecret ? faEyeSlash : faEye} className="text-white/60 w-3.5 h-3.5 sm:w-4 sm:h-4" />
                </button>
                <button
                  onClick={() => copyToClipboard(createdKey.client_secret, 'client_secret')}
                  className="p-1.5 sm:p-2 bg-white/10 rounded hover:bg-white/20 transition-colors shrink-0"
                >
                  <FontAwesomeIcon 
                    icon={copiedField === 'client_secret' ? faCheck : faCopy} 
                    className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${copiedField === 'client_secret' ? 'text-green-400' : 'text-white/60'}`} 
                  />
                </button>
              </div>
            </div>
          </div>
          <button
            onClick={() => {
              setCreatedKey(null);
              setShowSecret(false);
            }}
            className="mt-4 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors text-sm"
          >
            Sluiten
          </button>
        </div>
      )}

      {/* Header */}
      <div className="glass-card p-4 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <h2 className="text-white text-lg sm:text-xl font-semibold flex items-center gap-2">
            <div className="w-9 h-9 rounded-lg bg-amber-500/20 border border-amber-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faKey} className="text-amber-400 w-4 h-4" />
            </div>
            API Keys
          </h2>
          <button
            onClick={() => setShowCreateForm(true)}
            className="flex items-center gap-2 px-3 sm:px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm sm:text-base self-start sm:self-auto"
          >
            <FontAwesomeIcon icon={faPlus} className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            Nieuwe Key
          </button>
        </div>

        {/* Create Form */}
        {showCreateForm && (
          <div className="bg-white/5 rounded-lg p-3 sm:p-4 mb-4 border border-white/10">
            <h4 className="text-white font-medium mb-3 text-sm sm:text-base">Nieuwe API Key</h4>
            <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
              <input
                type="text"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                placeholder="Naam (bijv. Production)"
                className="flex-1 px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 text-sm"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => createMutation.mutate({ name: newKeyName })}
                  disabled={!newKeyName.trim() || createMutation.isPending}
                  className="flex-1 sm:flex-none px-3 sm:px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
                >
                  Aanmaken
                </button>
                <button
                  onClick={() => {
                    setShowCreateForm(false);
                    setNewKeyName('');
                  }}
                  className="flex-1 sm:flex-none px-3 sm:px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors text-sm"
                >
                  Annuleer
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Keys List */}
        {isLoading ? (
          <div className="text-center py-8">
            <FontAwesomeIcon icon={faKey} className="text-white/40 text-3xl animate-pulse" />
          </div>
        ) : apiKeys?.length === 0 ? (
          <div className="text-center py-8 bg-white/5 rounded-lg border border-dashed border-white/20">
            <FontAwesomeIcon icon={faKey} className="text-white/30 text-3xl mb-2" />
            <p className="text-white/50">Geen API keys</p>
            <p className="text-white/40 text-sm">Maak een nieuwe key aan om te beginnen</p>
          </div>
        ) : (
          <div className="space-y-2">
            {apiKeys?.map((key) => (
              <div
                key={key.id}
                className={`flex flex-col sm:flex-row sm:items-center justify-between gap-2 sm:gap-4 p-3 sm:p-4 rounded-lg border ${
                  key.is_active
                    ? 'bg-white/5 border-white/10'
                    : 'bg-red-500/5 border-red-500/20 opacity-60'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`w-2 h-2 rounded-full shrink-0 ${key.is_active ? 'bg-green-400' : 'bg-red-400'}`} />
                  <div className="min-w-0">
                    <div className="text-white font-medium text-sm sm:text-base truncate">{key.name}</div>
                    <code className="text-white/50 text-[10px] sm:text-xs font-mono block truncate">{key.client_id}</code>
                  </div>
                </div>
                <div className="flex items-center gap-2 self-end sm:self-auto shrink-0">
                  <button
                    onClick={() => toggleMutation.mutate({ id: key.id, is_active: !key.is_active })}
                    className={`px-2 sm:px-3 py-1 rounded text-[10px] sm:text-xs transition-colors ${
                      key.is_active
                        ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30'
                        : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                    }`}
                  >
                    {key.is_active ? 'Deactiveer' : 'Activeer'}
                  </button>
                  <button
                    onClick={() => setConfirmDelete({ key })}
                    className="p-1.5 sm:p-2 text-red-400 hover:bg-red-500/20 rounded transition-colors"
                  >
                    <FontAwesomeIcon icon={faTrash} className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Usage Info */}
      <div className="glass-card p-4 sm:p-6">
        <h3 className="text-white font-semibold mb-4 text-sm sm:text-base">API Gebruik</h3>
        <div className="bg-black/30 rounded-lg p-3 sm:p-4 font-mono text-xs sm:text-sm overflow-x-auto">
          <div className="text-white/60 mb-2"># Authenticatie met API key</div>
          <div className="text-cyan-400 whitespace-nowrap">curl -X GET "http://localhost:8000/api/documents" \</div>
          <div className="text-cyan-400 pl-2 sm:pl-4 whitespace-nowrap">-H "X-Client-ID: your_client_id" \</div>
          <div className="text-cyan-400 pl-2 sm:pl-4 whitespace-nowrap">-H "X-Client-Secret: your_client_secret"</div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-card max-w-md w-full p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-red-500/20 rounded-full flex items-center justify-center">
                  <FontAwesomeIcon icon={faTrash} className="text-red-400 w-5 h-5" />
                </div>
                <h3 className="text-white text-lg font-semibold">API Key verwijderen</h3>
              </div>
              <button
                onClick={() => setConfirmDelete(null)}
                className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
              </button>
            </div>

            <div className="mb-6">
              <p className="text-white/80 mb-2">
                Weet je zeker dat je deze API key wilt verwijderen?
              </p>
              <div className="bg-white/5 rounded-lg p-3 border border-white/10">
                <p className="text-white font-medium truncate">
                  {confirmDelete.key.name}
                </p>
                <p className="text-white/60 text-sm font-mono">
                  {confirmDelete.key.client_id}
                </p>
              </div>
              <p className="text-white/60 text-sm mt-3">
                Deze actie kan niet ongedaan worden gemaakt.
              </p>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={() => setConfirmDelete(null)}
                className="flex-1 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
                disabled={deleteMutation.isPending}
              >
                Annuleren
              </button>
              <button
                onClick={() => {
                  deleteMutation.mutate(confirmDelete.key.id);
                  setConfirmDelete(null);
                }}
                disabled={deleteMutation.isPending}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {deleteMutation.isPending ? (
                  <>
                    <FontAwesomeIcon icon={faSpinner} className="w-4 h-4 animate-spin mr-2" />
                    Verwijderen...
                  </>
                ) : (
                  'Verwijderen'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// MCP Tab
function McpTab() {
  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <h2 className="text-white text-xl font-semibold flex items-center gap-2 mb-4">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
            <FontAwesomeIcon icon={faPlug} className="text-cyan-400 w-4 h-4" />
          </div>
          MCP Integraties
        </h2>
        
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 mb-6">
          <h4 className="text-white font-medium mb-2">Wat is MCP?</h4>
          <p className="text-white/70 text-sm">
            Model Context Protocol (MCP) is een open standaard voor het verbinden van AI-modellen
            met externe tools en data bronnen. MProof kan als MCP server fungeren, zodat AI-assistenten
            direct toegang hebben tot document analyse functionaliteit.
          </p>
        </div>

        {/* MCP Server Config */}
        <div className="space-y-4">
          <h3 className="text-white font-semibold">MCP Server Configuratie</h3>
          
          <div className="bg-black/30 rounded-lg p-4 font-mono text-sm">
            <div className="text-white/60 mb-2">// Voeg toe aan ~/.cursor/mcp.json</div>
            <pre className="text-green-400 text-xs overflow-x-auto">{`{
  "mcpServers": {
    "mproof": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "X-Client-ID": "<your_client_id>",
        "X-Client-Secret": "<your_client_secret>"
      }
    }
  }
}`}</pre>
            <div className="text-green-200/80 text-xs mt-3 p-2 bg-green-500/10 rounded border border-green-500/20">
              <p className="font-medium mb-2 flex items-center gap-1.5">
                <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-400" />
                HTTP-gebaseerde MCP Server
              </p>
              <p className="mb-2">De MCP server draait als onderdeel van de backend API - geen aparte processen nodig!</p>
              <p className="mb-2 text-green-100/70">Gebruikt automatisch SSE streaming (geen polling).</p>
              <p className="mt-2 text-green-100/70 flex items-start gap-1.5">
                <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-400 mt-0.5 shrink-0" />
                <span>Vervang <code className="bg-black/30 px-1 rounded">&lt;your_client_id&gt;</code> en <code className="bg-black/30 px-1 rounded">&lt;your_client_secret&gt;</code> met de waarden van je API key hierboven.</span>
              </p>
              <p className="mt-2 text-green-100/70 flex items-start gap-1.5">
                <FontAwesomeIcon icon={faGlobe} className="w-3 h-3 text-blue-400 mt-0.5 shrink-0" />
                <span>Voor productie: verander <code className="bg-black/30 px-1 rounded">localhost:8000</code> naar je productie URL.</span>
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
            <div className="bg-white/5 rounded-lg p-4 border border-white/10">
              <h4 className="text-white font-medium mb-2 flex items-center gap-2">
                <div className="w-6 h-6 rounded-lg bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
                  <FontAwesomeIcon icon={faFileAlt} className="text-purple-400 w-3 h-3" />
                </div>
                Beschikbare Tools (13)
              </h4>
              <ul className="space-y-1.5 text-sm text-white/70 max-h-96 overflow-y-auto pr-2 scrollbar-thin">
                <li className="flex items-center gap-2">
                  <span className="text-green-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">list_documents</code>
                  - Lijst documenten met filters
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">get_document</code>
                  - Haal document details op
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">analyze_document</code>
                  - Trigger document analyse
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">list_subjects</code>
                  - Zoek subjects (personen, bedrijven)
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">get_document_text</code>
                  - Haal geëxtraheerde tekst op
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">get_document_metadata</code>
                  - Haal metadata op
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-red-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">get_fraud_analysis</code>
                  - Volledige fraude analyse
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">search_documents</code>
                  - Zoek op tekst, type of risicoscore
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-blue-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">train_classifier</code>
                  - Train Naive Bayes model
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-blue-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">train_bert_classifier</code>
                  - Train BERT embeddings model
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-blue-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">get_classifier_status</code>
                  - Naive Bayes status
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-blue-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">get_bert_classifier_status</code>
                  - BERT model status
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-red-400">•</span>
                  <code className="text-xs bg-white/10 px-1.5 py-0.5 rounded">list_high_risk_documents</code>
                  - Lijst hoge risico documenten
                </li>
              </ul>
            </div>

          </div>
        </div>

        {/* Example Prompts */}
        <div className="mt-6">
          <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faCommentDots} className="text-purple-400 w-4 h-4" />
            </div>
            Voorbeeld Prompts
          </h3>
          <p className="text-white/60 text-sm mb-4">
            Kopieer deze prompts naar je AI-assistent (bijv. Claude, Cursor) om MProof te gebruiken:
          </p>
          
          <div className="space-y-3">
            {/* 1. list_documents */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faFileAlt} className="w-3 h-3" />
                  Lijst documenten
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Toon alle documenten voor subject ID 5 met status 'done', maximaal 20 resultaten."
              </code>
            </div>

            {/* 2. get_document */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faFileAlt} className="w-3 h-3" />
                  Document details ophalen
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Haal alle details op van document ID 42, inclusief classificatie, metadata en risicoscores."
              </code>
            </div>

            {/* 3. analyze_document */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faRefresh} className="w-3 h-3" />
                  Document opnieuw analyseren
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Trigger een nieuwe analyse voor document ID 15 om de classificatie en metadata te updaten."
              </code>
            </div>

            {/* 4. list_subjects */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faDatabase} className="w-3 h-3" />
                  Subjects zoeken
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Zoek alle subjects met context 'company' die 'XYZ' in de naam hebben."
              </code>
            </div>

            {/* 5. get_document_text */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faFileAlt} className="w-3 h-3" />
                  Document tekst ophalen
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Haal de geëxtraheerde tekst op van document ID 42 en zoek naar het woord 'IBAN'."
              </code>
            </div>

            {/* 6. get_document_metadata */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faChartBar} className="w-3 h-3" />
                  Metadata ophalen
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Haal de geëxtraheerde metadata op van document ID 42 en toon alle gevonden velden."
              </code>
            </div>

            {/* 7. get_fraud_analysis */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                  Fraude analyse
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Voer een volledige fraude analyse uit op document ID 42. Check PDF metadata, image forensics en tekst anomalieën."
              </code>
            </div>

            {/* 8. search_documents */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faSearch} className="w-3 h-3" />
                  Documenten zoeken
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Zoek alle documenten van type 'bankafschrift' met risicoscore tussen 50-75% voor subject ID 3."
              </code>
            </div>

            {/* 9. train_classifier */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faGraduationCap} className="w-3 h-3" />
                  Naive Bayes trainen
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Train het Naive Bayes classificatie model voor 'backoffice' met de nieuwste training data."
              </code>
            </div>

            {/* 10. train_bert_classifier */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faBrain} className="w-3 h-3" />
                  BERT trainen
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Train het BERT embeddings model voor 'backoffice' met threshold 0.75 voor betere precisie."
              </code>
            </div>

            {/* 11. get_classifier_status */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faRobot} className="w-3 h-3" />
                  Naive Bayes status
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Bekijk de status van het Naive Bayes model voor 'backoffice': hoeveel document types zijn getraind?"
              </code>
            </div>

            {/* 12. get_bert_classifier_status */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faBrain} className="w-3 h-3" />
                  BERT status
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Bekijk de status van het BERT model: wanneer is het laatste getraind en hoeveel document types zijn er?"
              </code>
            </div>

            {/* 13. list_high_risk_documents */}
            <div className="bg-black/30 rounded-lg p-4 border border-white/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-300 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                  Hoge risico documenten
                </span>
              </div>
              <code className="text-sm text-green-400 block">
                "Toon alle documenten met CRITICAL risk level en risicoscore &gt; 75% voor subject ID 5."
              </code>
            </div>
          </div>
        </div>

        {/* Coming Soon */}
        <div className="mt-6 p-4 bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20 rounded-lg">
          <h4 className="text-white font-medium mb-2 flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-purple-500/20 border border-purple-500/30 flex items-center justify-center">
              <FontAwesomeIcon icon={faRocket} className="w-3 h-3 text-purple-400" />
            </div>
            Binnenkort
          </h4>
          <ul className="text-white/70 text-sm space-y-1">
            <li>• OAuth2 authenticatie voor MCP verbindingen</li>
            <li>• Real-time document processing events via MCP</li>
            <li>• Custom tool configuratie per API key</li>
            <li>• MCP client voor externe services (bijv. Notion, Google Drive)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
