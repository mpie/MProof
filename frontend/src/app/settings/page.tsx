'use client';

import { useState, useMemo, useEffect, useRef, Suspense } from 'react';
import { createPortal } from 'react-dom';
import { useQuery, useQueries, useMutation, useQueryClient } from '@tanstack/react-query';
import { useRouter, useSearchParams } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faCog, faKey, faRobot, faPlug, faPlus, faTrash, faCopy, faCheck,
  faEye, faEyeSlash, faRefresh, faExclamationTriangle, faGraduationCap,
  faFileAlt, faCode, faDatabase, faFilter, faToggleOn, faToggleOff, faTimes, faSpinner,
  faEdit, faSave,
  faFolder, faChevronDown, faBan, faHandPointer, faSearch, faBullseye, faBolt, faBrain,
  faLightbulb, faGlobe, faCommentDots, faChartBar, faRocket, faShieldAlt, faImage, faInfoCircle,
  faChevronRight, faCheckCircle, faSort, faList
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
  updateLLMSettings,
  LLMSettingsResponse,
  LLMHealthResponse,
  getAppSettings,
  updateAppSetting,
  listDocumentTypes,
  generateDocumentTypePrefill,
} from '@/lib/api';
import { useModel } from '@/context/ModelContext';
import { useAuth } from '@/context/AuthContext';

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
        <h3 className="text-slate-800 font-semibold flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-100 border border-blue-200 flex items-center justify-center">
            <FontAwesomeIcon icon={faDatabase} className="text-blue-400 w-4 h-4" />
          </div>
          {modelName ? `Document Types (${labels.length})` : `Getrainde Modellen (${labels.length})`}
          {modelName && (
            <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full font-normal">
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
            className="w-full sm:w-48 px-3 py-1.5 pl-8 rounded-lg bg-slate-50 border border-slate-200 text-slate-800 text-sm focus:border-blue-500/50 focus:outline-none"
          />
          <FontAwesomeIcon icon={faSearch} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 w-3 h-3" />
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
                className="bg-slate-50 rounded-lg p-4 border border-slate-200 hover:border-slate-300 hover:bg-slate-100 transition-all cursor-pointer group relative min-h-[120px]"
              >
                <div className="flex items-start justify-between gap-2 mb-2 pr-16">
                  <h4 className="text-slate-800 font-semibold text-sm flex-1 break-words" title={label}>
                    {label}
                  </h4>
                </div>
                
                {/* Create Document Type button if it doesn't exist */}
                {!documentTypeExists && (
                  <button
                    onClick={(e) => handleCreateDocumentType(label, e)}
                    className="absolute top-2.5 right-2.5 z-10 flex items-center gap-1 px-1.5 py-0.5 bg-green-50 hover:bg-green-100 border border-green-200 text-green-700 text-[10px] rounded transition-colors cursor-pointer opacity-70 hover:opacity-100"
                    title="Maak document type aan"
                  >
                    <FontAwesomeIcon icon={faPlus} className="w-2.5 h-2.5" />
                    <span>Maak aan</span>
                  </button>
                )}
                
                {/* Show model name when viewing all (Standaard) */}
                {!selectedModel && belongsToModel && (
                  <div className="mb-2">
                    <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded font-medium inline-flex items-center gap-1">
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
                        className="text-xs bg-purple-50 text-slate-700 px-1.5 py-0.5 rounded font-mono truncate max-w-[80px]"
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
                        className="text-xs text-purple-600 hover:text-purple-800 px-1.5 py-0.5 hover:bg-purple-100 rounded transition-colors cursor-pointer font-medium underline"
                        title={`Bekijk alle ${topTokens.length} woorden`}
                      >
                        +{topTokens.length - 3} meer
                      </button>
                    )}
                  </div>
                )}

                {/* Click hint with document count */}
                <div className="flex items-center justify-between gap-2 text-xs text-slate-500 group-hover:text-slate-600 transition-colors">
                  <div className="flex items-center gap-1.5">
                    <FontAwesomeIcon icon={faHandPointer} className="w-3 h-3" />
                    <span>Bekijk details</span>
                  </div>
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded font-medium">
                    {docCount}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {filteredLabels.length === 0 && searchQuery && (
        <div className="text-center py-4 text-slate-400 text-sm">
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
          <div className="bg-white rounded-xl border border-slate-200 max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl">
            <div className="p-5 border-b border-slate-200 flex items-center justify-between">
              <div>
                <h3 className="text-slate-800 font-bold text-lg">Herkende woorden: {selectedLabel}</h3>
                <p className="text-slate-400 text-sm mt-1">
                      {selectedTokens.length > 0 ? (
                    <>
                      {selectedTokens.length} unieke tokens • Gesorteerd op frequentie
                      {!selectedModel && selectedLabelModel && (
                        <span className="ml-2 text-purple-700">• Model: {selectedLabelModel}</span>
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
                className="text-slate-500 hover:text-slate-800 px-3 py-1 hover:bg-slate-100 rounded transition-colors"
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
                      className="inline-flex items-center gap-2 text-sm bg-purple-50 text-slate-800 px-3 py-1.5 rounded-lg font-mono border border-purple-200 hover:bg-purple-100 transition-colors"
                      title={`${t.token} komt ${t.count}x voor`}
                    >
                      <span className="truncate max-w-[200px]">{t.token}</span>
                      <span className="text-slate-500 text-xs font-semibold">×{t.count}</span>
                    </span>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="w-16 h-16 mx-auto mb-3 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-center">
                    <FontAwesomeIcon icon={faFileAlt} className="w-8 h-8 text-slate-400" />
                  </div>
                  <p className="text-slate-600 text-base font-medium">
                    Geen training data beschikbaar
                  </p>
                  <p className="text-slate-400 text-sm mt-2">
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
                    <p className="text-slate-400 text-xs mt-3">
                      Model: {selectedLabelModel}
                    </p>
                  )}
                </div>
              )}
            </div>

            <div className="p-5 border-t border-slate-200 flex justify-between items-center bg-slate-50">
              <p className="text-slate-400 text-sm flex items-center gap-1.5">
                <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-600" />
                Deze woorden helpen het model om "{selectedLabel}" documenten te herkennen
              </p>
              <button
                onClick={() => {
                  setShowTokensModal(false);
                  setSelectedLabel(null);
                }}
                className="px-5 py-2 rounded-lg bg-slate-100 hover:bg-slate-100 text-slate-800 text-sm font-medium transition-colors"
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

function SettingsPageInner() {
  const searchParams = useSearchParams();
  const initialTab = (searchParams.get('tab') as TabType | null) ?? 'model';
  const [activeTab, setActiveTab] = useState<TabType>(initialTab);

  // Sync if URL param changes (e.g. navigating from user menu)
  useEffect(() => {
    const t = searchParams.get('tab') as TabType | null;
    if (t) setActiveTab(t);
  }, [searchParams]);

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
        <h1 className="text-slate-800 text-xl sm:text-2xl lg:text-3xl font-bold flex items-center gap-2 sm:gap-3">
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-blue-100 border border-blue-200 flex items-center justify-center">
            <FontAwesomeIcon icon={faCog} className="text-blue-400 w-5 h-5 sm:w-6 sm:h-6" />
          </div>
          Instellingen
        </h1>
        <p className="text-slate-400 mt-1 text-sm sm:text-base max-w-xl leading-relaxed">
          Train het classificatie-model, beheer API-sleutels voor externe koppelingen en pas de fraude-detectie aan.
        </p>
      </div>

      {/* Tabs - scrollable on mobile */}
      <div className="glass-card p-1 flex rounded-lg sm:rounded-xl overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 sm:py-2.5 rounded-lg text-xs sm:text-sm font-medium transition-all whitespace-nowrap cursor-pointer ${
              activeTab === tab.id
                ? 'bg-gradient-to-r from-[#22d3d3]/15 to-[#FFC1F3]/15 text-slate-800 border border-[#22d3d3]/30 shadow-sm'
                : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
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
  const dropdownTriggerRef = useRef<HTMLDivElement>(null);
  const [dropdownPos, setDropdownPos] = useState({ top: 0, right: 0 });
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
        <FontAwesomeIcon icon={faRobot} className="text-slate-400 text-4xl animate-pulse" />
        <p className="text-slate-500 mt-2">Laden...</p>
      </div>
    );
  }

  const isTraining = trainMutation.isPending || (classifierStatus?.running ?? false);
  const trainedLabels = trainingDetails?.model?.labels ?? [];
  const classDocCounts = trainingDetails?.model?.class_doc_counts ?? {};

  return (
    <div className="space-y-4">
      {/* Hero status banner */}
      <div className={`relative overflow-hidden rounded-2xl border p-5 sm:p-6 ${
        isTraining
          ? 'bg-gradient-to-br from-amber-50 to-amber-100/40 border-amber-200'
          : trainingDetails?.model_exists
            ? 'bg-gradient-to-br from-[#22d3d3]/5 via-white to-[#FFC1F3]/5 border-[#22d3d3]/20'
            : 'bg-gradient-to-br from-slate-50 to-slate-100/50 border-slate-200'
      }`}>
        {trainingDetails?.model_exists && !isTraining && (
          <>
            <div className="absolute -top-16 -right-16 w-56 h-56 rounded-full bg-[#22d3d3]/10 blur-3xl pointer-events-none" />
            <div className="absolute -bottom-10 -left-10 w-40 h-40 rounded-full bg-[#FFC1F3]/10 blur-3xl pointer-events-none" />
          </>
        )}

        <div className="relative flex flex-col sm:flex-row sm:items-start gap-4 sm:gap-5">
          {/* Icon */}
          <div className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 ${
            isTraining
              ? 'bg-amber-100 border border-amber-200'
              : trainingDetails?.model_exists
                ? 'bg-gradient-to-br from-[#22d3d3]/20 to-[#FFC1F3]/20 border border-[#22d3d3]/30'
                : 'bg-slate-100 border border-slate-200'
          }`}>
            <FontAwesomeIcon
              icon={isTraining ? faSpinner : faRobot}
              className={`text-2xl ${isTraining ? 'text-amber-500 animate-spin' : trainingDetails?.model_exists ? 'text-[#22d3d3]' : 'text-slate-400'}`}
            />
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2.5 flex-wrap mb-1">
              <h2 className="text-slate-800 font-bold text-lg tracking-tight">Classificatie Model</h2>
              <span className={`text-[10px] px-2.5 py-1 rounded-full font-bold tracking-widest uppercase ${
                isTraining ? 'bg-amber-100 text-amber-700' :
                trainingDetails?.model_exists ? 'bg-emerald-100 text-emerald-700' :
                'bg-slate-100 text-slate-500'
              }`}>
                {isTraining ? '⏳ Trainen' : trainingDetails?.model_exists ? '● Actief' : '○ Niet getraind'}
              </span>
            </div>
            <p className="text-slate-400 text-sm">
              {trainingDetails?.model_exists
                ? `${trainedLabels.length} type${trainedLabels.length !== 1 ? 's' : ''} herkend · Getraind op ${new Date(trainingDetails.model!.updated_at).toLocaleDateString('nl-NL', { day: '2-digit', month: 'short', year: 'numeric' })}`
                : 'Bevestig documenten in de lijst om het model te trainen'}
            </p>

            {/* Stat strip */}
            {trainingDetails?.model_exists && trainingDetails.model && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4">
                {[
                  { label: 'Trainingsdocs', value: Object.values(classDocCounts).reduce((s: number, n) => s + Number(n), 0).toLocaleString() },
                  { label: 'Documenttypes', value: String(trainedLabels.length) },
                  { label: 'Vocabulaire', value: trainingDetails.model.vocab_size.toLocaleString() },
                  { label: 'Min. zekerheid', value: `${(trainingDetails.model.threshold * 100).toFixed(0)}%` },
                ].map(stat => (
                  <div key={stat.label} className="bg-white/70 backdrop-blur-sm rounded-xl p-3 border border-white/80 shadow-sm">
                    <div className="text-slate-400 text-[9px] uppercase tracking-widest mb-1">{stat.label}</div>
                    <div className="text-slate-800 text-xl font-bold tabular-nums leading-none">{stat.value}</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Train button */}
          <div className="shrink-0 self-start">
            <button
              onClick={() => trainMutation.mutate({ modelName: selectedModel, incremental: false })}
              disabled={isTraining}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                isTraining
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  : 'bg-[#22d3d3] hover:bg-[#1ab8b8] text-white cursor-pointer shadow-[0_2px_14px_rgba(34,211,211,0.35)] hover:shadow-[0_4px_20px_rgba(34,211,211,0.45)]'
              }`}
            >
              <FontAwesomeIcon icon={faRefresh} className={`w-3.5 h-3.5 ${isTraining ? 'animate-spin' : ''}`} />
              {isTraining ? 'Bezig...' : 'Hertrainen'}
            </button>
          </div>
        </div>

        {/* Training progress */}
        {isTraining && (
          <div className="relative mt-4 pt-4 border-t border-amber-200/60">
            <div className="flex items-center gap-2 mb-2">
              <FontAwesomeIcon icon={faSpinner} className="w-3.5 h-3.5 text-amber-500 animate-spin" />
              <span className="text-slate-700 text-sm font-medium">
                {classifierStatus?.current_label ? `Bezig met: ${classifierStatus.current_label}` : 'Documenten verwerken...'}
              </span>
              <span className="text-slate-400 text-xs ml-auto tabular-nums">
                {(() => {
                  const h = Math.floor(trainingElapsed / 3600);
                  const m = Math.floor((trainingElapsed % 3600) / 60);
                  const s = trainingElapsed % 60;
                  return [h > 0 && `${h}u`, m > 0 && `${m}m`, `${s}s`].filter(Boolean).join(' ');
                })()}
              </span>
            </div>
            <div className="h-1.5 bg-amber-100 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] rounded-full animate-pulse" style={{ width: '60%' }} />
            </div>
            {classifierStatus?.last_error && (
              <p className="text-red-600 text-xs mt-2 flex items-center gap-1">
                <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                {classifierStatus.last_error}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Document types — horizontal bar chart */}
      {trainingDetails?.model_exists && trainedLabels.length > 0 && (
        <div className="glass-card p-4 sm:p-5">
          <h3 className="text-slate-700 font-semibold text-sm mb-4 flex items-center gap-2">
            <FontAwesomeIcon icon={faList} className="w-3.5 h-3.5 text-[#22d3d3]" />
            Herkende documenttypes
          </h3>
          <div className="space-y-2.5">
            {(() => {
              const maxCount = Math.max(...trainedLabels.map((l: string) => Number(classDocCounts[l] ?? 0)), 1);
              return [...trainedLabels]
                .sort((a: string, b: string) => Number(classDocCounts[b] ?? 0) - Number(classDocCounts[a] ?? 0))
                .map((label: string) => {
                  const count = Number(classDocCounts[label] ?? 0);
                  const pct = Math.max((count / maxCount) * 100, 3);
                  const color = count >= 10 ? 'emerald' : count >= 3 ? 'amber' : 'red';
                  return (
                    <div key={label} className="flex items-center gap-3">
                      <div className="w-36 sm:w-48 text-slate-700 text-sm font-medium truncate shrink-0 leading-tight">
                        {label.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                      </div>
                      <div className="flex-1 h-5 bg-slate-100 rounded-lg overflow-hidden">
                        <div
                          className={`h-full rounded-lg transition-all duration-700 ${
                            color === 'emerald' ? 'bg-gradient-to-r from-emerald-400 to-emerald-500' :
                            color === 'amber' ? 'bg-gradient-to-r from-amber-400 to-amber-500' :
                            'bg-gradient-to-r from-red-400 to-red-500'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <div className={`text-xs font-bold tabular-nums shrink-0 w-14 text-right ${
                        color === 'emerald' ? 'text-emerald-600' :
                        color === 'amber' ? 'text-amber-600' : 'text-red-500'
                      }`}>
                        {count} doc{count !== 1 ? 's' : ''}
                      </div>
                    </div>
                  );
                });
            })()}
          </div>
          {trainedLabels.some((l: string) => Number(classDocCounts[l] ?? 0) < 3) && (
            <p className="text-amber-600 text-xs mt-4 pt-3 border-t border-slate-100 flex items-center gap-1.5">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
              Types met minder dan 3 voorbeelden zijn minder betrouwbaar. Bevestig meer documenten.
            </p>
          )}
          {trainedLabels.some((l: string) => Number(classDocCounts[l] ?? 0) < 2) && (
            <p className="text-red-600 text-xs mt-1 flex items-center gap-1.5">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
              Types met 1 voorbeeld geven altijd 100% zekerheid — het model kan nog niets vergelijken.
            </p>
          )}
        </div>
      )}

      {!trainingDetails?.model_exists && !isTraining && (
        <div className="glass-card p-8 text-center">
          <div className="w-16 h-16 rounded-2xl bg-slate-100 border border-slate-200 flex items-center justify-center mx-auto mb-4">
            <FontAwesomeIcon icon={faGraduationCap} className="w-7 h-7 text-slate-300" />
          </div>
          <p className="text-slate-600 font-semibold mb-1">Nog niet getraind</p>
          <p className="text-slate-400 text-sm max-w-sm mx-auto">
            Bevestig documentclassificaties in de documentenlijst en klik dan op "Hertrainen".
          </p>
        </div>
      )}

      {/* Advanced settings */}
      <details className="group">
        <summary className="glass-card p-4 flex items-center gap-2 cursor-pointer select-none list-none">
          <FontAwesomeIcon icon={faChevronDown} className="w-3 h-3 text-slate-400 transition-transform group-open:rotate-180" />
          <span className="text-slate-600 font-medium text-sm">Geavanceerde instellingen</span>
          <span className="text-slate-400 text-xs ml-1">(NB model, BERT, stopwoorden, prioriteit)</span>
        </summary>
        <div className="mt-3 space-y-6">
          {/* Model selector */}
          <div className="glass-card p-4 bg-gradient-to-r from-[#22d3d3]/6 to-[#FFC1F3]/6 border border-[#22d3d3]/20">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-slate-800 font-semibold text-sm flex items-center gap-2">
                  <FontAwesomeIcon icon={faRobot} className="w-3.5 h-3.5 text-[#22d3d3]" />
                  Actief model
                </h3>
                <p className="text-slate-500 text-xs mt-0.5">Voor multi-tenant omgevingen</p>
              </div>
              <div className="relative" ref={dropdownTriggerRef}>
                <button
                  onClick={() => {
                    if (!showModelDropdown && dropdownTriggerRef.current) {
                      const rect = dropdownTriggerRef.current.getBoundingClientRect();
                      setDropdownPos({ top: rect.bottom + 4, right: window.innerWidth - rect.right });
                    }
                    setShowModelDropdown(!showModelDropdown);
                  }}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white hover:bg-slate-50 border border-slate-300 text-slate-800 text-sm font-medium transition-all min-w-[140px] justify-between shadow-sm cursor-pointer"
                >
                  <span>{selectedModel || 'Standaard'}</span>
                  <FontAwesomeIcon icon={faChevronDown} className="w-3 h-3 opacity-60" />
                </button>
                {showModelDropdown && typeof document !== 'undefined' && createPortal(
                  <>
                    <div className="fixed inset-0 z-[9998]" onClick={() => setShowModelDropdown(false)} />
                    <div className="fixed z-[9999] min-w-[180px] rounded-xl border border-slate-300 shadow-2xl overflow-hidden bg-white" style={{ top: dropdownPos.top, right: dropdownPos.right }}>
                      <button onClick={() => { setSelectedModel(undefined); setShowModelDropdown(false); }}
                        className={`w-full px-4 py-2.5 text-left text-sm hover:bg-slate-100 transition-colors cursor-pointer ${!selectedModel ? 'text-slate-800 bg-slate-50' : 'text-slate-500'}`}>
                        Standaard (alle types)
                        {!selectedModel && <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-[#22d3d3] float-right mt-0.5" />}
                      </button>
                      {availableModels?.models?.map((model) => (
                        <button key={model.name} onClick={() => { setSelectedModel(model.name); setShowModelDropdown(false); }}
                          className={`w-full px-4 py-2.5 text-left text-sm hover:bg-slate-100 transition-colors cursor-pointer ${selectedModel === model.name ? 'text-slate-800 bg-slate-50' : 'text-slate-500'}`}>
                          {model.name}
                          {model.is_trained && <span className="text-[10px] bg-emerald-100 text-emerald-600 px-1.5 py-0.5 rounded ml-2">trained</span>}
                          {selectedModel === model.name && <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-[#22d3d3] float-right mt-0.5" />}
                        </button>
                      ))}
                    </div>
                  </>,
                  document.body
                )}
              </div>
            </div>
          </div>

          {/* NaiveBayes detail section */}
          <div id="naive-bayes" className="glass-card p-5 scroll-mt-4">
            <h3 className="text-slate-800 font-semibold mb-4 flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-[#22d3d3]/10 border border-[#22d3d3]/20 flex items-center justify-center">
                <FontAwesomeIcon icon={faGraduationCap} className="text-[#22d3d3] w-3.5 h-3.5" />
              </div>
              Naive Bayes — Modeldetails
              <button
                onClick={() => trainMutation.mutate({ modelName: selectedModel, incremental: true })}
                disabled={isTraining}
                className="ml-auto flex items-center gap-1.5 px-3 py-1.5 text-xs bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg disabled:opacity-50 cursor-pointer transition-colors"
              >
                <FontAwesomeIcon icon={faRefresh} className="w-3 h-3" />
                Incrementeel
              </button>
            </h3>

            {trainingDetails?.model_exists && trainingDetails.model ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                    <div className="text-slate-500 text-xs mb-1">Vocabulaire</div>
                    <div className="text-slate-800 font-semibold">{trainingDetails.model.vocab_size.toLocaleString()}</div>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                    <div className="text-slate-500 text-xs mb-1">Types</div>
                    <div className="text-slate-800 font-semibold">{trainingDetails.model.labels.length}</div>
                  </div>
                  <div className="bg-[#22d3d3]/5 rounded-lg p-3 border border-[#22d3d3]/20">
                    <div className="text-[#22d3d3] text-xs mb-1">Min. zekerheid</div>
                    <div className="text-slate-800 font-semibold">{(trainingDetails.model.threshold * 100).toFixed(0)}%</div>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                    <div className="text-slate-500 text-xs mb-1">Alpha</div>
                    <div className="text-slate-800 font-semibold">{trainingDetails.model.alpha}</div>
                  </div>
                </div>

                {trainingDetails.important_tokens_by_label && (
                  <TrainedLabelsGrid
                    labels={trainingDetails.model.labels}
                    docCounts={trainingDetails.model.class_doc_counts}
                    trainingFilesByLabel={trainingDetails.training_files_by_label}
                    tokensByLabel={trainingDetails.important_tokens_by_label}
                    modelName={selectedModel}
                    allModelsData={aggregatedTrainingDetails}
                  />
                )}
              </div>
            ) : !isTraining ? (
              <p className="text-slate-500 text-sm">Nog geen NB model getraind.</p>
            ) : null}
          </div>

          {/* BERT section */}
          <BertClassifierSection selectedModel={selectedModel} />

          {/* Stopwords */}
          <div id="stopwords" className="glass-card p-5 scroll-mt-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-slate-800 font-semibold flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-slate-100 border border-slate-200 flex items-center justify-center">
                  <FontAwesomeIcon icon={faFilter} className="text-slate-500 w-3.5 h-3.5" />
                </div>
                Stopwoorden
              </h3>
              <button
                onClick={() => setShowStopwordsModal(true)}
                className="text-xs text-[#22d3d3] hover:underline cursor-pointer"
              >
                Bekijk lijst ({totalStopwords})
              </button>
            </div>
            <p className="text-slate-500 text-sm">
              {totalStopwords} veelgebruikte woorden worden genegeerd tijdens training (de, het, van, ...).
              Deze worden niet meegewogen bij documentclassificatie.
            </p>
          </div>
        </div>
      </details>

      {/* Stopwords modal */}
      {showStopwordsModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowStopwordsModal(false)}>
          <div className="glass-card max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-slate-800 font-semibold">Stopwoorden ({totalStopwords})</h3>
              <button onClick={() => setShowStopwordsModal(false)} className="text-slate-400 hover:text-slate-600 cursor-pointer">
                <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-4">
              {Object.entries(dutchStopwords).map(([category, words]) => (
                <div key={category}>
                  <h4 className="text-slate-600 text-xs font-semibold uppercase tracking-wider mb-2">{category}</h4>
                  <div className="flex flex-wrap gap-1.5">
                    {words.map(w => (
                      <span key={w} className="px-2 py-0.5 bg-slate-100 rounded text-slate-600 text-xs">{w}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
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
        <h3 className="text-slate-800 font-semibold flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-100 border border-blue-200 flex items-center justify-center">
            <FontAwesomeIcon icon={faBrain} className="text-blue-400 w-4 h-4" />
          </div>
          BERT Embeddings Classifier
          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full font-normal">
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
            className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
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

      <p className="text-slate-500 text-sm mb-4">
        BERT gebruikt deep learning voor semantisch tekstbegrip. Het begrijpt de <em>betekenis</em> van woorden
        in context, niet alleen hun frequentie.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Status */}
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="text-slate-500 text-xs mb-1">Status</div>
          {bertStatus?.model_exists ? (
            <div className="text-green-600 font-semibold flex items-center gap-2">
              <FontAwesomeIcon icon={faCheck} className="w-3 h-3" />
              Getraind
            </div>
          ) : (
            <div className="text-yellow-600 font-semibold flex items-center gap-2">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
              Niet getraind
            </div>
          )}
        </div>

        {/* Model */}
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="text-slate-500 text-xs mb-1">BERT Model</div>
          <div className="text-slate-800 font-mono text-xs truncate" title={bertStatus?.bert_model}>
            {bertStatus?.bert_model || 'multilingual-MiniLM'}
          </div>
          <p className="text-slate-400 text-[10px] mt-2 leading-relaxed">
            {bertStatus?.bert_model?.includes('robbert') || bertStatus?.bert_model?.includes('NetherlandsForensicInstitute')
              ? 'Ontwikkeld door het Nederlands Forensisch Instituut (NFI). Gespecialiseerd in Nederlandse juridische en zakelijke documenten. Herkent o.a. contracten, facturen, ID-documenten, bankafschriften en officiële correspondentie.'
              : 'Dit model zet tekst om naar numerieke vectoren die de betekenis vastleggen, waardoor documenten vergeleken kunnen worden op inhoud.'}
          </p>
        </div>

        {/* Labels */}
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="text-slate-500 text-xs mb-1">Document Types</div>
          <div className="text-slate-800 font-semibold">
            {bertStatus?.labels?.length || 0}
          </div>
        </div>

        {/* Threshold */}
        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
          <div className="text-slate-500 text-xs mb-1">Threshold</div>
          <div className="text-slate-800 font-semibold">
            {Math.round((bertStatus?.threshold || bertThreshold) * 100)}%
          </div>
        </div>
      </div>

      {/* Training Info */}
      {bertStatus?.last_summary && (
        <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
          <div className="text-green-600 text-sm font-medium mb-2">Laatste Training</div>
          {bertStatus.finished_at && (
            <div className="text-slate-500 text-xs mb-2">
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
              <span className="text-slate-400">Documenten:</span>
              <span className="text-slate-800 ml-1">{bertStatus.last_summary.total_documents}</span>
            </div>
            <div title="Aantal document types dat BERT kan herkennen">
              <span className="text-slate-400">Types:</span>
              <span className="text-slate-800 ml-1">{bertStatus.last_summary.labels?.length || 0}</span>
            </div>
            <div 
              title="Dimensie van de BERT vector: elk document wordt omgezet naar 768 getallen die de 'betekenis' van het document representeren. Dit is een vast kenmerk van het BERT model."
              className="cursor-help"
            >
              <span className="text-slate-400">Vector grootte:</span>
              <span className="text-slate-800 ml-1">{bertStatus.last_summary.embedding_dim}</span>
            </div>
            <div title="Minimale zekerheid voordat BERT een classificatie accepteert">
              <span className="text-slate-400">Threshold:</span>
              <span className="text-slate-800 ml-1">{Math.round(bertStatus.last_summary.threshold * 100)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* BERT Model Summary - Compact */}
      {bertStatus?.model_exists && bertStatus?.last_summary && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-100 border border-blue-200 flex items-center justify-center">
                <FontAwesomeIcon icon={faBrain} className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <h3 className="text-blue-700 font-bold text-sm">BERT Model</h3>
                {bertStatus.finished_at && (
                  <p className="text-slate-400 text-xs">
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
              <div className="bg-slate-50 rounded px-2 py-1 border border-slate-200">
                <span className="text-slate-400">Types:</span>
                <span className="text-slate-800 font-bold ml-1">{bertStatus.last_summary.labels?.length || 0}</span>
              </div>
              <div className="bg-slate-50 rounded px-2 py-1 border border-slate-200">
                <span className="text-slate-400">Docs:</span>
                <span className="text-slate-800 font-bold ml-1">{bertStatus.last_summary.total_documents || 0}</span>
              </div>
              <div className="bg-slate-50 rounded px-2 py-1 border border-slate-200">
                <span className="text-slate-400">Threshold:</span>
                <span className="text-slate-800 font-bold ml-1">{Math.round((bertStatus.last_summary.threshold || 0.7) * 100)}%</span>
              </div>
            </div>
          </div>
          
          {/* How BERT works explanation */}
          <details className="mt-2">
            <summary className="text-slate-500 text-xs cursor-pointer hover:text-slate-800 transition-colors flex items-center gap-1">
              <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-600" />
              Hoe werkt BERT classificatie?
            </summary>
            <div className="mt-2 pt-2 border-t border-slate-200 text-xs space-y-2">
              <div className="bg-slate-50 rounded p-3">
                <div className="text-blue-600 font-medium mb-2">BERT zet tekst om naar vectoren</div>
                <p className="text-slate-500 mb-2">
                  Elk document wordt omgezet naar een &quot;vector&quot; van <strong className="text-slate-800">{bertStatus.last_summary.embedding_dim || 768} getallen</strong>. 
                  Deze getallen representeren de &quot;betekenis&quot; van het document.
                </p>
                <div className="bg-white rounded p-2 font-mono text-[10px] text-slate-400">
                  &quot;Factuur voor levering...&quot; → [0.23, -0.87, 0.45, ... 768 getallen]
                </div>
              </div>
              <div className="bg-slate-50 rounded p-3">
                <div className="text-blue-600 font-medium mb-2">Vergelijking via afstand</div>
                <p className="text-slate-500">
                  BERT vergelijkt de vector van een nieuw document met de vectoren van bekende types. 
                  Het type met de kleinste &quot;afstand&quot; wint (als boven threshold).
                </p>
              </div>
              <div className="text-slate-400 text-[10px] flex items-center gap-1">
                <FontAwesomeIcon icon={faInfoCircle} className="w-3 h-3" />
                768 dimensies is standaard voor het &quot;all-MiniLM-L6-v2&quot; BERT model
              </div>
            </div>
          </details>

          {bertStatus.last_summary.samples_per_label && Object.keys(bertStatus.last_summary.samples_per_label).length > 0 && (
            <details className="mt-2">
              <summary className="text-slate-500 text-xs cursor-pointer hover:text-slate-800 transition-colors">
                Document types ({Object.keys(bertStatus.last_summary.samples_per_label).length}) •
                Semantisch begrip • Beter dan NB bij variaties
              </summary>
              <div className="mt-2 pt-2 border-t border-slate-200">
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
                          className="bg-slate-50 rounded px-2 py-1.5 border border-slate-200 flex items-center justify-between hover:bg-slate-100 hover:border-cyan-500/30 transition-colors cursor-pointer text-left"
                        >
                          <span className="text-slate-600 truncate">{label}</span>
                          <span className={`font-semibold shrink-0 ml-1 ${
                            quality === 'good' ? 'text-green-600' : 
                            quality === 'medium' ? 'text-yellow-600' : 
                            'text-red-600'
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
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setSelectedBertLabel(null)}>
              <div className="bg-white border border-slate-200 rounded-xl p-6 max-w-lg w-full shadow-2xl" onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-slate-800 font-semibold flex items-center gap-2">
                    <FontAwesomeIcon icon={faBrain} className="text-cyan-600 w-5 h-5" />
                    Model: {selectedBertLabel}
                  </h3>
                  <button onClick={() => setSelectedBertLabel(null)} className="text-slate-500 hover:text-slate-800 p-1">
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
                      <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 text-center">
                        <div className={`text-4xl font-bold ${
                          quality === 'good' ? 'text-green-600' : 
                          quality === 'medium' ? 'text-yellow-600' : 
                          'text-red-600'
                        }`}>{count}</div>
                        <div className="text-slate-500 text-sm mt-1">training documenten</div>
                      </div>

                      {/* What does this mean */}
                      <div className="space-y-3">
                        <h4 className="text-slate-600 font-medium text-sm">Wat betekent dit getal?</h4>
                        <p className="text-slate-500 text-sm">
                          In de map <code className="bg-white px-1 py-0.5 rounded text-cyan-700 text-xs">data/{selectedBertLabel}/</code> staan 
                          <strong className="text-slate-800"> {count} PDF&apos;s</strong> verdeeld over verschillende document type mappen 
                          (bijv. <code className="bg-white px-1 py-0.5 rounded text-slate-500 text-xs">offerte/</code>, 
                          <code className="bg-white px-1 py-0.5 rounded text-slate-500 text-xs">factuur/</code>, etc.).
                        </p>
                        <p className="text-slate-500 text-sm">
                          BERT leert hiermee welke documenten bij het model <strong className="text-cyan-700">{selectedBertLabel}</strong> horen 
                          en kan nieuwe documenten automatisch classificeren naar het juiste document type.
                        </p>
                      </div>

                      {/* Quality indicator */}
                      <div className={`rounded-lg p-3 border ${
                        quality === 'good' ? 'bg-green-50 border-green-200' :
                        quality === 'medium' ? 'bg-yellow-50 border-yellow-200' :
                        'bg-red-50 border-red-200'
                      }`}>
                        <div className="flex items-center gap-2">
                          <FontAwesomeIcon 
                            icon={quality === 'good' ? faCheck : quality === 'medium' ? faExclamationTriangle : faExclamationTriangle} 
                            className={`w-4 h-4 ${
                              quality === 'good' ? 'text-green-600' : 
                              quality === 'medium' ? 'text-yellow-600' : 
                              'text-red-600'
                            }`} 
                          />
                          <span className={`font-medium text-sm ${
                            quality === 'good' ? 'text-green-700' : 
                            quality === 'medium' ? 'text-yellow-700' : 
                            'text-red-600'
                          }`}>
                            {quality === 'good' ? 'Voldoende training data' : 
                             quality === 'medium' ? 'Matig - meer voorbeelden aanbevolen' : 
                             'Onvoldoende - voeg meer voorbeelden toe'}
                          </span>
                        </div>
                        <p className="text-slate-400 text-xs mt-2">
                          {quality === 'good'
                            ? 'Met 10+ documenten per document type kan BERT betrouwbaar classificeren.'
                            : quality === 'medium'
                            ? 'Met 5-9 documenten werkt BERT redelijk, maar meer voorbeelden per type verbeteren nauwkeurigheid.'
                            : 'Met minder dan 5 documenten per type is BERT onbetrouwbaar. Voeg meer voorbeelden toe.'}
                        </p>
                      </div>

                      {/* Folder structure explanation */}
                      <div className="bg-cyan-50 border border-cyan-200 rounded-lg p-3">
                        <div className="text-cyan-700 text-xs font-medium mb-2">Folder structuur:</div>
                        <div className="bg-white rounded p-2 font-mono text-[10px] text-slate-500 space-y-0.5">
                          <div className="text-purple-700">data/</div>
                          <div className="ml-3 text-blue-600">{selectedBertLabel}/ <span className="text-slate-400">← model ({count} PDF&apos;s totaal)</span></div>
                          <div className="ml-6 text-cyan-700">offerte/ <span className="text-slate-400">← document type</span></div>
                          <div className="ml-9 text-slate-400">offerte1.pdf, offerte2.pdf, ...</div>
                          <div className="ml-6 text-cyan-700">factuur/ <span className="text-slate-400">← document type</span></div>
                          <div className="ml-9 text-slate-400">factuur1.pdf, factuur2.pdf, ...</div>
                          <div className="ml-6 text-slate-400">... meer document types</div>
                        </div>
                        <p className="text-slate-400 text-[10px] mt-2">
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
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="text-red-600 text-sm font-medium mb-1">Error</div>
          <div className="text-red-600/70 text-xs">
            {trainBertMutation.error?.message || bertStatus?.last_error || 'Onbekende fout'}
          </div>
          {trainBertMutation.error && (
            <button
              onClick={() => trainBertMutation.reset()}
              className="mt-2 text-xs text-red-600 hover:text-red-600 underline"
            >
              Reset error
            </button>
          )}
        </div>
      )}

      {/* Training Progress */}
      {(trainBertMutation.isPending || bertStatus?.running) ? (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center gap-3">
            <FontAwesomeIcon icon={faSpinner} className="w-5 h-5 text-blue-400 animate-spin" />
            <div className="flex-1">
              <div className="text-blue-600 font-medium text-sm mb-1">
                {!selectedModel ? 'Training alle BERT modellen...' : `Training BERT model "${selectedModel}"...`}
              </div>
              <div className="text-slate-500 text-xs space-y-1">
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
                <div className="text-slate-400">
                  <span>Training stappen:</span>
                  <ul className="list-none ml-2 mt-1 space-y-0.5">
                    <li className="flex items-center gap-1.5">
                      {bertStatus?.model_downloaded ? (
                        <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-600 shrink-0" />
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
                  <div className="text-amber-600 text-xs mt-2 flex items-center gap-1.5">
                    <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                    Training duurt langer dan verwacht. Check de backend logs voor details.
                  </div>
                )}
                {bertStatus?.last_error && (
                  <div className="text-red-600 text-xs mt-2 flex items-center gap-1.5">
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
      <div className="mt-4 p-4 bg-slate-50 border border-slate-200 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="text-slate-500 text-sm">Similarity Threshold</span>
          <span className="text-blue-600 font-mono text-sm">{Math.round(bertThreshold * 100)}%</span>
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
        <div className="flex justify-between text-[10px] text-slate-400 mt-1">
          <span>50% - Meer matches</span>
          <span>95% - Alleen zekere matches</span>
        </div>
      </div>

      {/* Comparison with Naive Bayes - Compact */}
      <details className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <summary className="text-slate-600 font-medium text-sm cursor-pointer hover:text-slate-800 transition-colors">
          BERT vs Naive Bayes
        </summary>
        <div className="mt-3">
          <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-blue-600 font-medium mb-2 flex items-center gap-2">
              <div className="w-5 h-5 rounded bg-blue-100 border border-blue-200 flex items-center justify-center">
                <FontAwesomeIcon icon={faBrain} className="w-2.5 h-2.5" />
              </div>
              BERT
            </div>
            <ul className="text-slate-500 space-y-1.5">
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-600 shrink-0" /> Begrijpt context</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-600 shrink-0" /> Synoniemen</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-600 shrink-0" /> Weinig data</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-600 shrink-0" /> ~100ms</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-600 shrink-0" /> ~500MB RAM</li>
            </ul>
          </div>
          <div>
            <div className="text-purple-600 font-medium mb-2 flex items-center gap-2">
              <div className="w-5 h-5 rounded bg-purple-100 border border-purple-200 flex items-center justify-center">
                <FontAwesomeIcon icon={faBolt} className="w-2.5 h-2.5" />
              </div>
              Naive Bayes
            </div>
            <ul className="text-slate-500 space-y-1.5">
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-600 shrink-0" /> Zeer snel (~1ms)</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-600 shrink-0" /> Weinig geheugen</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-600 shrink-0" /> Veel data</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-600 shrink-0" /> Mist context</li>
              <li className="flex items-center gap-1.5"><FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 text-amber-600 shrink-0" /> Woordfrequentie</li>
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
  const [showElaConfig, setShowElaConfig] = useState(false);

  const { data: appSettings, isLoading } = useQuery({
    queryKey: ['app-settings'],
    queryFn: () => getAppSettings(),
  });

  const getSetting = (key: string, fallback: string) =>
    appSettings?.find(s => s.key === key)?.value ?? fallback;

  const elaEnabled    = getSetting('ela_enabled', 'false') === 'true';
  const exifEnabled   = getSetting('exif_enabled', 'false') === 'true';
  const elaMinSize    = getSetting('ela_min_size', '150');
  const elaQuality    = getSetting('ela_quality', '95');
  const elaAllowNonJpeg = getSetting('ela_allow_non_jpeg', 'false') === 'true';

  const updateSettingMutation = useMutation({
    mutationFn: ({ key, value }: { key: string; value: string }) => updateAppSetting(key, value),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['app-settings'] }),
  });

  const Toggle = ({ settingKey, enabled }: { settingKey: string; enabled: boolean }) => (
    <button
      onClick={() => updateSettingMutation.mutate({ key: settingKey, value: enabled ? 'false' : 'true' })}
      disabled={updateSettingMutation.isPending}
      className={`relative flex-shrink-0 w-14 h-7 rounded-full transition-colors cursor-pointer ${enabled ? 'bg-green-500' : 'bg-slate-200'}`}
    >
      <div className={`absolute top-1 left-1 w-5 h-5 bg-white rounded-full transition-transform ${enabled ? 'translate-x-7' : 'translate-x-0'}`} />
    </button>
  );

  const ALWAYS_ON = [
    {
      icon: faFileAlt,
      color: 'text-blue-600',
      bg: 'bg-blue-50 border-blue-200',
      title: 'PDF Metadata Analyse',
      desc: 'Controleert aanmaaksoftware (PyPDF, iText, Photoshop), tijdstempels en ontbrekende metadata. Altijd actief.',
    },
    {
      icon: faSearch,
      color: 'text-purple-600',
      bg: 'bg-purple-50 border-purple-200',
      title: 'Tekst Anomalie Detectie',
      desc: 'Detecteert unicode-manipulatie, overdreven herhaalpatronen en verdachte tekenreeksen. Altijd actief.',
    },
    {
      icon: faShieldAlt,
      color: 'text-emerald-600',
      bg: 'bg-emerald-50 border-emerald-200',
      title: 'Betrouwbaarheidsscore',
      desc: 'Waarschuwt bij lage classificatiezekerheid (< 50%) of grote afstand tot bekende documenttypes. Altijd actief.',
    },
    {
      icon: faCheckCircle,
      color: 'text-yellow-600',
      bg: 'bg-yellow-50 border-yellow-200',
      title: 'Veldvalidatie',
      desc: 'Controleert geëxtraheerde velden op vervaldatums, formaten en plausibiliteit via configureerbare regels per documenttype. Altijd actief.',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-red-100 border border-red-200 flex items-center justify-center">
            <FontAwesomeIcon icon={faShieldAlt} className="text-red-600 w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-800">Fraud Detection</h2>
            <p className="text-slate-500 text-sm">Alle analyses worden opgeslagen in de database en direct toegepast</p>
          </div>
        </div>
      </div>

      {/* Always-on modules */}
      <div className="glass-card p-6">
        <h3 className="text-slate-800 font-semibold text-sm mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faCheckCircle} className="w-4 h-4 text-emerald-400" />
          Altijd actief
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {ALWAYS_ON.map(m => (
            <div key={m.title} className={`p-4 rounded-xl border ${m.bg} flex items-start gap-3`}>
              <FontAwesomeIcon icon={m.icon} className={`${m.color} w-4 h-4 mt-0.5 flex-shrink-0`} />
              <div>
                <div className="text-slate-800 text-sm font-medium mb-0.5">{m.title}</div>
                <div className="text-slate-400 text-xs leading-relaxed">{m.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Toggleable modules */}
      <div className="glass-card p-6">
        <h3 className="text-slate-800 font-semibold text-sm mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faImage} className="w-4 h-4 text-orange-600" />
          Optionele beeldanalyse
          <span className="text-slate-400 text-xs font-normal ml-1">— standaard uit, kan ruis geven bij scans</span>
        </h3>
        {isLoading ? (
          <div className="flex items-center justify-center py-6">
            <FontAwesomeIcon icon={faSpinner} className="w-5 h-5 text-slate-400 animate-spin" />
          </div>
        ) : (
          <div className="space-y-3">
            {/* ELA */}
            <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-200">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <div className="w-10 h-10 rounded-lg bg-red-100 border border-red-200 flex items-center justify-center flex-shrink-0">
                  <FontAwesomeIcon icon={faImage} className="w-5 h-5 text-red-600" />
                </div>
                <div className="min-w-0">
                  <div className="text-slate-800 font-medium text-sm">Error Level Analysis (ELA)</div>
                  <div className="text-slate-400 text-xs mt-0.5">Detecteert JPEG manipulatie via compressie-inconsistenties in afbeeldingen</div>
                </div>
              </div>
              <Toggle settingKey="ela_enabled" enabled={elaEnabled} />
            </div>

            {/* ELA advanced config */}
            {elaEnabled && (
              <div className="ml-4 border-l-2 border-red-500/30 pl-4">
                <button
                  onClick={() => setShowElaConfig(v => !v)}
                  className="text-slate-400 text-xs flex items-center gap-1 hover:text-slate-600 transition-colors mb-3"
                >
                  <FontAwesomeIcon icon={showElaConfig ? faChevronDown : faChevronRight} className="w-3 h-3" />
                  Geavanceerde ELA instellingen
                </button>
                {showElaConfig && (
                  <div className="space-y-3 bg-slate-50 rounded-lg p-4 border border-slate-200">
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <div className="text-slate-600 text-xs font-medium">Minimale afbeeldingsgrootte (px)</div>
                        <div className="text-slate-400 text-xs">Afbeeldingen kleiner dan dit worden overgeslagen</div>
                      </div>
                      <input
                        type="number"
                        min={50} max={500}
                        defaultValue={elaMinSize}
                        onBlur={e => updateSettingMutation.mutate({ key: 'ela_min_size', value: e.target.value })}
                        className="w-20 bg-slate-100 border border-slate-300 rounded px-2 py-1 text-slate-800 text-xs text-right"
                      />
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <div className="text-slate-600 text-xs font-medium">JPEG kwaliteit (85–99)</div>
                        <div className="text-slate-400 text-xs">Hogere waarde = subtielere manipulaties zichtbaar</div>
                      </div>
                      <input
                        type="number"
                        min={85} max={99}
                        defaultValue={elaQuality}
                        onBlur={e => updateSettingMutation.mutate({ key: 'ela_quality', value: e.target.value })}
                        className="w-20 bg-slate-100 border border-slate-300 rounded px-2 py-1 text-slate-800 text-xs text-right"
                      />
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <div className="text-slate-600 text-xs font-medium">Niet-JPEG ook analyseren</div>
                        <div className="text-slate-400 text-xs">PNG, BMP etc. — minder betrouwbaar resultaat</div>
                      </div>
                      <Toggle settingKey="ela_allow_non_jpeg" enabled={elaAllowNonJpeg} />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* EXIF */}
            <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-200">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <div className="w-10 h-10 rounded-lg bg-orange-100 border border-orange-200 flex items-center justify-center flex-shrink-0">
                  <FontAwesomeIcon icon={faImage} className="w-5 h-5 text-orange-600" />
                </div>
                <div className="min-w-0">
                  <div className="text-slate-800 font-medium text-sm">EXIF Analyse</div>
                  <div className="text-slate-400 text-xs mt-0.5">Detecteert bewerkingssoftware in afbeeldingsmetadata (Photoshop, GIMP, Lightroom)</div>
                </div>
              </div>
              <Toggle settingKey="exif_enabled" enabled={exifEnabled} />
            </div>
          </div>
        )}
      </div>

      {/* Info */}
      <div className="glass-card p-6">
        <h3 className="text-slate-800 font-semibold text-sm mb-3 flex items-center gap-2">
          <FontAwesomeIcon icon={faLightbulb} className="w-4 h-4 text-yellow-400" />
          Signalen beheren
        </h3>
        <p className="text-slate-400 text-sm">
          Fraude-signalen (zoekregels op basis van sleutelwoorden of regex) worden beheerd via het{' '}
          <a href="/signals" className="text-blue-600 hover:text-blue-700 underline">Signalen</a>-tabblad.
          Per documenttype zijn veldvalidatieregels instelbaar via{' '}
          <a href="/document-types" className="text-blue-600 hover:text-blue-700 underline">Documenttypen</a>.
        </p>
      </div>
    </div>
  );
}

// LLM Tab - Switch between Ollama and vLLM
function LLMTab() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const isSuperAdmin = user?.role === 'super_admin';
  const [switching, setSwitching] = useState(false);
  const [editingProvider, setEditingProvider] = useState<'ollama' | 'vllm' | null>(null);
  const [editUrl, setEditUrl] = useState('');
  const [editModel, setEditModel] = useState('');
  const [editMaxTokens, setEditMaxTokens] = useState('');
  const [editContextLength, setEditContextLength] = useState('');
  const [editSaved, setEditSaved] = useState(false);

  const updateMutation = useMutation({
    mutationFn: ({ provider, data }: { provider: 'ollama' | 'vllm'; data: Parameters<typeof updateLLMSettings>[1] }) =>
      updateLLMSettings(provider, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-settings'] });
      queryClient.invalidateQueries({ queryKey: ['llm-health'] });
      setEditingProvider(null);
      setEditSaved(true);
      setTimeout(() => setEditSaved(false), 2000);
    },
  });

  const startEdit = (provider: 'ollama' | 'vllm', e: React.MouseEvent) => {
    e.stopPropagation();
    const prov = settings?.providers[provider];
    setEditUrl(prov?.base_url || '');
    setEditModel(prov?.model || '');
    setEditMaxTokens(String(prov?.max_tokens || ''));
    setEditContextLength(String(prov?.context_length || ''));
    setEditingProvider(provider);
  };

  const saveEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!editingProvider) return;
    updateMutation.mutate({
      provider: editingProvider,
      data: {
        base_url: editUrl || undefined,
        model: editModel || undefined,
        max_tokens: editMaxTokens ? parseInt(editMaxTokens, 10) : undefined,
        context_length: (editingProvider === 'vllm' && editContextLength) ? parseInt(editContextLength, 10) : undefined,
      },
    });
  };

  const cancelEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingProvider(null);
  };

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
          <div className="w-10 h-10 rounded-xl bg-purple-100 border border-purple-200 flex items-center justify-center">
            <FontAwesomeIcon icon={faBrain} className="text-purple-400 w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-800">LLM Provider</h2>
            <p className="text-slate-500 text-sm">Kies tussen Ollama en vLLM voor AI-gebaseerde classificatie</p>
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
                  ? 'bg-green-50 border-green-200'
                  : 'bg-slate-50 border-slate-200 hover:border-slate-300 hover:bg-slate-100'
              }`}
            >
              {switching && settings?.active_provider !== 'ollama' && (
                <div className="absolute inset-0 bg-black/50 rounded-xl flex items-center justify-center">
                  <FontAwesomeIcon icon={faSpinner} className="w-6 h-6 text-slate-800 animate-spin" />
                </div>
              )}
              
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-orange-100 border border-orange-200 flex items-center justify-center">
                    <span className="text-orange-600 font-bold text-lg">🦙</span>
                  </div>
                  <div>
                    <h3 className="text-slate-800 font-semibold">Ollama</h3>
                    <p className="text-slate-400 text-xs">Lokale LLM server</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {settings?.active_provider === 'ollama' && (
                    <span className="px-2 py-1 bg-green-100 text-green-600 text-xs rounded-full font-medium">
                      Actief
                    </span>
                  )}
                  {isSuperAdmin && editingProvider !== 'ollama' && (
                    <button
                      onClick={(e) => startEdit('ollama', e)}
                      className="p-1 text-slate-400 hover:text-slate-600 rounded"
                      title="Bewerk instellingen"
                    >
                      <FontAwesomeIcon icon={faEdit} className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>

              {editingProvider === 'ollama' ? (
                <div className="space-y-2 text-sm" onClick={e => e.stopPropagation()}>
                  <div>
                    <label className="text-slate-400 text-xs block mb-1">URL</label>
                    <input type="text" value={editUrl} onChange={e => setEditUrl(e.target.value)} className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs font-mono text-slate-700 focus:outline-none focus:border-blue-400" />
                  </div>
                  <div>
                    <label className="text-slate-400 text-xs block mb-1">Model</label>
                    <input type="text" value={editModel} onChange={e => setEditModel(e.target.value)} className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs text-slate-700 focus:outline-none focus:border-blue-400" />
                  </div>
                  <div>
                    <label className="text-slate-400 text-xs block mb-1">Max tokens</label>
                    <input type="number" value={editMaxTokens} onChange={e => setEditMaxTokens(e.target.value)} className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs text-slate-700 focus:outline-none focus:border-blue-400" />
                  </div>
                  <div className="flex gap-2 pt-1">
                    <button onClick={saveEdit} disabled={updateMutation.isPending} className="flex items-center gap-1 px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 disabled:opacity-50">
                      <FontAwesomeIcon icon={faSave} className="w-3 h-3" />
                      Opslaan
                    </button>
                    <button onClick={cancelEdit} className="px-3 py-1 text-slate-500 text-xs rounded hover:bg-slate-100">
                      Annuleren
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">URL:</span>
                    <span className="text-slate-600 font-mono text-xs">{settings?.providers.ollama.base_url}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Model:</span>
                    <span className="text-slate-600">{settings?.providers.ollama.model}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Max tokens:</span>
                    <span className="text-slate-600">{settings?.providers.ollama.max_tokens?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Status:</span>
                    {health?.providers.ollama.reachable ? (
                      <span className="flex items-center gap-1 text-green-600">
                        <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                        Online
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-red-600">
                        <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                        Offline
                      </span>
                    )}
                  </div>
                  {health?.providers.ollama.reachable && (
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">Model beschikbaar:</span>
                      {health.providers.ollama.model_available ? (
                        <FontAwesomeIcon icon={faCheck} className="text-green-600 w-4 h-4" />
                      ) : (
                        <FontAwesomeIcon icon={faTimes} className="text-red-600 w-4 h-4" />
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="mt-4 pt-3 border-t border-slate-200">
                <p className="text-slate-400 text-xs">
                  Sequentiële verwerking • Ideaal voor ontwikkeling
                </p>
              </div>
            </div>

            {/* vLLM Card */}
            <div
              onClick={() => handleSwitch('vllm')}
              className={`relative p-5 rounded-xl border-2 transition-all cursor-pointer ${
                settings?.active_provider === 'vllm'
                  ? 'bg-green-50 border-green-200'
                  : 'bg-slate-50 border-slate-200 hover:border-slate-300 hover:bg-slate-100'
              }`}
            >
              {switching && settings?.active_provider !== 'vllm' && (
                <div className="absolute inset-0 bg-black/50 rounded-xl flex items-center justify-center">
                  <FontAwesomeIcon icon={faSpinner} className="w-6 h-6 text-slate-800 animate-spin" />
                </div>
              )}

              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-blue-100 border border-blue-200 flex items-center justify-center">
                    <FontAwesomeIcon icon={faRocket} className="text-blue-400 w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-slate-800 font-semibold">vLLM</h3>
                    <p className="text-slate-400 text-xs">High-performance inference</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {settings?.active_provider === 'vllm' && (
                    <span className="px-2 py-1 bg-green-100 text-green-600 text-xs rounded-full font-medium">
                      Actief
                    </span>
                  )}
                  {isSuperAdmin && editingProvider !== 'vllm' && (
                    <button
                      onClick={(e) => startEdit('vllm', e)}
                      className="p-1 text-slate-400 hover:text-slate-600 rounded"
                      title="Bewerk instellingen"
                    >
                      <FontAwesomeIcon icon={faEdit} className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>

              {editingProvider === 'vllm' ? (
                <div className="space-y-2 text-sm" onClick={e => e.stopPropagation()}>
                  <div>
                    <label className="text-slate-400 text-xs block mb-1">URL</label>
                    <input type="text" value={editUrl} onChange={e => setEditUrl(e.target.value)} className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs font-mono text-slate-700 focus:outline-none focus:border-blue-400" />
                  </div>
                  <div>
                    <label className="text-slate-400 text-xs block mb-1">Model</label>
                    <input type="text" value={editModel} onChange={e => setEditModel(e.target.value)} className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs text-slate-700 focus:outline-none focus:border-blue-400" />
                  </div>
                  <div>
                    <label className="text-slate-400 text-xs block mb-1">Max tokens (output)</label>
                    <input type="number" value={editMaxTokens} onChange={e => setEditMaxTokens(e.target.value)} className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs text-slate-700 focus:outline-none focus:border-blue-400" />
                  </div>
                  <div>
                    <label className="text-slate-400 text-xs block mb-1">Context venster (input + output)</label>
                    <input type="number" value={editContextLength} onChange={e => setEditContextLength(e.target.value)} className="w-full px-2 py-1 bg-white border border-slate-300 rounded text-xs text-slate-700 focus:outline-none focus:border-blue-400" placeholder="4096" />
                  </div>
                  <div className="flex gap-2 pt-1">
                    <button onClick={saveEdit} disabled={updateMutation.isPending} className="flex items-center gap-1 px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 disabled:opacity-50">
                      <FontAwesomeIcon icon={faSave} className="w-3 h-3" />
                      Opslaan
                    </button>
                    <button onClick={cancelEdit} className="px-3 py-1 text-slate-500 text-xs rounded hover:bg-slate-100">
                      Annuleren
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">URL:</span>
                    <span className="text-slate-600 font-mono text-xs">{settings?.providers.vllm.base_url}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Model:</span>
                    <span className="text-slate-600">{settings?.providers.vllm.model}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Max tokens:</span>
                    <span className="text-slate-600">{settings?.providers.vllm.max_tokens?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Context venster:</span>
                    <span className="text-slate-600">{settings?.providers.vllm.context_length?.toLocaleString() ?? '4096'}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Status:</span>
                    {health?.providers.vllm.reachable ? (
                      <span className="flex items-center gap-1 text-green-600">
                        <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                        Online
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-red-600">
                        <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                        Offline
                      </span>
                    )}
                  </div>
                  {health?.providers.vllm.reachable && (
                    <div className="flex justify-between items-center">
                      <span className="text-slate-400">Model beschikbaar:</span>
                      {health.providers.vllm.model_available ? (
                        <FontAwesomeIcon icon={faCheck} className="text-green-600 w-4 h-4" />
                      ) : (
                        <FontAwesomeIcon icon={faTimes} className="text-red-600 w-4 h-4" />
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="mt-4 pt-3 border-t border-slate-200">
                <p className="text-slate-400 text-xs">
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
                ? 'text-slate-400 cursor-not-allowed'
                : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100 cursor-pointer'
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
        <h3 className="text-slate-800 font-semibold mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faLightbulb} className="text-yellow-400 w-4 h-4" />
          Configuratie
        </h3>
        <div className="text-slate-500 text-sm space-y-3">
          <p>
            Alle LLM-instellingen worden opgeslagen in de database. Wijzigingen via deze pagina zijn direct actief — geen herstart nodig.
          </p>
          <div className="bg-white rounded-lg p-4 text-xs space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-emerald-600 font-mono">provider</span>
              <span className="text-slate-400">—</span>
              <span>Actieve provider (ollama of vllm), direct omschakelbaar</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-emerald-600 font-mono">base_url</span>
              <span className="text-slate-400">—</span>
              <span>Endpoint van de LLM server</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-emerald-600 font-mono">model</span>
              <span className="text-slate-400">—</span>
              <span>Modelnaam zoals geregistreerd op de server</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-emerald-600 font-mono">max_tokens</span>
              <span className="text-slate-400">—</span>
              <span>Maximaal aantal tokens in het antwoord</span>
            </div>
          </div>
          <p className="text-slate-400 text-xs">
            Alleen <code className="bg-slate-100 px-1 rounded">DATABASE_URL</code> en bestandspaden staan nog in de <code className="bg-slate-100 px-1 rounded">.env</code> file.
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
      <div className="glass-card p-4 sm:p-6 border border-blue-200 bg-blue-50">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-100 border border-blue-200 flex items-center justify-center shrink-0">
            <FontAwesomeIcon icon={faFilter} className="text-blue-400 w-5 h-5" />
          </div>
          <div>
            <h3 className="text-slate-800 font-semibold mb-1">Wat zijn Skip Markers?</h3>
            <p className="text-slate-500 text-sm leading-relaxed">
              Skip markers zijn tekstpatronen die aangeven waar de documentverwerking moet stoppen.
              Als een skip marker wordt gevonden, wordt alle tekst daarna genegeerd. Dit is handig voor:
            </p>
            <ul className="text-slate-500 text-sm mt-2 space-y-1 list-disc list-inside">
              <li>Algemene voorwaarden onderaan documenten</li>
              <li>Repeterende headers/footers</li>
              <li>Automatisch gegenereerde content</li>
            </ul>
            <p className="text-slate-400 text-xs mt-3 flex items-center gap-1.5">
              <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 text-amber-600" />
              Dit bespaart LLM tokens en vermindert verwerkingstijd voor grote documenten.
            </p>
          </div>
        </div>
      </div>

      {/* Create Form */}
      {!showCreateForm ? (
        <button
          onClick={() => setShowCreateForm(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-100 hover:bg-blue-200 border border-blue-200 text-blue-700 text-sm font-medium transition-all"
        >
          <FontAwesomeIcon icon={faPlus} className="w-3 h-3" />
          Nieuwe Skip Marker
        </button>
      ) : (
        <div className="glass-card p-4 sm:p-6 space-y-4">
          <h3 className="text-slate-800 font-semibold">Nieuwe Skip Marker</h3>
          
          <div>
            <label className="block text-slate-500 text-xs mb-1">Patroon *</label>
            <input
              type="text"
              value={newPattern}
              onChange={(e) => setNewPattern(e.target.value)}
              placeholder="bijv. Algemene Voorwaarden"
              className="w-full px-3 py-2 rounded-lg bg-slate-50 border border-slate-200 text-slate-800 text-sm focus:border-blue-500/50 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-slate-500 text-xs mb-1">Beschrijving</label>
            <input
              type="text"
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              placeholder="Optionele uitleg"
              className="w-full px-3 py-2 rounded-lg bg-slate-50 border border-slate-200 text-slate-800 text-sm focus:border-blue-500/50 focus:outline-none"
            />
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={newIsRegex}
              onChange={(e) => setNewIsRegex(e.target.checked)}
              className="w-4 h-4 rounded border-slate-300 bg-slate-50 text-blue-500 focus:ring-blue-500/50"
            />
            <span className="text-slate-500 text-sm">Regex patroon</span>
            {newIsRegex && (
              <span className="text-amber-600/80 text-xs">(bijv. Pagina \d+ van \d+)</span>
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
              className="px-4 py-2 rounded-lg bg-slate-100 hover:bg-slate-100 text-slate-500 text-sm"
            >
              Annuleren
            </button>
          </div>
        </div>
      )}

      {/* Markers List */}
      <div className="glass-card p-4 sm:p-6">
        <h3 className="text-slate-800 font-semibold mb-4">
          Skip Markers ({markers?.length || 0})
        </h3>

        {isLoading ? (
          <div className="text-slate-500 text-sm">Laden...</div>
        ) : !markers?.length ? (
          <div className="text-slate-500 text-sm">Geen skip markers geconfigureerd.</div>
        ) : (
          <div className="space-y-2">
            {markers.map((marker) => (
              <div
                key={marker.id}
                className={`flex items-center justify-between gap-3 p-3 rounded-lg border transition-all ${
                  marker.is_active
                    ? 'bg-slate-50 border-slate-200'
                    : 'bg-white border-slate-100 opacity-60'
                }`}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <code className="text-slate-800 text-sm font-mono truncate">
                      {marker.pattern}
                    </code>
                    {marker.is_regex && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-100 text-purple-700 border border-purple-200">
                        regex
                      </span>
                    )}
                  </div>
                  {marker.description && (
                    <div className="text-slate-400 text-xs mt-0.5 truncate">
                      {marker.description}
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => toggleMutation.mutate({ id: marker.id, is_active: !marker.is_active })}
                    className={`p-2 rounded-lg transition-all ${
                      marker.is_active
                        ? 'text-emerald-600 hover:bg-emerald-100'
                        : 'text-slate-400 hover:bg-slate-100'
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
                    className="p-2 rounded-lg text-red-500 hover:text-red-600 hover:bg-red-100 transition-all"
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
        <div className="glass-card p-4 sm:p-6 border-2 border-green-200 bg-green-50">
          <div className="flex items-center gap-2 mb-4">
            <FontAwesomeIcon icon={faCheck} className="text-green-600 w-4 h-4" />
            <h3 className="text-slate-800 font-semibold text-sm sm:text-base">API Key Aangemaakt!</h3>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-2 sm:p-3 mb-4">
            <p className="text-amber-700 text-xs sm:text-sm flex items-start sm:items-center gap-2 font-medium">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 sm:w-4 sm:h-4 shrink-0 mt-0.5 sm:mt-0" />
              <span>Kopieer de client secret nu! Deze wordt niet meer getoond.</span>
            </p>
          </div>
          <div className="space-y-3">
            <div>
              <label className="text-slate-500 text-xs block mb-1">Client ID</label>
              <div className="flex items-center gap-1 sm:gap-2">
                <code className="flex-1 bg-white px-2 sm:px-3 py-2 rounded text-slate-800 font-mono text-[10px] sm:text-sm overflow-x-auto">
                  {createdKey.client_id}
                </code>
                <button
                  onClick={() => copyToClipboard(createdKey.client_id, 'client_id')}
                  className="p-1.5 sm:p-2 bg-slate-100 rounded hover:bg-slate-200 transition-colors shrink-0"
                >
                  <FontAwesomeIcon 
                    icon={copiedField === 'client_id' ? faCheck : faCopy} 
                    className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${copiedField === 'client_id' ? 'text-green-600' : 'text-slate-500'}`} 
                  />
                </button>
              </div>
            </div>
            <div>
              <label className="text-slate-500 text-xs block mb-1">Client Secret</label>
              <div className="flex items-center gap-1 sm:gap-2">
                <code className="flex-1 bg-white px-2 sm:px-3 py-2 rounded text-slate-800 font-mono text-[10px] sm:text-sm overflow-x-auto">
                  {showSecret ? createdKey.client_secret : '•'.repeat(20)}
                </code>
                <button
                  onClick={() => setShowSecret(!showSecret)}
                  className="p-1.5 sm:p-2 bg-slate-100 rounded hover:bg-slate-200 transition-colors shrink-0"
                >
                  <FontAwesomeIcon icon={showSecret ? faEyeSlash : faEye} className="text-slate-500 w-3.5 h-3.5 sm:w-4 sm:h-4" />
                </button>
                <button
                  onClick={() => copyToClipboard(createdKey.client_secret, 'client_secret')}
                  className="p-1.5 sm:p-2 bg-slate-100 rounded hover:bg-slate-200 transition-colors shrink-0"
                >
                  <FontAwesomeIcon 
                    icon={copiedField === 'client_secret' ? faCheck : faCopy} 
                    className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${copiedField === 'client_secret' ? 'text-green-600' : 'text-slate-500'}`} 
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
            className="mt-4 px-4 py-2 bg-slate-100 text-slate-800 rounded-lg hover:bg-slate-200 transition-colors text-sm"
          >
            Sluiten
          </button>
        </div>
      )}

      {/* Header */}
      <div className="glass-card p-4 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <h2 className="text-slate-800 text-lg sm:text-xl font-semibold flex items-center gap-2">
            <div className="w-9 h-9 rounded-lg bg-amber-100 border border-amber-200 flex items-center justify-center">
              <FontAwesomeIcon icon={faKey} className="text-amber-600 w-4 h-4" />
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
          <div className="bg-slate-50 rounded-lg p-3 sm:p-4 mb-4 border border-slate-200">
            <h4 className="text-slate-800 font-medium mb-3 text-sm sm:text-base">Nieuwe API Key</h4>
            <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
              <input
                type="text"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                placeholder="Naam (bijv. Production)"
                className="flex-1 px-3 py-2 bg-slate-100 border border-slate-300 rounded-lg text-slate-800 placeholder-slate-400 text-sm"
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
                  className="flex-1 sm:flex-none px-3 sm:px-4 py-2 bg-slate-100 text-slate-800 rounded-lg hover:bg-slate-200 transition-colors text-sm"
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
            <FontAwesomeIcon icon={faKey} className="text-slate-400 text-3xl animate-pulse" />
          </div>
        ) : apiKeys?.length === 0 ? (
          <div className="text-center py-8 bg-slate-50 rounded-lg border border-dashed border-slate-300">
            <FontAwesomeIcon icon={faKey} className="text-slate-500 text-3xl mb-2" />
            <p className="text-slate-400">Geen API keys</p>
            <p className="text-slate-400 text-sm">Maak een nieuwe key aan om te beginnen</p>
          </div>
        ) : (
          <div className="space-y-2">
            {apiKeys?.map((key) => (
              <div
                key={key.id}
                className={`flex flex-col sm:flex-row sm:items-center justify-between gap-2 sm:gap-4 p-3 sm:p-4 rounded-lg border ${
                  key.is_active
                    ? 'bg-slate-50 border-slate-200'
                    : 'bg-red-50 border-red-200 opacity-60'
                }`}
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`w-2 h-2 rounded-full shrink-0 ${key.is_active ? 'bg-green-400' : 'bg-red-400'}`} />
                  <div className="min-w-0">
                    <div className="text-slate-800 font-medium text-sm sm:text-base truncate">{key.name}</div>
                    <code className="text-slate-400 text-[10px] sm:text-xs font-mono block truncate">{key.client_id}</code>
                  </div>
                </div>
                <div className="flex items-center gap-2 self-end sm:self-auto shrink-0">
                  <button
                    onClick={() => toggleMutation.mutate({ id: key.id, is_active: !key.is_active })}
                    className={`px-2 sm:px-3 py-1 rounded text-[10px] sm:text-xs transition-colors ${
                      key.is_active
                        ? 'bg-amber-100 text-amber-700 hover:bg-amber-200'
                        : 'bg-green-100 text-green-600 hover:bg-green-200'
                    }`}
                  >
                    {key.is_active ? 'Deactiveer' : 'Activeer'}
                  </button>
                  <button
                    onClick={() => setConfirmDelete({ key })}
                    className="p-1.5 sm:p-2 text-red-600 hover:bg-red-100 rounded transition-colors"
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
        <h3 className="text-slate-800 font-semibold mb-4 text-sm sm:text-base">API Gebruik</h3>
        <div className="bg-white rounded-lg p-3 sm:p-4 font-mono text-xs sm:text-sm overflow-x-auto">
          <div className="text-slate-500 mb-2"># Authenticatie met API key</div>
          <div className="text-cyan-600 whitespace-nowrap">curl -X GET "http://localhost:8000/api/documents" \</div>
          <div className="text-cyan-600 pl-2 sm:pl-4 whitespace-nowrap">-H "X-Client-ID: your_client_id" \</div>
          <div className="text-cyan-600 pl-2 sm:pl-4 whitespace-nowrap">-H "X-Client-Secret: your_client_secret"</div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {confirmDelete && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-card max-w-md w-full p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                  <FontAwesomeIcon icon={faTrash} className="text-red-600 w-5 h-5" />
                </div>
                <h3 className="text-slate-800 text-lg font-semibold">API Key verwijderen</h3>
              </div>
              <button
                onClick={() => setConfirmDelete(null)}
                className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <FontAwesomeIcon icon={faTimes} className="w-5 h-5" />
              </button>
            </div>

            <div className="mb-6">
              <p className="text-slate-600 mb-2">
                Weet je zeker dat je deze API key wilt verwijderen?
              </p>
              <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
                <p className="text-slate-800 font-medium truncate">
                  {confirmDelete.key.name}
                </p>
                <p className="text-slate-500 text-sm font-mono">
                  {confirmDelete.key.client_id}
                </p>
              </div>
              <p className="text-slate-500 text-sm mt-3">
                Deze actie kan niet ongedaan worden gemaakt.
              </p>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={() => setConfirmDelete(null)}
                className="flex-1 px-4 py-2 bg-slate-100 text-slate-800 rounded-lg hover:bg-slate-200 transition-colors"
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

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// MCP Tab
function McpTab() {
  const [selectedKeyId, setSelectedKeyId] = useState<number | null>(null);
  const [mcpSecret, setMcpSecret] = useState('');
  const [configCopied, setConfigCopied] = useState(false);

  const { data: apiKeys } = useQuery({
    queryKey: ['api-keys'],
    queryFn: listApiKeys,
  });

  const activeKeys = apiKeys?.filter(k => k.is_active) || [];
  const selectedKey = activeKeys.find(k => k.id === selectedKeyId) || activeKeys[0] || null;
  const clientId = selectedKey?.client_id || '<your_client_id>';
  const clientSecret = mcpSecret || '<your_client_secret>';

  const configSnippet = JSON.stringify({
    mcpServers: {
      mproof: {
        url: `${BACKEND_URL}/mcp`,
        headers: {
          'X-Client-ID': clientId,
          'X-Client-Secret': clientSecret,
        },
      },
    },
  }, null, 2);

  const copyConfig = async () => {
    await navigator.clipboard.writeText(configSnippet);
    setConfigCopied(true);
    setTimeout(() => setConfigCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <h2 className="text-slate-800 text-xl font-semibold flex items-center gap-2 mb-4">
          <div className="w-9 h-9 rounded-lg bg-cyan-100 border border-cyan-200 flex items-center justify-center">
            <FontAwesomeIcon icon={faPlug} className="text-cyan-600 w-4 h-4" />
          </div>
          MCP Integraties
        </h2>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <h4 className="text-slate-800 font-medium mb-2">Wat is MCP?</h4>
          <p className="text-slate-500 text-sm">
            Model Context Protocol (MCP) is een open standaard voor het verbinden van AI-modellen
            met externe tools en data bronnen. MProof kan als MCP server fungeren, zodat AI-assistenten
            direct toegang hebben tot document analyse functionaliteit.
          </p>
        </div>

        {/* MCP Server Config */}
        <div className="space-y-4">
          <h3 className="text-slate-800 font-semibold">MCP Server Configuratie</h3>

          {/* Key selector */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-slate-500 text-xs block mb-1.5">API Key</label>
              {activeKeys.length > 0 ? (
                <select
                  value={selectedKeyId ?? (activeKeys[0]?.id ?? '')}
                  onChange={e => setSelectedKeyId(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-700 focus:outline-none focus:border-cyan-400"
                >
                  {activeKeys.map(k => (
                    <option key={k.id} value={k.id}>{k.name}</option>
                  ))}
                </select>
              ) : (
                <div className="px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-400">
                  Geen actieve API keys — maak er eerst één aan op het API Keys tabblad
                </div>
              )}
            </div>
            <div>
              <label className="text-slate-500 text-xs block mb-1.5">
                Client Secret
                <span className="ml-1 text-slate-400">(alleen getoond bij aanmaken)</span>
              </label>
              <input
                type="password"
                value={mcpSecret}
                onChange={e => setMcpSecret(e.target.value)}
                placeholder="Plak hier jouw client secret..."
                className="w-full px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-700 placeholder-slate-300 focus:outline-none focus:border-cyan-400"
              />
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 font-mono text-sm border border-slate-200">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400 text-xs">~/.cursor/mcp.json</span>
              <button onClick={copyConfig} className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg border transition-colors ${configCopied ? 'bg-green-50 border-green-200 text-green-600' : 'bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100'}`}>
                <FontAwesomeIcon icon={configCopied ? faCheck : faCopy} className="w-3 h-3" />
                {configCopied ? 'Gekopieerd!' : 'Kopieer'}
              </button>
            </div>
            <pre className="text-green-600 text-xs overflow-x-auto">{configSnippet}</pre>
            <div className="text-slate-500 text-xs mt-3 p-2 bg-slate-50 rounded border border-slate-200 space-y-1.5">
              <p className="flex items-center gap-1.5">
                <FontAwesomeIcon icon={faCheck} className="w-3 h-3 text-green-500 flex-shrink-0" />
                HTTP SSE-server — geen apart proces nodig
              </p>
              <p className="flex items-center gap-1.5">
                <FontAwesomeIcon icon={faGlobe} className="w-3 h-3 text-blue-400 flex-shrink-0" />
                Productie: vervang <code className="bg-white px-1 rounded mx-1">localhost:8000</code> door je server URL
              </p>
              {!mcpSecret && (
                <p className="flex items-center gap-1.5 text-amber-600">
                  <FontAwesomeIcon icon={faLightbulb} className="w-3 h-3 flex-shrink-0" />
                  Vul je client secret in om een kant-en-klare configuratie te kopiëren
                </p>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
            <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
              <h4 className="text-slate-800 font-medium mb-2 flex items-center gap-2">
                <div className="w-6 h-6 rounded-lg bg-purple-100 border border-purple-200 flex items-center justify-center">
                  <FontAwesomeIcon icon={faFileAlt} className="text-purple-400 w-3 h-3" />
                </div>
                Beschikbare Tools (13)
              </h4>
              <ul className="space-y-1.5 text-sm text-slate-500 max-h-96 overflow-y-auto pr-2 scrollbar-thin">
                <li className="flex items-center gap-2">
                  <span className="text-green-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">list_documents</code>
                  - Lijst documenten met filters
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">get_document</code>
                  - Haal document details op
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">analyze_document</code>
                  - Trigger document analyse
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">list_references</code>
                  - Zoek references (personen, bedrijven)
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">get_document_text</code>
                  - Haal geëxtraheerde tekst op
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">get_document_metadata</code>
                  - Haal metadata op
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-red-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">get_fraud_analysis</code>
                  - Volledige fraude analyse
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-green-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">search_documents</code>
                  - Zoek op tekst, type of risicoscore
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-blue-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">train_classifier</code>
                  - Train Naive Bayes model
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-blue-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">train_bert_classifier</code>
                  - Train BERT embeddings model
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-blue-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">get_classifier_status</code>
                  - Naive Bayes status
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-blue-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">get_bert_classifier_status</code>
                  - BERT model status
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-red-600">•</span>
                  <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">list_high_risk_documents</code>
                  - Lijst hoge risico documenten
                </li>
              </ul>
            </div>

          </div>
        </div>

        {/* Example Prompts */}
        <div className="mt-6">
          <h3 className="text-slate-800 font-semibold mb-4 flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-purple-100 border border-purple-200 flex items-center justify-center">
              <FontAwesomeIcon icon={faCommentDots} className="text-purple-400 w-4 h-4" />
            </div>
            Voorbeeld Prompts
          </h3>
          <p className="text-slate-500 text-sm mb-4">
            Kopieer deze prompts naar je AI-assistent (bijv. Claude, Cursor) om MProof te gebruiken:
          </p>
          
          <div className="space-y-3">
            {/* 1. list_documents */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faFileAlt} className="w-3 h-3" />
                  Lijst documenten
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Toon alle documenten voor reference ID 5 met status 'done', maximaal 20 resultaten."
              </code>
            </div>

            {/* 2. get_document */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faFileAlt} className="w-3 h-3" />
                  Document details ophalen
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Haal alle details op van document ID 42, inclusief classificatie, metadata en risicoscores."
              </code>
            </div>

            {/* 3. analyze_document */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faRefresh} className="w-3 h-3" />
                  Document opnieuw analyseren
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Trigger een nieuwe analyse voor document ID 15 om de classificatie en metadata te updaten."
              </code>
            </div>

            {/* 4. list_references */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faDatabase} className="w-3 h-3" />
                  References zoeken
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Zoek alle references met context 'company' die 'XYZ' in de naam hebben."
              </code>
            </div>

            {/* 5. get_document_text */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faFileAlt} className="w-3 h-3" />
                  Document tekst ophalen
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Haal de geëxtraheerde tekst op van document ID 42 en zoek naar het woord 'IBAN'."
              </code>
            </div>

            {/* 6. get_document_metadata */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faChartBar} className="w-3 h-3" />
                  Metadata ophalen
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Haal de geëxtraheerde metadata op van document ID 42 en toon alle gevonden velden."
              </code>
            </div>

            {/* 7. get_fraud_analysis */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                  Fraude analyse
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Voer een volledige fraude analyse uit op document ID 42. Check PDF metadata, image forensics en tekst anomalieën."
              </code>
            </div>

            {/* 8. search_documents */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faSearch} className="w-3 h-3" />
                  Documenten zoeken
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Zoek alle documenten van type 'bankafschrift' met risicoscore tussen 50-75% voor reference ID 3."
              </code>
            </div>

            {/* 9. train_classifier */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faGraduationCap} className="w-3 h-3" />
                  Naive Bayes trainen
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Train het Naive Bayes classificatie model voor 'backoffice' met de nieuwste training data."
              </code>
            </div>

            {/* 10. train_bert_classifier */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faBrain} className="w-3 h-3" />
                  BERT trainen
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Train het BERT embeddings model voor 'backoffice' met threshold 0.75 voor betere precisie."
              </code>
            </div>

            {/* 11. get_classifier_status */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faRobot} className="w-3 h-3" />
                  Naive Bayes status
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Bekijk de status van het Naive Bayes model voor 'backoffice': hoeveel document types zijn getraind?"
              </code>
            </div>

            {/* 12. get_bert_classifier_status */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faBrain} className="w-3 h-3" />
                  BERT status
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Bekijk de status van het BERT model: wanneer is het laatste getraind en hoeveel document types zijn er?"
              </code>
            </div>

            {/* 13. list_high_risk_documents */}
            <div className="bg-white rounded-lg p-4 border border-slate-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-purple-700 text-xs font-medium flex items-center gap-1.5">
                  <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3" />
                  Hoge risico documenten
                </span>
              </div>
              <code className="text-sm text-green-600 block">
                "Toon alle documenten met CRITICAL risk level en risicoscore &gt; 75% voor reference ID 5."
              </code>
            </div>
          </div>
        </div>

        {/* Coming Soon */}
        <div className="mt-6 p-4 bg-purple-50 border border-purple-200 rounded-lg">
          <h4 className="text-slate-800 font-medium mb-2 flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-purple-100 border border-purple-200 flex items-center justify-center">
              <FontAwesomeIcon icon={faRocket} className="w-3 h-3 text-purple-400" />
            </div>
            Binnenkort
          </h4>
          <ul className="text-slate-500 text-sm space-y-1">
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

export default function SettingsPage() {
  return (
    <Suspense>
      <SettingsPageInner />
    </Suspense>
  );
}
