'use client';

import { useState, useMemo, useEffect, Suspense } from 'react';
import { useQuery, useQueries, useMutation, useQueryClient } from '@tanstack/react-query';
import { useSearchParams } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPlus, faEdit, faTrash, faSave, faTimes, faCog, faExclamationTriangle,
  faChevronRight, faFileAlt, faListUl, faCode, faInfoCircle, faCheck, faRobot, faGraduationCap, faFolder, faShieldAlt, faSpinner
} from '@fortawesome/free-solid-svg-icons';
import { SignalPolicyEditor } from '@/components/SignalPolicyEditor';
import {
  listDocumentTypes,
  createDocumentType,
  updateDocumentType,
  deleteDocumentType,
  checkDocumentTypeNameUnique,
  checkDocumentTypeSlugUnique,
  createDocumentTypeField,
  updateDocumentTypeField,
  deleteDocumentTypeField,
  DocumentType,
  DocumentTypeField,
  getTrainingDetails,
  TrainingDetails,
  getAvailableModels,
  generateDocumentTypePrefill,
  ValidationRule,
} from '@/lib/api';
import { useModel } from '@/context/ModelContext';

const fieldTypeLabels: Record<string, string> = {
  string: 'Tekst',
  number: 'Nummer',
  date: 'Datum',
  money: 'Bedrag',
  currency: 'Valuta',
  iban: 'IBAN',
  enum: 'Keuze',
};

const fieldTypeColors: Record<string, string> = {
  string: 'bg-blue-100 text-blue-700',
  number: 'bg-purple-100 text-purple-700',
  date: 'bg-amber-100 text-amber-700',
  money: 'bg-emerald-100 text-emerald-700',
  currency: 'bg-teal-100 text-teal-700',
  iban: 'bg-cyan-100 text-cyan-700',
  enum: 'bg-pink-100 text-pink-700',
};

function DocumentTypesAdminContent() {
  const [selectedType, setSelectedType] = useState<DocumentType | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingType, setEditingType] = useState<DocumentType | null>(null);
  const [prefillData, setPrefillData] = useState<Partial<DocumentType> | null>(null);
  const [showFieldForm, setShowFieldForm] = useState(false);
  const [editingField, setEditingField] = useState<DocumentTypeField | null>(null);
  const [showPolicyEditor, setShowPolicyEditor] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [typeToDelete, setTypeToDelete] = useState<DocumentType | null>(null);
  const queryClient = useQueryClient();
  const searchParams = useSearchParams();
  
  // Get selected model from global context
  const { selectedModel } = useModel();
  
  const [isGeneratingLLM, setIsGeneratingLLM] = useState(false);
  
  // Check for prefill query parameters
  useEffect(() => {
    const create = searchParams.get('create');
    if (create === 'true') {
      const name = searchParams.get('name');
      const slug = searchParams.get('slug');
      const classification_hints = searchParams.get('classification_hints');
      const description = searchParams.get('description');
      const extraction_prompt_preamble = searchParams.get('extraction_prompt_preamble');
      const generateLLM = searchParams.get('generate_llm') === 'true';
      
      if (name && slug) {
        setPrefillData({
          name: name,
          slug: slug,
          classification_hints: classification_hints || undefined,
          description: description || undefined,
          extraction_prompt_preamble: extraction_prompt_preamble || undefined,
        });
        setShowCreateForm(true);
        
        // Generate LLM prefill if requested
        if (generateLLM && !description && !extraction_prompt_preamble) {
          setIsGeneratingLLM(true);
          const generatePrefill = async () => {
            try {
              const tokens = classification_hints?.split('\n').map(line => line.replace('kw:', '').trim()).filter(Boolean).join(', ') || '';
              const data = await generateDocumentTypePrefill({
                name: name,
                keywords: tokens,
              });
              
              setPrefillData(prev => ({
                ...prev,
                description: data.description || prev?.description,
                extraction_prompt_preamble: data.extraction_prompt_preamble || prev?.extraction_prompt_preamble,
              }));
            } catch (error) {
              console.warn('Failed to generate LLM prefill:', error);
            } finally {
              setIsGeneratingLLM(false);
            }
          };
          
          generatePrefill();
        }
        
        // Clean up URL
        window.history.replaceState({}, '', '/document-types');
      }
    }
  }, [searchParams]);

  const { data: documentTypes, isLoading } = useQuery({
    queryKey: ['document-types'],
    queryFn: () => listDocumentTypes(),
  });

  const { data: availableModels } = useQuery({
    queryKey: ['available-models'],
    queryFn: () => getAvailableModels(),
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

  // Get training details for selected model, or aggregate for "Standaard"
  const { data: trainingDetails } = useQuery({
    queryKey: ['training-details', selectedModel],
    queryFn: () => getTrainingDetails(selectedModel),
    enabled: !!selectedModel, // Only fetch when a specific model is selected
  });

  // Aggregate training details when "Standaard" is selected
  const aggregatedTrainingDetails = useMemo(() => {
    if (selectedModel || !availableModels?.models || availableModels.models.length === 0) {
      return null;
    }

    // Check if at least one model exists
    const hasAnyModel = allModelTrainingQueries.some(query => query.data?.model_exists);
    
    // Aggregate training files and tokens across all models
    const allTrainingFiles: Record<string, Array<{ path: string; sha256: string; updated_at: string; }>> = {};
    const allTokens: Record<string, Array<{ token: string; count: number }>> = {};
    const allDocCounts: Record<string, number> = {};
    let latestModel: any = null;

    allModelTrainingQueries.forEach((query, index) => {
      const details = query.data;
      if (details?.model_exists && details.model) {
        // Use the most recently updated model as the "main" model for stats
        if (!latestModel || (details.model.updated_at && details.model.updated_at > latestModel.updated_at)) {
          latestModel = details.model;
        }

        // Aggregate training files
        if (details.training_files_by_label) {
          Object.entries(details.training_files_by_label).forEach(([label, files]) => {
            if (!allTrainingFiles[label]) {
              allTrainingFiles[label] = [];
            }
            // Merge files, avoiding duplicates
            const existingPaths = new Set(allTrainingFiles[label].map(f => f.path));
            files.forEach((file: any) => {
              if (!existingPaths.has(file.path)) {
                allTrainingFiles[label].push(file);
              }
            });
          });
        }

        // Aggregate tokens
        if (details.important_tokens_by_label) {
          Object.entries(details.important_tokens_by_label).forEach(([label, tokens]) => {
            if (!allTokens[label]) {
              allTokens[label] = [];
            }
            // Merge tokens, summing counts
            const tokenMap = new Map(allTokens[label].map((t: any) => [t.token, t.count]));
            tokens.forEach((tokenInfo: any) => {
              const existing = tokenMap.get(tokenInfo.token) || 0;
              tokenMap.set(tokenInfo.token, existing + tokenInfo.count);
            });
            allTokens[label] = Array.from(tokenMap.entries())
              .map(([token, count]) => ({ token, count }))
              .sort((a, b) => b.count - a.count)
              .slice(0, 50); // Keep top 50
          });
        }

        // Aggregate doc counts
        if (details.model.class_doc_counts) {
          Object.entries(details.model.class_doc_counts).forEach(([label, count]) => {
            allDocCounts[label] = (allDocCounts[label] || 0) + (count as number);
          });
        }
      }
    });

    return {
      model_exists: hasAnyModel,
      model: latestModel ? {
        ...latestModel,
        class_doc_counts: allDocCounts,
      } : null,
      training_files_by_label: allTrainingFiles,
      important_tokens_by_label: allTokens,
    };
  }, [selectedModel, availableModels, allModelTrainingQueries]);

  // Use aggregated details when "Standaard" is selected, otherwise use single model details
  const effectiveTrainingDetails = selectedModel ? trainingDetails : aggregatedTrainingDetails;

  // Get the selected model's details
  const selectedModelDetails = useMemo(() => {
    if (!selectedModel || !availableModels?.models) return null;
    return availableModels.models.find(m => m.name === selectedModel);
  }, [selectedModel, availableModels]);

  // Track which document types have training data in the selected model
  const modelDocTypeSlugs = useMemo(() => {
    if (!selectedModel || !selectedModelDetails) return new Set<string>();
    return new Set(selectedModelDetails.document_types?.map(dt => dt.slug) || []);
  }, [selectedModel, selectedModelDetails]);

  // Always show all configured document types. The selected model only affects
  // the training metadata shown for each type, not whether the type exists.
  const filteredDocumentTypes = useMemo(() => {
    if (!documentTypes) return [];
    return documentTypes;
  }, [documentTypes]);

  const createMutation = useMutation({
    mutationFn: createDocumentType,
    onSuccess: (newType) => {
      queryClient.invalidateQueries({ queryKey: ['document-types'] });
      setShowCreateForm(false);
      setSelectedType(newType);
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ slug, data }: { slug: string; data: any }) =>
      updateDocumentType(slug, data),
    onSuccess: (updatedType) => {
      queryClient.invalidateQueries({ queryKey: ['document-types'] });
      setEditingType(null);
      setSelectedType(updatedType);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteDocumentType,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document-types'] });
      setSelectedType(null);
    },
  });

  const createFieldMutation = useMutation({
    mutationFn: ({ slug, data }: { slug: string; data: any }) =>
      createDocumentTypeField(slug, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document-types'] });
      setShowFieldForm(false);
    },
  });

  const updateFieldMutation = useMutation({
    mutationFn: ({ slug, fieldId, data }: { slug: string; fieldId: number; data: any }) =>
      updateDocumentTypeField(slug, fieldId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document-types'] });
      setEditingField(null);
      setShowFieldForm(false);
    },
  });

  const deleteFieldMutation = useMutation({
    mutationFn: ({ slug, fieldId }: { slug: string; fieldId: number }) => 
      deleteDocumentTypeField(slug, fieldId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document-types'] });
    },
  });

  // Keep selectedType in sync with updated data
  const currentType = selectedType
    ? documentTypes?.find(t => t.slug === selectedType.slug) || selectedType
    : null;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <FontAwesomeIcon icon={faCog} className="text-slate-400 text-4xl animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
        <div>
          <h1 className="text-slate-800 text-xl sm:text-2xl lg:text-3xl font-bold">Document Types</h1>
          <p className="text-slate-400 mt-1 text-sm sm:text-base max-w-xl leading-relaxed">
            Definieer welke documentsoorten het systeem herkent en welke gegevens eruit worden gehaald.
            Elk type heeft eigen velden (bijv. IBAN, datum, bedrag) die automatisch worden geëxtraheerd.
          </p>
        </div>
        <button
          onClick={() => {
            setShowCreateForm(true);
            setSelectedType(null);
            setEditingType(null);
            setPrefillData(null);
          }}
          className="flex items-center space-x-2 px-3 sm:px-4 py-2 sm:py-2.5 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-lg sm:rounded-xl hover:from-[#1ab8b8] hover:to-[#e8a8d8] transition-all shadow-lg shadow-[#FFC1F3]/20 cursor-pointer text-sm sm:text-base self-start sm:self-auto whitespace-nowrap"
        >
          <FontAwesomeIcon icon={faPlus} className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
          <span className="font-medium">Nieuw Type</span>
        </button>
      </div>

      {/* Selected Model Info */}
      {selectedModel && selectedModelDetails && (
        <div className="glass-card p-3 sm:p-4 bg-purple-50 border border-purple-200">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <FontAwesomeIcon icon={faRobot} className="w-4 h-4 text-purple-400" />
              <span className="text-slate-800 font-medium text-sm">Model: {selectedModel}</span>
              {selectedModelDetails.is_trained && (
                <span className="text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded">trained</span>
              )}
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <FontAwesomeIcon icon={faFolder} className="w-3 h-3" />
              <span className="font-mono truncate max-w-[200px] sm:max-w-[400px]" title={selectedModelDetails.path}>
                {selectedModelDetails.path ? (() => {
                  // Convert absolute path to relative (e.g., /var/www/.../data/backoffice -> data/backoffice)
                  const dataIndex = selectedModelDetails.path.indexOf('/data/');
                  return dataIndex >= 0 ? selectedModelDetails.path.substring(dataIndex + 1) : selectedModelDetails.path;
                })() : 'N/A'}
              </span>
            </div>
          </div>
          <div className="mt-2 text-xs text-purple-600">
            {selectedModelDetails.document_types?.length || 0} getraind van {documentTypes?.length || 0} document types · {selectedModelDetails.total_files || 0} training bestanden
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 sm:gap-4 lg:gap-6">
        {/* Left Panel - Document Types List */}
        <div className="lg:col-span-1">
          <div className="glass-card p-4 space-y-2">
            <h2 className="text-slate-800 font-semibold px-2 pb-2 border-b border-slate-200">
              Document Types ({filteredDocumentTypes?.length || 0})
              {selectedModel && selectedModelDetails && documentTypes && (
                <span className="text-slate-400 text-xs font-normal ml-2">
                  ({selectedModelDetails.document_types?.length || 0} met trainingdata)
                </span>
              )}
            </h2>
            
            {filteredDocumentTypes?.length === 0 && (
              <div className="text-center py-8">
                <FontAwesomeIcon icon={faFileAlt} className="text-slate-500 text-3xl mb-2" />
                <p className="text-slate-400 text-sm">
                  {selectedModel ? `Geen document types met training data in model "${selectedModel}"` : 'Geen document types'}
                </p>
              </div>
            )}

            {filteredDocumentTypes?.map((docType) => (
              <button
                key={docType.slug}
                onClick={() => {
                  setSelectedType(docType);
                  setShowCreateForm(false);
                  setEditingType(null);
                  setShowFieldForm(false);
                  setEditingField(null);
                }}
                className={`w-full flex items-center justify-between p-3 rounded-xl transition-all text-left ${
                  currentType?.slug === docType.slug
                    ? 'bg-blue-50 border border-blue-200'
                    : 'hover:bg-slate-50 border border-transparent'
                }`}
              >
                <div className="flex items-center space-x-3 min-w-0">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                    currentType?.slug === docType.slug ? 'bg-blue-100' : 'bg-slate-100'
                  }`}>
                    <FontAwesomeIcon
                      icon={faFileAlt}
                      className={currentType?.slug === docType.slug ? 'text-blue-600' : 'text-slate-500'}
                    />
                  </div>
                  <div className="min-w-0">
                    <p className="text-slate-800 font-medium truncate">{docType.name}</p>
                    <p className="text-slate-400 text-xs">
                      {docType.fields.length} velden
                      {selectedModel && selectedModelDetails && !modelDocTypeSlugs.has(docType.slug) && (
                        <span className="text-amber-600"> · geen trainingdata</span>
                      )}
                    </p>
                  </div>
                </div>
                <FontAwesomeIcon
                  icon={faChevronRight}
                  className={`w-3 h-3 flex-shrink-0 ${
                    currentType?.slug === docType.slug ? 'text-blue-600' : 'text-slate-500'
                  }`}
                />
              </button>
            ))}
          </div>
        </div>

        {/* Right Panel - Details/Edit */}
        <div className="lg:col-span-2">
          {/* Create Form */}
          {showCreateForm && (
            <div className="glass-card p-4 sm:p-6">
              {isGeneratingLLM && (
                <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg flex items-center gap-3">
                  <FontAwesomeIcon icon={faSpinner} className="w-5 h-5 text-blue-400 animate-spin" />
                  <div>
                    <div className="text-blue-600 font-medium text-sm">LLM wordt geraadpleegd...</div>
                    <div className="text-blue-500 text-xs mt-1">Beschrijving en extractie instructies worden gegenereerd</div>
                  </div>
                </div>
              )}
              <DocumentTypeForm
                initialData={prefillData || undefined}
                isGeneratingLLM={isGeneratingLLM}
                onSubmit={(data) => {
                  createMutation.mutate(data);
                  setPrefillData(null); // Clear prefill after submission
                }}
                onCancel={() => {
                  setShowCreateForm(false);
                  setPrefillData(null); // Clear prefill on cancel
                  setIsGeneratingLLM(false);
                }}
                isLoading={createMutation.isPending}
              />
            </div>
          )}

          {/* Type Details */}
          {currentType && !showCreateForm && (
            <div className="space-y-4 sm:space-y-6">
              {/* Type Header Card */}
              <div className="glass-card p-4 sm:p-6">
                {editingType?.slug === currentType.slug ? (
                  <DocumentTypeForm
                    initialData={currentType}
                    onSubmit={(data) => updateMutation.mutate({ slug: currentType.slug, data })}
                    onCancel={() => setEditingType(null)}
                    isLoading={updateMutation.isPending}
                  />
                ) : (
                  <div>
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h2 className="text-slate-800 text-2xl font-bold">{currentType.name}</h2>
                        <p className="text-slate-400 text-sm mt-1">
                          <code className="bg-slate-100 px-2 py-0.5 rounded">{currentType.slug}</code>
                        </p>
                      </div>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => setEditingType(currentType)}
                          className="flex items-center space-x-2 px-3 py-2 bg-slate-100 text-slate-800 rounded-lg hover:bg-slate-200 transition-colors cursor-pointer"
                        >
                          <FontAwesomeIcon icon={faEdit} className="w-4 h-4" />
                          <span className="text-sm">Bewerken</span>
                        </button>
                        <button
                          onClick={() => {
                            setTypeToDelete(currentType);
                            setShowDeleteModal(true);
                          }}
                          className="flex items-center space-x-2 px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors cursor-pointer"
                        >
                          <FontAwesomeIcon icon={faTrash} className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {currentType.description && (
                      <p className="text-slate-500 mb-4">{currentType.description}</p>
                    )}

                    {/* Hints & Preamble Accordions */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {currentType.classification_hints && (
                        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                          <h4 className="text-slate-600 text-sm font-medium mb-2 flex items-center">
                            <FontAwesomeIcon icon={faCode} className="w-3 h-3 mr-2 text-amber-400" />
                            Classificatie Hints
                          </h4>
                          <pre className="text-slate-500 text-xs whitespace-pre-wrap font-mono max-h-32 overflow-y-auto">
                            {currentType.classification_hints}
                          </pre>
                        </div>
                      )}
                      {currentType.extraction_prompt_preamble && (
                        <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                          <h4 className="text-slate-600 text-sm font-medium mb-2 flex items-center">
                            <FontAwesomeIcon icon={faInfoCircle} className="w-3 h-3 mr-2 text-cyan-400" />
                            Extractie Preamble
                          </h4>
                          <pre className="text-slate-500 text-xs whitespace-pre-wrap font-mono max-h-32 overflow-y-auto">
                            {currentType.extraction_prompt_preamble}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Fields Card */}
              {!editingType && (
                <div className="glass-card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-slate-800 font-semibold flex items-center">
                      <FontAwesomeIcon icon={faListUl} className="w-4 h-4 mr-2 text-purple-400" />
                      Velden ({currentType.fields.length})
                    </h3>
                    <button
                      onClick={() => {
                        setShowFieldForm(true);
                        setEditingField(null);
                      }}
                      className="flex items-center space-x-2 px-3 py-1.5 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-lg hover:from-[#1ab8b8] hover:to-[#e8a8d8] transition-all text-sm cursor-pointer shadow-sm"
                    >
                      <FontAwesomeIcon icon={faPlus} className="w-3 h-3" />
                      <span>Veld Toevoegen</span>
                    </button>
                  </div>

                  {/* Field Form */}
                  {showFieldForm && (
                    <div className="mb-4">
                      <FieldForm
                        initialData={editingField}
                        onSubmit={(data) => {
                          if (editingField) {
                            updateFieldMutation.mutate({
                              slug: currentType.slug,
                              fieldId: editingField.id,
                              data
                            });
                          } else {
                            createFieldMutation.mutate({
                              slug: currentType.slug,
                              data
                            });
                          }
                        }}
                        onCancel={() => {
                          setShowFieldForm(false);
                          setEditingField(null);
                        }}
                        isLoading={createFieldMutation.isPending || updateFieldMutation.isPending}
                      />
                    </div>
                  )}

                  {/* Fields List */}
                  {currentType.fields.length === 0 && !showFieldForm && (
                    <div className="text-center py-8 bg-slate-50 rounded-xl border-2 border-dashed border-slate-200">
                      <FontAwesomeIcon icon={faListUl} className="text-slate-500 text-3xl mb-2" />
                      <p className="text-slate-400">Geen velden gedefinieerd</p>
                      <p className="text-slate-500 text-sm mt-1">Klik op "Veld Toevoegen" om te starten</p>
                    </div>
                  )}

                  <div className="space-y-1">
                    {currentType.fields
                      .filter(field => !editingField || field.id !== editingField.id)
                      .map((field) => (
                      <div
                        key={field.id}
                        className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0 px-2 sm:px-3 py-2 bg-slate-50 rounded-lg border border-slate-200 hover:border-slate-300 transition-colors group"
                      >
                        <div className="flex items-center space-x-2 sm:space-x-3 min-w-0 flex-1">
                          <div className={`px-1.5 sm:px-2 py-0.5 rounded text-[10px] sm:text-xs font-medium shrink-0 ${fieldTypeColors[field.field_type] || 'bg-gray-100 text-gray-700'}`}>
                            {fieldTypeLabels[field.field_type] || field.field_type}
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center space-x-1.5 sm:space-x-2 flex-wrap">
                              <span className="text-slate-800 font-medium text-xs sm:text-sm">{field.label}</span>
                              <code className="text-slate-400 text-[10px] sm:text-xs bg-slate-100 px-1 py-0.5 rounded">
                                {field.key}
                              </code>
                              {field.regex && (
                                <code className="text-amber-700 text-[10px] sm:text-xs bg-amber-50 px-1 py-0.5 rounded truncate max-w-[100px] sm:max-w-[150px]" title={field.regex}>
                                  {field.regex}
                                </code>
                              )}
                              {field.enum_values && field.enum_values.length > 0 && (
                                <span className="text-slate-400 text-[10px] sm:text-xs truncate hidden sm:inline">
                                  [{field.enum_values.join(', ')}]
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center justify-between sm:justify-end space-x-2 shrink-0">
                          {/* Required Toggle */}
                          <div className="flex items-center space-x-1.5 sm:space-x-2">
                            <span className="text-slate-400 text-[10px] sm:text-xs hidden sm:inline">Verplicht</span>
                            <button
                              type="button"
                              onClick={() => {
                                updateFieldMutation.mutate({
                                  slug: currentType.slug,
                                  fieldId: field.id,
                                  data: { required: !field.required }
                                });
                              }}
                              className={`w-7 h-4 sm:w-8 sm:h-5 rounded-full transition-colors flex items-center cursor-pointer ${field.required ? 'bg-emerald-500' : 'bg-slate-200'}`}
                              title={field.required ? 'Verplicht' : 'Optioneel'}
                            >
                              <div className={`w-2.5 h-2.5 sm:w-3 sm:h-3 bg-white rounded-full shadow-md transform transition-transform ${field.required ? 'translate-x-3.5 sm:translate-x-4' : 'translate-x-0.5 sm:translate-x-1'}`} />
                            </button>
                          </div>
                          {/* Edit/Delete Buttons */}
                          <div className="flex space-x-0.5 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setEditingField(field);
                                setShowFieldForm(true);
                              }}
                              className="p-1.5 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded"
                              title="Bewerken"
                            >
                              <FontAwesomeIcon icon={faEdit} className="w-3 h-3" />
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                if (confirm(`Veld "${field.label}" verwijderen?`)) {
                                  deleteFieldMutation.mutate({ slug: currentType.slug, fieldId: field.id });
                                }
                              }}
                              className="p-1.5 text-red-400 hover:text-red-600 hover:bg-red-500/10 rounded cursor-pointer"
                              title="Verwijderen"
                            >
                              <FontAwesomeIcon icon={faTrash} className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Classification Policy Card */}
              {!editingType && currentType && (
                <div className="glass-card p-6">
                  {showPolicyEditor ? (
                    <SignalPolicyEditor
                      slug={currentType.slug}
                      onClose={() => setShowPolicyEditor(false)}
                    />
                  ) : (
                    <div>
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-slate-800 font-semibold flex items-center">
                          <FontAwesomeIcon icon={faShieldAlt} className="w-4 h-4 mr-2 text-indigo-400" />
                          Classificatiebeleid
                        </h3>
                        <button
                          onClick={() => setShowPolicyEditor(true)}
                          className="flex items-center space-x-2 px-3 py-1.5 bg-teal-600 text-white rounded-lg hover:bg-teal-500 transition-colors text-sm cursor-pointer"
                        >
                          <FontAwesomeIcon icon={faEdit} className="w-3 h-3" />
                          <span>Configureren</span>
                        </button>
                      </div>
                      {currentType.classification_policy_json ? (
                        <div className="space-y-3">
                          <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg text-sm">
                            <div className="text-indigo-700 font-medium mb-2">Aangepast Beleid Actief</div>

                            {/* Signal-based summary */}
                            <div className="space-y-2 mb-3">
                              {(currentType.classification_policy_json as any).requirements?.length > 0 && (
                                <div className="text-xs text-slate-500">
                                  <span className="text-teal-700">Vereisten:</span>{' '}
                                  {(currentType.classification_policy_json as any).requirements
                                    .map((r: any) => `${r.signal} ${r.op} ${r.value}`)
                                    .join(', ')}
                                </div>
                              )}
                              {(currentType.classification_policy_json as any).exclusions?.length > 0 && (
                                <div className="text-xs text-slate-500">
                                  <span className="text-red-600">Uitsluitingen:</span>{' '}
                                  {(currentType.classification_policy_json as any).exclusions
                                    .map((e: any) => `${e.signal} ${e.op} ${e.value}`)
                                    .join(', ')}
                                </div>
                              )}
                            </div>

                            <div className="grid grid-cols-2 gap-2 text-xs text-slate-500">
                              <div>
                                Getraind Model: {currentType.classification_policy_json.acceptance?.trained_model?.enabled !== false ? (
                                  <span className="text-green-600">Aan</span>
                                ) : (
                                  <span className="text-red-600">Uit</span>
                                )}
                              </div>
                              <div>
                                Deterministisch: {currentType.classification_policy_json.acceptance?.deterministic?.enabled !== false ? (
                                  <span className="text-green-600">Aan</span>
                                ) : (
                                  <span className="text-red-600">Uit</span>
                                )}
                              </div>
                              <div>
                                LLM: {currentType.classification_policy_json.acceptance?.llm?.enabled !== false ? (
                                  <span className="text-green-600">Aan</span>
                                ) : (
                                  <span className="text-red-600">Uit</span>
                                )}
                              </div>
                              <div>
                                Min. Zekerheid: {((currentType.classification_policy_json.acceptance?.trained_model?.min_confidence || 0.85) * 100).toFixed(0)}%
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-4 bg-slate-50 rounded-xl border border-dashed border-slate-200">
                          <p className="text-slate-400 text-sm">Standaard beleid wordt gebruikt</p>
                          <p className="text-slate-500 text-xs mt-1">Klik op Configureren om aangepaste regels in te stellen</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Training Details Card */}
              {!editingType && currentType && effectiveTrainingDetails && (
                <div className="glass-card p-6">
                  <h3 className="text-slate-800 font-semibold flex items-center mb-4">
                    <FontAwesomeIcon icon={faRobot} className="w-4 h-4 mr-2 text-purple-400" />
                    Classificatie Training
                  </h3>

                  {/* Deterministic (Classification Hints) - Now fallback */}
                  {currentType.classification_hints && (
                    <div className="mb-4 p-4 bg-slate-50 border border-slate-200 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <FontAwesomeIcon icon={faCode} className="w-4 h-4 text-gray-400" />
                        <h4 className="text-slate-800 font-medium text-sm">Fallback: Keywords/Regex</h4>
                        <span className="text-xs bg-gray-100 px-2 py-0.5 rounded text-gray-600">Secundair</span>
                      </div>
                      <p className="text-slate-500 text-xs mb-2">
                        Deze hints worden gebruikt als fallback wanneer getrainde modellen (Naive Bayes/BERT) geen match vinden of lage confidence hebben. <strong>Let op:</strong> Als alle <code>kw:</code> keywords matchen (STRONG match), heeft dat wel prioriteit boven getrainde modellen.
                      </p>
                      <pre className="text-slate-400 text-xs whitespace-pre-wrap font-mono bg-slate-50 p-2 rounded max-h-32 overflow-y-auto">
                        {currentType.classification_hints}
                      </pre>
                    </div>
                  )}

                  {/* Trained Model Info */}
                  {effectiveTrainingDetails?.model_exists && effectiveTrainingDetails?.model && (
                    <div className="space-y-4">
                      {/* Model Stats */}
                      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <div className="flex items-center gap-2 mb-3">
                          <FontAwesomeIcon icon={faGraduationCap} className="w-4 h-4 text-blue-400" />
                          <h4 className="text-slate-800 font-medium text-sm">Getraind Model (Naive Bayes)</h4>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div>
                            <span className="text-slate-500">Getraind op:</span>
                            <span className="text-slate-800 ml-2">{new Date(effectiveTrainingDetails.model.updated_at).toLocaleDateString('nl-NL')}</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Threshold:</span>
                            <span className="text-slate-800 ml-2">{(effectiveTrainingDetails.model.threshold * 100).toFixed(0)}%</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Vocabulaire:</span>
                            <span className="text-slate-800 ml-2">{effectiveTrainingDetails.model.vocab_size.toLocaleString()} tokens</span>
                          </div>
                          <div>
                            <span className="text-slate-500">Labels:</span>
                            <span className="text-slate-800 ml-2">{effectiveTrainingDetails.model.labels.length}</span>
                          </div>
                        </div>
                      </div>

                      {/* Training Data for this Document Type */}
                      {effectiveTrainingDetails.training_files_by_label[currentType.slug] && (
                        <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                          <h4 className="text-slate-800 font-medium text-sm mb-2 flex items-center gap-2">
                            <FontAwesomeIcon icon={faFileAlt} className="w-3 h-3 text-green-400" />
                            Training Data
                          </h4>
                          <div className="text-xs text-slate-500 mb-2">
                            <span className="font-semibold">{effectiveTrainingDetails.training_files_by_label[currentType.slug].length}</span> documenten gebruikt voor training
                            {effectiveTrainingDetails.model?.class_doc_counts[currentType.slug] && (
                              <span className="ml-2">
                                ({effectiveTrainingDetails.model.class_doc_counts[currentType.slug]} documenten in model)
                              </span>
                            )}
                          </div>
                          <div className="max-h-32 overflow-y-auto space-y-1">
                            {effectiveTrainingDetails.training_files_by_label[currentType.slug].slice(0, 10).map((file, idx) => (
                              <div key={idx} className="text-xs text-slate-400 font-mono bg-slate-50 p-1.5 rounded truncate">
                                {file.path.split('/').pop()}
                              </div>
                            ))}
                            {effectiveTrainingDetails.training_files_by_label[currentType.slug].length > 10 && (
                              <div className="text-xs text-slate-400 italic">
                                +{effectiveTrainingDetails.training_files_by_label[currentType.slug].length - 10} meer...
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Important Tokens */}
                      {effectiveTrainingDetails.important_tokens_by_label[currentType.slug] && (
                        <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                          <h4 className="text-slate-800 font-medium text-sm mb-2 flex items-center gap-2">
                            <FontAwesomeIcon icon={faCode} className="w-3 h-3 text-purple-400" />
                            Belangrijkste Tokens (Top 20)
                          </h4>
                          <p className="text-slate-500 text-xs mb-2">
                            Deze woorden/tokens zijn het meest kenmerkend voor dit document type volgens het getrainde model.
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {effectiveTrainingDetails.important_tokens_by_label[currentType.slug].slice(0, 20).map((tokenInfo, idx) => (
                              <div
                                key={idx}
                                className="px-2 py-1 bg-purple-100 border border-purple-200 rounded text-xs text-purple-700 font-mono"
                                title={`Aantal: ${tokenInfo.count}`}
                              >
                                {tokenInfo.token} <span className="text-purple-500">({tokenInfo.count})</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* No Training Data */}
                      {!effectiveTrainingDetails.training_files_by_label[currentType.slug] && (
                        <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg text-center">
                          <p className="text-slate-400 text-sm">Geen training data beschikbaar voor dit document type</p>
                          <p className="text-slate-400 text-xs mt-1">Plaats documenten in data/{currentType.slug}/ en train het model</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* No Model */}
                  {!effectiveTrainingDetails?.model_exists && (
                    <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg text-center">
                      <p className="text-slate-400 text-sm">Geen getraind model beschikbaar</p>
                      <p className="text-slate-400 text-xs mt-1">Train het model om classificatie details te zien</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Empty State */}
          {!currentType && !showCreateForm && (
            <div className="glass-card p-12 text-center">
              <FontAwesomeIcon icon={faFileAlt} className="text-slate-500 text-5xl mb-4" />
              <h3 className="text-slate-800 text-lg font-medium mb-2">Selecteer een document type</h3>
              <p className="text-slate-400 max-w-md mx-auto">
                Klik op een document type in de lijst om de details te bekijken en velden te beheren.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && typeToDelete && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass-card p-6 max-w-md w-full space-y-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
                <FontAwesomeIcon icon={faExclamationTriangle} className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-slate-800 text-lg font-semibold">Document Type Verwijderen</h3>
                <p className="text-slate-500 text-sm">Deze actie kan niet ongedaan worden gemaakt</p>
              </div>
            </div>
            
            <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
              <p className="text-slate-600 text-sm mb-2">Weet je zeker dat je het volgende document type wilt verwijderen?</p>
              <div className="space-y-1">
                <p className="text-slate-800 font-medium">{typeToDelete.name}</p>
                <p className="text-slate-400 text-xs font-mono">{typeToDelete.slug}</p>
                <p className="text-slate-400 text-xs mt-2">
                  {typeToDelete.fields.length} veld{typeToDelete.fields.length !== 1 ? 'en' : ''} zullen ook worden verwijderd
                </p>
              </div>
            </div>

            <div className="flex space-x-3 pt-2">
              <button
                onClick={() => {
                  deleteMutation.mutate(typeToDelete.slug);
                  setShowDeleteModal(false);
                  setTypeToDelete(null);
                }}
                disabled={deleteMutation.isPending}
                className="flex-1 flex items-center justify-center space-x-2 px-4 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {deleteMutation.isPending ? (
                  <>
                    <FontAwesomeIcon icon={faSpinner} className="w-4 h-4 animate-spin" />
                    <span>Verwijderen...</span>
                  </>
                ) : (
                  <>
                    <FontAwesomeIcon icon={faTrash} className="w-4 h-4" />
                    <span>Verwijderen</span>
                  </>
                )}
              </button>
              <button
                onClick={() => {
                  setShowDeleteModal(false);
                  setTypeToDelete(null);
                }}
                disabled={deleteMutation.isPending}
                className="flex-1 px-4 py-2.5 bg-slate-100 text-slate-800 rounded-lg hover:bg-slate-200 transition-colors cursor-pointer"
              >
                Annuleren
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function DocumentTypesAdmin() {
  return (
    <Suspense fallback={<div className="text-slate-800">Loading...</div>}>
      <DocumentTypesAdminContent />
    </Suspense>
  );
}

// Document Type Form Component
interface DocumentTypeFormProps {
  initialData?: Partial<DocumentType>;
  onSubmit: (data: any) => void;
  onCancel: () => void;
  isLoading: boolean;
  isGeneratingLLM?: boolean;
}

function DocumentTypeForm({ initialData, onSubmit, onCancel, isLoading, isGeneratingLLM = false }: DocumentTypeFormProps) {
  const [formData, setFormData] = useState({
    name: initialData?.name || '',
    slug: initialData?.slug || '',
    description: initialData?.description || '',
    classification_hints: initialData?.classification_hints || '',
    extraction_prompt_preamble: initialData?.extraction_prompt_preamble || '',
  });
  
  // Update formData when initialData changes (for LLM prefill updates)
  useEffect(() => {
    if (initialData) {
      setFormData(prev => ({
        ...prev,
        description: initialData.description || prev.description,
        extraction_prompt_preamble: initialData.extraction_prompt_preamble || prev.extraction_prompt_preamble,
      }));
    }
  }, [initialData?.description, initialData?.extraction_prompt_preamble]);
  const [nameError, setNameError] = useState<string | null>(null);
  const [slugError, setSlugError] = useState<string | null>(null);

  const generateSlug = (name: string): string => {
    return name
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '');
  };

  const { data: isNameUnique } = useQuery({
    queryKey: ['document-type-name-check', formData.name, initialData?.slug],
    queryFn: () => checkDocumentTypeNameUnique(formData.name, initialData?.slug),
    enabled: formData.name.length > 0,
  });

  const { data: isSlugUnique } = useQuery({
    queryKey: ['document-type-slug-check', formData.slug, initialData?.slug],
    queryFn: () => checkDocumentTypeSlugUnique(formData.slug, initialData?.slug),
    enabled: formData.slug.length > 0,
  });

  const handleNameChange = (name: string) => {
    const newSlug = initialData ? formData.slug : generateSlug(name);
    setFormData(prev => ({ ...prev, name, slug: newSlug }));
    setNameError(null);
  };

  const handleSlugChange = (slug: string) => {
    const cleanSlug = slug.toLowerCase().replace(/[^a-z0-9-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
    setFormData(prev => ({ ...prev, slug: cleanSlug }));
    setSlugError(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isNameUnique === false) {
      setNameError('Deze naam bestaat al');
      return;
    }
    if (isSlugUnique === false) {
      setSlugError('Deze slug bestaat al');
      return;
    }
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
          <FontAwesomeIcon icon={initialData ? faEdit : faPlus} className="text-blue-600" />
        </div>
        <h3 className="text-slate-800 text-xl font-semibold">
          {initialData ? 'Document Type Bewerken' : 'Nieuw Document Type'}
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-slate-600 text-sm font-medium mb-2">Naam *</label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => handleNameChange(e.target.value)}
            className={`w-full px-4 py-2.5 bg-slate-100 border rounded-xl text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-blue-400 focus:border-transparent ${
              nameError ? 'border-red-400' : 'border-slate-300'
            }`}
            placeholder="Bijv. Bankafschrift"
            required
          />
          {nameError && (
            <p className="text-red-600 text-xs mt-1 flex items-center">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 mr-1" />
              {nameError}
            </p>
          )}
        </div>

        <div>
          <label className="block text-slate-600 text-sm font-medium mb-2">
            Slug {initialData ? '' : '(automatisch)'}
          </label>
          <input
            type="text"
            value={formData.slug}
            onChange={(e) => handleSlugChange(e.target.value)}
            readOnly={!initialData}
            className={`w-full px-4 py-2.5 border rounded-xl text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-blue-400 focus:border-transparent ${
              slugError ? 'border-red-400 bg-slate-100' : initialData ? 'bg-slate-100 border-slate-300' : 'bg-slate-50 border-slate-200 cursor-not-allowed'
            }`}
            required
          />
          {slugError && (
            <p className="text-red-600 text-xs mt-1 flex items-center">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 mr-1" />
              {slugError}
            </p>
          )}
        </div>
      </div>

      <div>
        <label className="block text-slate-600 text-sm font-medium mb-2">
          Omschrijving
          <span className="text-slate-400 font-normal ml-2">(korte beschrijving van dit document type)</span>
        </label>
        <div className="relative">
          {isGeneratingLLM && !formData.description && (
            <div className="absolute inset-0 bg-slate-50 rounded-xl flex items-center justify-center z-10">
              <div className="flex items-center gap-2 text-blue-600">
                <FontAwesomeIcon icon={faSpinner} className="w-4 h-4 animate-spin" />
                <span className="text-sm">Wordt gegenereerd...</span>
              </div>
            </div>
          )}
          <textarea
            value={formData.description}
            onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
            rows={2}
            disabled={isGeneratingLLM && !formData.description}
            className="w-full px-4 py-2.5 bg-slate-100 border border-slate-300 rounded-xl text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-blue-400 focus:border-transparent text-sm disabled:opacity-50"
            placeholder="Korte beschrijving van dit document type..."
          />
        </div>
      </div>

      <div>
        <label className="block text-slate-600 text-sm font-medium mb-2">
          Classificatie Hints
          <span className="text-slate-400 font-normal ml-2">(trefwoorden per regel)</span>
        </label>
        <textarea
          value={formData.classification_hints}
          onChange={(e) => setFormData(prev => ({ ...prev, classification_hints: e.target.value }))}
          rows={4}
          className="w-full px-4 py-2.5 bg-slate-100 border border-slate-300 rounded-xl text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-blue-400 focus:border-transparent font-mono text-sm"
          placeholder="kw:bankafschrift&#10;kw:rekeningnummer&#10;re:NL\d{2}[A-Z]{4}\d{10}"
        />
      </div>

      <div>
        <label className="block text-slate-600 text-sm font-medium mb-2">
          Extractie Prompt Preamble
          <span className="text-slate-400 font-normal ml-2">(extra context voor LLM)</span>
        </label>
        <div className="relative">
          {isGeneratingLLM && !formData.extraction_prompt_preamble && (
            <div className="absolute inset-0 bg-slate-50 rounded-xl flex items-center justify-center z-10">
              <div className="flex items-center gap-2 text-blue-600">
                <FontAwesomeIcon icon={faSpinner} className="w-4 h-4 animate-spin" />
                <span className="text-sm">Wordt gegenereerd...</span>
              </div>
            </div>
          )}
          <textarea
            value={formData.extraction_prompt_preamble}
            onChange={(e) => setFormData(prev => ({ ...prev, extraction_prompt_preamble: e.target.value }))}
            rows={3}
            disabled={isGeneratingLLM && !formData.extraction_prompt_preamble}
            className="w-full px-4 py-2.5 bg-slate-100 border border-slate-300 rounded-xl text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-blue-400 focus:border-transparent font-mono text-sm disabled:opacity-50"
            placeholder="Extra instructies voor metadata extractie..."
          />
        </div>
      </div>

      <div className="flex space-x-3 pt-2">
        <button
          type="submit"
          disabled={isLoading}
          className="flex items-center space-x-2 px-5 py-2.5 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-xl hover:from-[#1ab8b8] hover:to-[#e8a8d8] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <FontAwesomeIcon icon={faSave} className="w-4 h-4" />
          <span>{isLoading ? 'Opslaan...' : 'Opslaan'}</span>
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-5 py-2.5 bg-slate-100 text-slate-800 rounded-xl hover:bg-slate-200 transition-colors cursor-pointer"
        >
          Annuleren
        </button>
      </div>
    </form>
  );
}

function AddValidationRule({ onAdd }: { onAdd: (rule: ValidationRule) => void }) {
  const [ruleType, setRuleType] = useState('');
  const [params, setParams] = useState<Record<string, string>>({});

  const paramFields: Record<string, Array<{key: string; label: string; placeholder: string}>> = {
    amount_range: [
      { key: 'min', label: 'Minimum (€)', placeholder: '50000' },
      { key: 'max', label: 'Maximum (€)', placeholder: '5000000' },
    ],
    date_max_age_days: [
      { key: 'max_days', label: 'Max leeftijd (dagen)', placeholder: '180' },
    ],
    cross_field_ratio: [
      { key: 'other_field', label: 'Ander veld (key)', placeholder: 'marktwaarde' },
      { key: 'min_ratio', label: 'Min ratio', placeholder: '0.65' },
      { key: 'max_ratio', label: 'Max ratio', placeholder: '0.90' },
    ],
    date_not_expired: [],
  };

  const handleAdd = () => {
    if (!ruleType) return;

    // Validate required params are filled
    const required = paramFields[ruleType] || [];
    const missing = required.filter(f => !params[f.key]?.trim());
    if (missing.length > 0) return; // don't add if any param is empty

    const typedParams: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(params)) {
      typedParams[k] = isNaN(Number(v)) ? v : Number(v);
    }
    onAdd({ validation_type: ruleType as ValidationRule['validation_type'], params: typedParams });
    setRuleType('');
    setParams({});
  };

  return (
    <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-2">
      <select
        value={ruleType}
        onChange={(e) => { setRuleType(e.target.value); setParams({}); }}
        className="w-full px-2 py-1.5 bg-slate-100 border border-slate-300 rounded text-slate-800 text-xs focus:ring-1 focus:ring-amber-400"
      >
        <option value="">— Regeltype kiezen —</option>
        <option value="amount_range">Bedrag bereik</option>
        <option value="date_max_age_days">Datum max. leeftijd</option>
        <option value="cross_field_ratio">Kruisveld ratio</option>
        <option value="date_not_expired">Datum niet verlopen</option>
      </select>
      {ruleType && paramFields[ruleType]?.map(f => (
        <input
          key={f.key}
          type="text"
          placeholder={`${f.label} (${f.placeholder})`}
          value={params[f.key] || ''}
          onChange={(e) => setParams(prev => ({ ...prev, [f.key]: e.target.value }))}
          className="w-full px-2 py-1.5 bg-slate-100 border border-slate-300 rounded text-slate-800 text-xs placeholder-slate-400"
        />
      ))}
      {ruleType && (
        <button
          type="button"
          onClick={handleAdd}
          className="px-3 py-1 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded text-xs hover:from-[#1ab8b8] hover:to-[#e8a8d8] transition-all"
        >
          Toevoegen
        </button>
      )}
    </div>
  );
}

// Field Form Component
interface FieldFormProps {
  initialData?: DocumentTypeField | null;
  onSubmit: (data: any) => void;
  onCancel: () => void;
  isLoading: boolean;
}

function FieldForm({ initialData, onSubmit, onCancel, isLoading }: FieldFormProps) {
  const [formData, setFormData] = useState({
    key: initialData?.key || '',
    label: initialData?.label || '',
    field_type: initialData?.field_type || 'string',
    required: initialData?.required || false,
    description: initialData?.description || '',
    enum_values: initialData?.enum_values ? initialData.enum_values.join(', ') : '',
    regex: initialData?.regex || '',
    validation_rules: initialData?.validation_rules || [] as ValidationRule[],
  });

  const generateKey = (label: string): string => {
    return label
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, '_');
  };

  const handleLabelChange = (label: string) => {
    const newKey = initialData ? formData.key : generateKey(label);
    setFormData(prev => ({ ...prev, label, key: newKey }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const submitData = {
      ...formData,
      enum_values: formData.field_type === 'enum' && formData.enum_values
        ? formData.enum_values.split(',').map(v => v.trim()).filter(v => v)
        : null,
      regex: formData.regex || null,
      validation_rules: formData.validation_rules,
    };
    onSubmit(submitData);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-gradient-to-br from-emerald-500/10 to-blue-500/10 p-5 rounded-xl border border-emerald-500/20 space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-slate-800 font-medium flex items-center">
          <FontAwesomeIcon icon={initialData ? faEdit : faPlus} className="w-4 h-4 mr-2 text-emerald-400" />
          {initialData ? 'Veld Bewerken' : 'Nieuw Veld'}
        </h4>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-slate-600 text-sm font-medium mb-1.5">Label *</label>
          <input
            type="text"
            value={formData.label}
            onChange={(e) => handleLabelChange(e.target.value)}
            className="w-full px-3 py-2 bg-slate-100 border border-slate-300 rounded-lg text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm"
            placeholder="Bijv. IBAN Nummer"
            required
          />
        </div>

        <div>
          <label className="block text-slate-600 text-sm font-medium mb-1.5">
            Key {initialData ? '' : '(automatisch)'}
          </label>
          <input
            type="text"
            value={formData.key}
            onChange={(e) => setFormData(prev => ({ ...prev, key: e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_') }))}
            readOnly={!initialData}
            className={`w-full px-3 py-2 border rounded-lg text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm font-mono ${
              initialData ? 'bg-slate-100 border-slate-300' : 'bg-slate-50 border-slate-200 cursor-not-allowed'
            }`}
            required
          />
        </div>
      </div>

      <div>
        <label className="block text-slate-600 text-sm font-medium mb-1.5">Type *</label>
        <select
          value={formData.field_type}
          onChange={(e) => setFormData(prev => ({ ...prev, field_type: e.target.value as 'string' | 'number' | 'date' | 'money' | 'currency' | 'iban' | 'enum' }))}
          className="w-full px-3 py-2 bg-slate-100 border border-slate-300 rounded-lg text-slate-800 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm"
        >
          {Object.entries(fieldTypeLabels).map(([value, label]) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
      </div>

      {formData.field_type === 'enum' && (
        <div>
          <label className="block text-slate-600 text-sm font-medium mb-1.5">Opties (komma-gescheiden)</label>
          <input
            type="text"
            value={formData.enum_values}
            onChange={(e) => setFormData(prev => ({ ...prev, enum_values: e.target.value }))}
            placeholder="optie1, optie2, optie3"
            className="w-full px-3 py-2 bg-slate-100 border border-slate-300 rounded-lg text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm"
          />
        </div>
      )}

      <div>
        <label className="block text-slate-600 text-sm font-medium mb-1.5">
          Regex Validatie
          <span className="text-slate-400 font-normal ml-2">(optioneel)</span>
        </label>
        <input
          type="text"
          value={formData.regex}
          onChange={(e) => setFormData(prev => ({ ...prev, regex: e.target.value }))}
          placeholder="^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}.*$"
          className="w-full px-3 py-2 bg-slate-100 border border-slate-300 rounded-lg text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm font-mono"
        />
        <p className="text-slate-400 text-xs mt-1">Reguliere expressie voor waarde validatie</p>
      </div>

      <div>
        <label className="block text-slate-600 text-sm font-medium mb-1.5">Hulptekst</label>
        <input
          type="text"
          value={formData.description}
          onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
          placeholder="Extra uitleg over dit veld..."
          className="w-full px-3 py-2 bg-slate-100 border border-slate-300 rounded-lg text-slate-800 placeholder-slate-400 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm"
        />
      </div>

      <div>
        <label className="block text-slate-600 text-sm font-medium mb-1.5">
          Validatieregels
          <span className="text-slate-400 font-normal ml-2">(optioneel, voor fraude detectie)</span>
        </label>

        {formData.validation_rules.length > 0 && (
          <div className="space-y-2 mb-2">
            {formData.validation_rules.map((rule, idx) => (
              <div key={`${rule.validation_type}-${idx}`} className="flex items-center justify-between p-2 bg-slate-50 rounded-lg border border-slate-200 text-xs">
                <div className="flex items-center gap-2 text-slate-500">
                  <span className="text-amber-700 font-medium">{rule.validation_type}</span>
                  <span>{JSON.stringify(rule.params)}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setFormData(prev => ({ ...prev, validation_rules: prev.validation_rules.filter((_, i) => i !== idx) }))}
                  className="text-red-400 hover:text-red-600 px-1"
                >×</button>
              </div>
            ))}
          </div>
        )}

        <AddValidationRule onAdd={(rule) => setFormData(prev => ({ ...prev, validation_rules: [...prev.validation_rules, rule] }))} />
      </div>

      <div className="flex space-x-2 pt-2">
        <button
          type="submit"
          disabled={isLoading}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-lg hover:from-[#1ab8b8] hover:to-[#e8a8d8] disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <FontAwesomeIcon icon={faCheck} className="w-3 h-3" />
          <span>{isLoading ? 'Opslaan...' : 'Opslaan'}</span>
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 bg-slate-100 text-slate-800 rounded-lg hover:bg-slate-200 transition-colors text-sm cursor-pointer"
        >
          Annuleren
        </button>
      </div>
    </form>
  );
}
