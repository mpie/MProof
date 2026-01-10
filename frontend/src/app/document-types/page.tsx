'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPlus, faEdit, faTrash, faSave, faTimes, faCog, faExclamationTriangle,
  faChevronRight, faFileAlt, faListUl, faCode, faInfoCircle, faCheck
} from '@fortawesome/free-solid-svg-icons';
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
  DocumentTypeField
} from '@/lib/api';

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
  string: 'bg-blue-500/20 text-blue-400',
  number: 'bg-purple-500/20 text-purple-400',
  date: 'bg-amber-500/20 text-amber-400',
  money: 'bg-emerald-500/20 text-emerald-400',
  currency: 'bg-teal-500/20 text-teal-400',
  iban: 'bg-cyan-500/20 text-cyan-400',
  enum: 'bg-pink-500/20 text-pink-400',
};

export default function DocumentTypesAdmin() {
  const [selectedType, setSelectedType] = useState<DocumentType | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingType, setEditingType] = useState<DocumentType | null>(null);
  const [showFieldForm, setShowFieldForm] = useState(false);
  const [editingField, setEditingField] = useState<DocumentTypeField | null>(null);
  const queryClient = useQueryClient();

  const { data: documentTypes, isLoading } = useQuery({
    queryKey: ['document-types'],
    queryFn: listDocumentTypes,
  });

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
    mutationFn: (id: number) => deleteDocumentTypeField(id),
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
        <FontAwesomeIcon icon={faCog} className="text-white/40 text-4xl animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-white text-3xl font-bold">Document Types</h1>
          <p className="text-white/60 mt-1">Beheer document types en hun velden</p>
        </div>
        <button
          onClick={() => {
            setShowCreateForm(true);
            setSelectedType(null);
            setEditingType(null);
          }}
          className="flex items-center space-x-2 px-4 py-2.5 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl hover:from-blue-500 hover:to-blue-600 transition-all shadow-lg shadow-blue-500/25"
        >
          <FontAwesomeIcon icon={faPlus} className="w-4 h-4" />
          <span className="font-medium">Nieuw Type</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel - Document Types List */}
        <div className="lg:col-span-1">
          <div className="glass-card p-4 space-y-2">
            <h2 className="text-white font-semibold px-2 pb-2 border-b border-white/10">
              Document Types ({documentTypes?.length || 0})
            </h2>
            
            {documentTypes?.length === 0 && (
              <div className="text-center py-8">
                <FontAwesomeIcon icon={faFileAlt} className="text-white/20 text-3xl mb-2" />
                <p className="text-white/40 text-sm">Geen document types</p>
              </div>
            )}

            {documentTypes?.map((docType) => (
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
                    ? 'bg-blue-500/20 border border-blue-500/30'
                    : 'hover:bg-white/5 border border-transparent'
                }`}
              >
                <div className="flex items-center space-x-3 min-w-0">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                    currentType?.slug === docType.slug ? 'bg-blue-500/30' : 'bg-white/10'
                  }`}>
                    <FontAwesomeIcon
                      icon={faFileAlt}
                      className={currentType?.slug === docType.slug ? 'text-blue-400' : 'text-white/60'}
                    />
                  </div>
                  <div className="min-w-0">
                    <p className="text-white font-medium truncate">{docType.name}</p>
                    <p className="text-white/50 text-xs">{docType.fields.length} velden</p>
                  </div>
                </div>
                <FontAwesomeIcon
                  icon={faChevronRight}
                  className={`w-3 h-3 flex-shrink-0 ${
                    currentType?.slug === docType.slug ? 'text-blue-400' : 'text-white/30'
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
            <div className="glass-card p-6">
              <DocumentTypeForm
                onSubmit={(data) => createMutation.mutate(data)}
                onCancel={() => setShowCreateForm(false)}
                isLoading={createMutation.isPending}
              />
            </div>
          )}

          {/* Type Details */}
          {currentType && !showCreateForm && (
            <div className="space-y-6">
              {/* Type Header Card */}
              <div className="glass-card p-6">
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
                        <h2 className="text-white text-2xl font-bold">{currentType.name}</h2>
                        <p className="text-white/50 text-sm mt-1">
                          <code className="bg-white/10 px-2 py-0.5 rounded">{currentType.slug}</code>
                        </p>
                      </div>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => setEditingType(currentType)}
                          className="flex items-center space-x-2 px-3 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
                        >
                          <FontAwesomeIcon icon={faEdit} className="w-4 h-4" />
                          <span className="text-sm">Bewerken</span>
                        </button>
                        <button
                          onClick={() => {
                            if (confirm(`Document type "${currentType.name}" verwijderen?`)) {
                              deleteMutation.mutate(currentType.slug);
                            }
                          }}
                          className="flex items-center space-x-2 px-3 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                        >
                          <FontAwesomeIcon icon={faTrash} className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {currentType.description && (
                      <p className="text-white/70 mb-4">{currentType.description}</p>
                    )}

                    {/* Hints & Preamble Accordions */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {currentType.classification_hints && (
                        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                          <h4 className="text-white/80 text-sm font-medium mb-2 flex items-center">
                            <FontAwesomeIcon icon={faCode} className="w-3 h-3 mr-2 text-amber-400" />
                            Classificatie Hints
                          </h4>
                          <pre className="text-white/60 text-xs whitespace-pre-wrap font-mono max-h-32 overflow-y-auto">
                            {currentType.classification_hints}
                          </pre>
                        </div>
                      )}
                      {currentType.extraction_prompt_preamble && (
                        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                          <h4 className="text-white/80 text-sm font-medium mb-2 flex items-center">
                            <FontAwesomeIcon icon={faInfoCircle} className="w-3 h-3 mr-2 text-cyan-400" />
                            Extractie Preamble
                          </h4>
                          <pre className="text-white/60 text-xs whitespace-pre-wrap font-mono max-h-32 overflow-y-auto">
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
                    <h3 className="text-white font-semibold flex items-center">
                      <FontAwesomeIcon icon={faListUl} className="w-4 h-4 mr-2 text-purple-400" />
                      Velden ({currentType.fields.length})
                    </h3>
                    <button
                      onClick={() => {
                        setShowFieldForm(true);
                        setEditingField(null);
                      }}
                      className="flex items-center space-x-2 px-3 py-1.5 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors text-sm"
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
                    <div className="text-center py-8 bg-white/5 rounded-xl border-2 border-dashed border-white/10">
                      <FontAwesomeIcon icon={faListUl} className="text-white/20 text-3xl mb-2" />
                      <p className="text-white/40">Geen velden gedefinieerd</p>
                      <p className="text-white/30 text-sm mt-1">Klik op "Veld Toevoegen" om te starten</p>
                    </div>
                  )}

                  <div className="space-y-1">
                    {currentType.fields
                      .filter(field => !editingField || field.id !== editingField.id)
                      .map((field) => (
                      <div
                        key={field.id}
                        className="flex items-center justify-between px-3 py-2 bg-white/5 rounded-lg border border-white/10 hover:border-white/20 transition-colors group"
                      >
                        <div className="flex items-center space-x-3 min-w-0 flex-1">
                          <div className={`px-2 py-0.5 rounded text-xs font-medium shrink-0 ${fieldTypeColors[field.field_type] || 'bg-gray-500/20 text-gray-400'}`}>
                            {fieldTypeLabels[field.field_type] || field.field_type}
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center space-x-2 flex-wrap">
                              <span className="text-white font-medium text-sm">{field.label}</span>
                              <code className="text-white/40 text-xs bg-white/10 px-1 py-0.5 rounded">
                                {field.key}
                              </code>
                              {field.regex && (
                                <code className="text-amber-400/70 text-xs bg-amber-500/10 px-1 py-0.5 rounded truncate max-w-[150px]" title={field.regex}>
                                  {field.regex}
                                </code>
                              )}
                              {field.enum_values && field.enum_values.length > 0 && (
                                <span className="text-white/40 text-xs truncate">
                                  [{field.enum_values.join(', ')}]
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2 shrink-0">
                          {/* Required Toggle */}
                          <button
                            type="button"
                            onClick={() => {
                              updateFieldMutation.mutate({
                                slug: currentType.slug,
                                fieldId: field.id,
                                data: { required: !field.required }
                              });
                            }}
                            className={`w-8 h-5 rounded-full transition-colors flex items-center ${field.required ? 'bg-emerald-500' : 'bg-white/20'}`}
                            title={field.required ? 'Verplicht' : 'Optioneel'}
                          >
                            <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform ${field.required ? 'translate-x-4' : 'translate-x-1'}`} />
                          </button>
                          {/* Edit/Delete Buttons */}
                          <div className="flex space-x-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                              onClick={() => {
                                setEditingField(field);
                                setShowFieldForm(true);
                              }}
                              className="p-1.5 text-white/60 hover:text-white hover:bg-white/10 rounded"
                              title="Bewerken"
                            >
                              <FontAwesomeIcon icon={faEdit} className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => {
                                if (confirm(`Veld "${field.label}" verwijderen?`)) {
                                  deleteFieldMutation.mutate(field.id);
                                }
                              }}
                              className="p-1.5 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded"
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
            </div>
          )}

          {/* Empty State */}
          {!currentType && !showCreateForm && (
            <div className="glass-card p-12 text-center">
              <FontAwesomeIcon icon={faFileAlt} className="text-white/20 text-5xl mb-4" />
              <h3 className="text-white text-lg font-medium mb-2">Selecteer een document type</h3>
              <p className="text-white/50 max-w-md mx-auto">
                Klik op een document type in de lijst om de details te bekijken en velden te beheren.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Document Type Form Component
interface DocumentTypeFormProps {
  initialData?: DocumentType;
  onSubmit: (data: any) => void;
  onCancel: () => void;
  isLoading: boolean;
}

function DocumentTypeForm({ initialData, onSubmit, onCancel, isLoading }: DocumentTypeFormProps) {
  const [formData, setFormData] = useState({
    name: initialData?.name || '',
    slug: initialData?.slug || '',
    description: initialData?.description || '',
    classification_hints: initialData?.classification_hints || '',
    extraction_prompt_preamble: initialData?.extraction_prompt_preamble || '',
  });
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
        <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
          <FontAwesomeIcon icon={initialData ? faEdit : faPlus} className="text-blue-400" />
        </div>
        <h3 className="text-white text-xl font-semibold">
          {initialData ? 'Document Type Bewerken' : 'Nieuw Document Type'}
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-white/80 text-sm font-medium mb-2">Naam *</label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => handleNameChange(e.target.value)}
            className={`w-full px-4 py-2.5 bg-white/10 border rounded-xl text-white placeholder-white/40 focus:ring-2 focus:ring-blue-400 focus:border-transparent ${
              nameError ? 'border-red-400' : 'border-white/20'
            }`}
            placeholder="Bijv. Bankafschrift"
            required
          />
          {nameError && (
            <p className="text-red-400 text-xs mt-1 flex items-center">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 mr-1" />
              {nameError}
            </p>
          )}
        </div>

        <div>
          <label className="block text-white/80 text-sm font-medium mb-2">
            Slug {initialData ? '' : '(automatisch)'}
          </label>
          <input
            type="text"
            value={formData.slug}
            onChange={(e) => handleSlugChange(e.target.value)}
            readOnly={!initialData}
            className={`w-full px-4 py-2.5 border rounded-xl text-white placeholder-white/40 focus:ring-2 focus:ring-blue-400 focus:border-transparent ${
              slugError ? 'border-red-400 bg-white/10' : initialData ? 'bg-white/10 border-white/20' : 'bg-white/5 border-white/10 cursor-not-allowed'
            }`}
            required
          />
          {slugError && (
            <p className="text-red-400 text-xs mt-1 flex items-center">
              <FontAwesomeIcon icon={faExclamationTriangle} className="w-3 h-3 mr-1" />
              {slugError}
            </p>
          )}
        </div>
      </div>

      <div>
        <label className="block text-white/80 text-sm font-medium mb-2">Beschrijving</label>
        <textarea
          value={formData.description}
          onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
          rows={2}
          className="w-full px-4 py-2.5 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/40 focus:ring-2 focus:ring-blue-400 focus:border-transparent"
          placeholder="Korte beschrijving van dit document type..."
        />
      </div>

      <div>
        <label className="block text-white/80 text-sm font-medium mb-2">
          Classificatie Hints
          <span className="text-white/40 font-normal ml-2">(trefwoorden per regel)</span>
        </label>
        <textarea
          value={formData.classification_hints}
          onChange={(e) => setFormData(prev => ({ ...prev, classification_hints: e.target.value }))}
          rows={4}
          className="w-full px-4 py-2.5 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/40 focus:ring-2 focus:ring-blue-400 focus:border-transparent font-mono text-sm"
          placeholder="kw:bankafschrift&#10;kw:rekeningnummer&#10;re:NL\d{2}[A-Z]{4}\d{10}"
        />
      </div>

      <div>
        <label className="block text-white/80 text-sm font-medium mb-2">
          Extractie Prompt Preamble
          <span className="text-white/40 font-normal ml-2">(extra context voor LLM)</span>
        </label>
        <textarea
          value={formData.extraction_prompt_preamble}
          onChange={(e) => setFormData(prev => ({ ...prev, extraction_prompt_preamble: e.target.value }))}
          rows={3}
          className="w-full px-4 py-2.5 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/40 focus:ring-2 focus:ring-blue-400 focus:border-transparent font-mono text-sm"
          placeholder="Extra instructies voor metadata extractie..."
        />
      </div>

      <div className="flex space-x-3 pt-2">
        <button
          type="submit"
          disabled={isLoading}
          className="flex items-center space-x-2 px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-emerald-700 text-white rounded-xl hover:from-emerald-500 hover:to-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          <FontAwesomeIcon icon={faSave} className="w-4 h-4" />
          <span>{isLoading ? 'Opslaan...' : 'Opslaan'}</span>
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-5 py-2.5 bg-white/10 text-white rounded-xl hover:bg-white/20 transition-colors"
        >
          Annuleren
        </button>
      </div>
    </form>
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
    };
    onSubmit(submitData);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-gradient-to-br from-emerald-500/10 to-blue-500/10 p-5 rounded-xl border border-emerald-500/20 space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-white font-medium flex items-center">
          <FontAwesomeIcon icon={initialData ? faEdit : faPlus} className="w-4 h-4 mr-2 text-emerald-400" />
          {initialData ? 'Veld Bewerken' : 'Nieuw Veld'}
        </h4>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-white/80 text-sm font-medium mb-1.5">Label *</label>
          <input
            type="text"
            value={formData.label}
            onChange={(e) => handleLabelChange(e.target.value)}
            className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm"
            placeholder="Bijv. IBAN Nummer"
            required
          />
        </div>

        <div>
          <label className="block text-white/80 text-sm font-medium mb-1.5">
            Key {initialData ? '' : '(automatisch)'}
          </label>
          <input
            type="text"
            value={formData.key}
            onChange={(e) => setFormData(prev => ({ ...prev, key: e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '_') }))}
            readOnly={!initialData}
            className={`w-full px-3 py-2 border rounded-lg text-white placeholder-white/40 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm font-mono ${
              initialData ? 'bg-white/10 border-white/20' : 'bg-white/5 border-white/10 cursor-not-allowed'
            }`}
            required
          />
        </div>
      </div>

      <div>
        <label className="block text-white/80 text-sm font-medium mb-1.5">Type *</label>
        <select
          value={formData.field_type}
          onChange={(e) => setFormData(prev => ({ ...prev, field_type: e.target.value }))}
          className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm"
        >
          {Object.entries(fieldTypeLabels).map(([value, label]) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
      </div>

      {formData.field_type === 'enum' && (
        <div>
          <label className="block text-white/80 text-sm font-medium mb-1.5">Opties (komma-gescheiden)</label>
          <input
            type="text"
            value={formData.enum_values}
            onChange={(e) => setFormData(prev => ({ ...prev, enum_values: e.target.value }))}
            placeholder="optie1, optie2, optie3"
            className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm"
          />
        </div>
      )}

      <div>
        <label className="block text-white/80 text-sm font-medium mb-1.5">
          Regex Validatie
          <span className="text-white/40 font-normal ml-2">(optioneel)</span>
        </label>
        <input
          type="text"
          value={formData.regex}
          onChange={(e) => setFormData(prev => ({ ...prev, regex: e.target.value }))}
          placeholder="^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}.*$"
          className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm font-mono"
        />
        <p className="text-white/40 text-xs mt-1">Reguliere expressie voor waarde validatie</p>
      </div>

      <div>
        <label className="block text-white/80 text-sm font-medium mb-1.5">Hulptekst</label>
        <input
          type="text"
          value={formData.description}
          onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
          placeholder="Extra uitleg over dit veld..."
          className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:ring-2 focus:ring-emerald-400 focus:border-transparent text-sm"
        />
      </div>

      <div className="flex space-x-2 pt-2">
        <button
          type="submit"
          disabled={isLoading}
          className="flex items-center space-x-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
        >
          <FontAwesomeIcon icon={faCheck} className="w-3 h-3" />
          <span>{isLoading ? 'Opslaan...' : 'Opslaan'}</span>
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors text-sm"
        >
          Annuleren
        </button>
      </div>
    </form>
  );
}
