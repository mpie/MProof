'use client';

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPlus, faTrash, faSave, faCode, faEye,
  faCheckCircle, faTimesCircle, faShieldAlt
} from '@fortawesome/free-solid-svg-icons';
import {
  listSignals,
  getDocumentTypePolicy,
  updateDocumentTypePolicy,
  deleteDocumentTypePolicy,
  previewEligibility,
  Signal,
  SignalRequirement,
  ClassificationPolicy,
  AcceptanceConfig,
  EligibilityPreviewResponse,
} from '@/lib/api';

interface SignalPolicyEditorProps {
  slug: string;
  onClose?: () => void;
}

const DEFAULT_ACCEPTANCE: AcceptanceConfig = {
  trained_model: { enabled: true, min_confidence: 0.85, min_margin: 0.10 },
  deterministic: { enabled: true },
  llm: { enabled: true, require_evidence: true },
};

export function SignalPolicyEditor({ slug, onClose }: SignalPolicyEditorProps) {
  const queryClient = useQueryClient();

  // State
  const [requirements, setRequirements] = useState<SignalRequirement[]>([]);
  const [exclusions, setExclusions] = useState<SignalRequirement[]>([]);
  const [acceptance, setAcceptance] = useState<AcceptanceConfig>(DEFAULT_ACCEPTANCE);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [rawJson, setRawJson] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [previewText, setPreviewText] = useState('');
  const [previewResult, setPreviewResult] = useState<EligibilityPreviewResponse | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Queries
  const { data: signalsData } = useQuery({
    queryKey: ['signals'],
    queryFn: listSignals,
  });

  const { data: policyData, isLoading: isPolicyLoading } = useQuery({
    queryKey: ['document-type-policy', slug],
    queryFn: () => getDocumentTypePolicy(slug),
  });

  const signals = signalsData?.signals || [];

  // Reset preview when slug changes
  useEffect(() => {
    setPreviewResult(null);
  }, [slug]);

  // Initialize form from policy
  useEffect(() => {
    if (policyData?.policy) {
      const policy = policyData.policy;
      setRequirements(policy.requirements || []);
      setExclusions(policy.exclusions || []);
      setAcceptance(policy.acceptance || DEFAULT_ACCEPTANCE);
      setRawJson(JSON.stringify(policy, null, 2));
    } else if (policyData !== undefined) {
      // Only reset if we have a definitive "no policy" response, not during loading
      setRequirements([]);
      setExclusions([]);
      setAcceptance(DEFAULT_ACCEPTANCE);
      setRawJson(JSON.stringify({ requirements: [], exclusions: [], acceptance: DEFAULT_ACCEPTANCE }, null, 2));
    }
  }, [policyData, slug]);

  // Mutations
  const updateMutation = useMutation({
    mutationFn: (policy: ClassificationPolicy) => updateDocumentTypePolicy(slug, policy),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document-type-policy', slug] });
      queryClient.invalidateQueries({ queryKey: ['document-types'] });
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2000);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteDocumentTypePolicy(slug),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document-type-policy', slug] });
      queryClient.invalidateQueries({ queryKey: ['document-types'] });
      setRequirements([]);
      setExclusions([]);
      setAcceptance(DEFAULT_ACCEPTANCE);
    },
  });

  const previewMutation = useMutation({
    mutationFn: async () => {
      const policy = buildPolicy();
      return previewEligibility(slug, previewText, policy);
    },
    onSuccess: (result) => {
      setPreviewResult(result);
    },
  });

  // Build policy from form state
  const buildPolicy = (): ClassificationPolicy => ({
    requirements,
    exclusions,
    acceptance,
  });

  // Handle save
  const handleSave = () => {
    if (showAdvanced) {
      try {
        const parsed = JSON.parse(rawJson);
        updateMutation.mutate(parsed);
      } catch {
        setJsonError('Ongeldige JSON');
      }
    } else {
      const policy = buildPolicy();
      setRawJson(JSON.stringify(policy, null, 2));
      updateMutation.mutate(policy);
    }
  };

  // Handle raw JSON changes
  const handleRawJsonChange = (value: string) => {
    setRawJson(value);
    try {
      JSON.parse(value);
      setJsonError(null);
    } catch {
      setJsonError('Ongeldige JSON');
    }
  };

  // Sync form when switching from advanced mode
  useEffect(() => {
    if (!showAdvanced && rawJson) {
      try {
        const parsed = JSON.parse(rawJson);
        setRequirements(parsed.requirements || []);
        setExclusions(parsed.exclusions || []);
        setAcceptance(parsed.acceptance || DEFAULT_ACCEPTANCE);
      } catch {
        // Ignore parse errors when switching
      }
    }
  }, [showAdvanced, rawJson]);

  // Update raw JSON when form changes
  useEffect(() => {
    if (!showAdvanced) {
      const policy = buildPolicy();
      setRawJson(JSON.stringify(policy, null, 2));
    }
  }, [requirements, exclusions, acceptance, showAdvanced]);

  // Add requirement
  const addRequirement = () => {
    const firstSignal = signals.find(s => s.signal_type === 'count') || signals[0];
    if (firstSignal) {
      const newReq: SignalRequirement = {
        signal: firstSignal.key,
        op: firstSignal.signal_type === 'boolean' ? '==' : '>=',
        value: firstSignal.signal_type === 'boolean' ? true : 1,
      };
      setRequirements([...requirements, newReq]);
    }
  };

  // Add exclusion
  const addExclusion = () => {
    const firstSignal = signals.find(s => s.signal_type === 'boolean') || signals[0];
    if (firstSignal) {
      const newExcl: SignalRequirement = {
        signal: firstSignal.key,
        op: '==',
        value: true,
      };
      setExclusions([...exclusions, newExcl]);
    }
  };

  // Update requirement
  const updateRequirement = (index: number, field: keyof SignalRequirement, value: unknown) => {
    const updated = [...requirements];
    updated[index] = { ...updated[index], [field]: value };

    if (field === 'signal') {
      const signal = signals.find(s => s.key === value);
      if (signal) {
        updated[index].value = signal.signal_type === 'boolean' ? true : 1;
        updated[index].op = signal.signal_type === 'boolean' ? '==' : '>=';
      }
    }

    setRequirements(updated);
  };

  // Update exclusion
  const updateExclusion = (index: number, field: keyof SignalRequirement, value: unknown) => {
    const updated = [...exclusions];
    updated[index] = { ...updated[index], [field]: value };

    if (field === 'signal') {
      const signal = signals.find(s => s.key === value);
      if (signal) {
        updated[index].value = signal.signal_type === 'boolean' ? true : 1;
        updated[index].op = '==';
      }
    }

    setExclusions(updated);
  };

  // Remove requirement/exclusion
  const removeRequirement = (index: number) => {
    setRequirements(requirements.filter((_, i) => i !== index));
  };

  const removeExclusion = (index: number) => {
    setExclusions(exclusions.filter((_, i) => i !== index));
  };

  // Render signal dropdown
  const SignalSelect = ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="flex-1 px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
    >
      {signals.map((s) => (
        <option key={s.key} value={s.key} className="bg-gray-800">
          {s.label} ({s.signal_type})
        </option>
      ))}
    </select>
  );

  // Render operator dropdown
  const OperatorSelect = ({ value, onChange, isBoolean }: { value: string; onChange: (v: string) => void; isBoolean: boolean }) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-20 px-2 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm text-center"
    >
      {isBoolean ? (
        <>
          <option value="==" className="bg-gray-800">=</option>
          <option value="!=" className="bg-gray-800">≠</option>
        </>
      ) : (
        <>
          <option value=">=" className="bg-gray-800">≥</option>
          <option value=">" className="bg-gray-800">&gt;</option>
          <option value="<=" className="bg-gray-800">≤</option>
          <option value="<" className="bg-gray-800">&lt;</option>
          <option value="==" className="bg-gray-800">=</option>
          <option value="!=" className="bg-gray-800">≠</option>
        </>
      )}
    </select>
  );

  if (isPolicyLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin w-6 h-6 border-2 border-white/30 border-t-white rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <FontAwesomeIcon icon={faShieldAlt} className="text-indigo-400" />
          Classificatiebeleid
        </h3>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
            showAdvanced
              ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
              : 'bg-white/10 text-white/60 hover:bg-white/20'
          }`}
        >
          <FontAwesomeIcon icon={faCode} className="mr-2" />
          {showAdvanced ? 'Eenvoudig' : 'JSON'}
        </button>
      </div>

      {showAdvanced ? (
        /* JSON Editor */
        <div className="space-y-4">
          <div className="bg-white/5 border border-white/10 rounded-xl p-4">
            <textarea
              value={rawJson}
              onChange={(e) => handleRawJsonChange(e.target.value)}
              className="w-full h-80 px-4 py-3 bg-gray-900 border border-white/20 rounded-lg text-white font-mono text-sm resize-y"
              placeholder="Voer beleid JSON in..."
            />
            {jsonError && (
              <p className="text-red-400 text-sm mt-2">{jsonError}</p>
            )}
          </div>
        </div>
      ) : (
        /* Signal-based Editor */
        <div className="space-y-6">
          {/* Requirements */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-white font-medium">Vereisten</h4>
                <p className="text-white/50 text-sm">Document moet aan ALLE vereisten voldoen</p>
              </div>
              <button
                onClick={addRequirement}
                className="px-3 py-1.5 bg-teal-500/20 text-teal-300 rounded-lg text-sm hover:bg-teal-500/30 transition-colors"
              >
                <FontAwesomeIcon icon={faPlus} className="mr-2" />
                Toevoegen
              </button>
            </div>

            {requirements.length === 0 ? (
              <p className="text-white/40 text-sm italic">Geen vereisten (alle documenten komen in aanmerking)</p>
            ) : (
              <div className="space-y-2">
                {requirements.map((req, index) => {
                  const signal = signals.find(s => s.key === req.signal);
                  const isBoolean = signal?.signal_type === 'boolean';

                  return (
                    <div key={index} className="flex items-center gap-2 bg-white/5 rounded-lg p-2">
                      <SignalSelect
                        value={req.signal}
                        onChange={(v) => updateRequirement(index, 'signal', v)}
                      />
                      <OperatorSelect
                        value={req.op}
                        onChange={(v) => updateRequirement(index, 'op', v)}
                        isBoolean={isBoolean || false}
                      />
                      {isBoolean ? (
                        <select
                          value={String(req.value)}
                          onChange={(e) => updateRequirement(index, 'value', e.target.value === 'true')}
                          className="w-24 px-2 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
                        >
                          <option value="true" className="bg-gray-800">Ja</option>
                          <option value="false" className="bg-gray-800">Nee</option>
                        </select>
                      ) : (
                        <input
                          type="number"
                          value={Number(req.value)}
                          onChange={(e) => updateRequirement(index, 'value', parseInt(e.target.value) || 0)}
                          className="w-24 px-2 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
                          min={0}
                        />
                      )}
                      <button
                        onClick={() => removeRequirement(index)}
                        className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
                      >
                        <FontAwesomeIcon icon={faTrash} />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Exclusions */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-white font-medium">Uitsluitingen</h4>
                <p className="text-white/50 text-sm">Als EEN uitsluiting matcht, komt document NIET in aanmerking</p>
              </div>
              <button
                onClick={addExclusion}
                className="px-3 py-1.5 bg-red-500/20 text-red-300 rounded-lg text-sm hover:bg-red-500/30 transition-colors"
              >
                <FontAwesomeIcon icon={faPlus} className="mr-2" />
                Toevoegen
              </button>
            </div>

            {exclusions.length === 0 ? (
              <p className="text-white/40 text-sm italic">Geen uitsluitingen</p>
            ) : (
              <div className="space-y-2">
                {exclusions.map((excl, index) => {
                  const signal = signals.find(s => s.key === excl.signal);
                  const isBoolean = signal?.signal_type === 'boolean';

                  return (
                    <div key={index} className="flex items-center gap-2 bg-red-500/5 border border-red-500/10 rounded-lg p-2">
                      <SignalSelect
                        value={excl.signal}
                        onChange={(v) => updateExclusion(index, 'signal', v)}
                      />
                      <OperatorSelect
                        value={excl.op}
                        onChange={(v) => updateExclusion(index, 'op', v)}
                        isBoolean={isBoolean || false}
                      />
                      {isBoolean ? (
                        <select
                          value={String(excl.value)}
                          onChange={(e) => updateExclusion(index, 'value', e.target.value === 'true')}
                          className="w-24 px-2 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
                        >
                          <option value="true" className="bg-gray-800">Ja</option>
                          <option value="false" className="bg-gray-800">Nee</option>
                        </select>
                      ) : (
                        <input
                          type="number"
                          value={Number(excl.value)}
                          onChange={(e) => updateExclusion(index, 'value', parseInt(e.target.value) || 0)}
                          className="w-24 px-2 py-2 bg-white/10 border border-white/20 rounded-lg text-white text-sm"
                          min={0}
                        />
                      )}
                      <button
                        onClick={() => removeExclusion(index)}
                        className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
                      >
                        <FontAwesomeIcon icon={faTrash} />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Acceptance Thresholds */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-4">
            <h4 className="text-white font-medium mb-4">Acceptatiedrempels</h4>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Trained Model */}
              <div className="p-3 bg-teal-500/10 border border-teal-500/20 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-teal-300 text-sm font-medium">Getraind Model</span>
                  <button
                    onClick={() => setAcceptance({
                      ...acceptance,
                      trained_model: { ...acceptance.trained_model, enabled: !acceptance.trained_model?.enabled }
                    })}
                    className={`w-8 h-5 rounded-full transition-colors flex items-center ${
                      acceptance.trained_model?.enabled ? 'bg-teal-500' : 'bg-white/20'
                    }`}
                  >
                    <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform ${
                      acceptance.trained_model?.enabled ? 'translate-x-4' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
                {acceptance.trained_model?.enabled && (
                  <div className="space-y-2">
                    <div>
                      <label className="text-white/50 text-xs">Min. Zekerheid</label>
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.05}
                        value={acceptance.trained_model?.min_confidence ?? 0.85}
                        onChange={(e) => setAcceptance({
                          ...acceptance,
                          trained_model: { ...acceptance.trained_model, min_confidence: parseFloat(e.target.value) }
                        })}
                        className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-white/50 text-xs">Min. Marge</label>
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.05}
                        value={acceptance.trained_model?.min_margin ?? 0.10}
                        onChange={(e) => setAcceptance({
                          ...acceptance,
                          trained_model: { ...acceptance.trained_model, min_margin: parseFloat(e.target.value) }
                        })}
                        className="w-full px-2 py-1 bg-white/10 border border-white/20 rounded text-white text-sm"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Deterministic */}
              <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-amber-300 text-sm font-medium">Deterministisch</span>
                  <button
                    onClick={() => setAcceptance({
                      ...acceptance,
                      deterministic: { enabled: !acceptance.deterministic?.enabled }
                    })}
                    className={`w-8 h-5 rounded-full transition-colors flex items-center ${
                      acceptance.deterministic?.enabled ? 'bg-amber-500' : 'bg-white/20'
                    }`}
                  >
                    <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform ${
                      acceptance.deterministic?.enabled ? 'translate-x-4' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
              </div>

              {/* LLM */}
              <div className="p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-purple-300 text-sm font-medium">LLM</span>
                  <button
                    onClick={() => setAcceptance({
                      ...acceptance,
                      llm: { ...acceptance.llm, enabled: !acceptance.llm?.enabled }
                    })}
                    className={`w-8 h-5 rounded-full transition-colors flex items-center ${
                      acceptance.llm?.enabled ? 'bg-purple-500' : 'bg-white/20'
                    }`}
                  >
                    <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform ${
                      acceptance.llm?.enabled ? 'translate-x-4' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
                {acceptance.llm?.enabled && (
                  <div className="flex items-center gap-2">
                    <label className="text-white/50 text-xs">Bewijs vereist</label>
                    <button
                      onClick={() => setAcceptance({
                        ...acceptance,
                        llm: { ...acceptance.llm, require_evidence: !acceptance.llm?.require_evidence }
                      })}
                      className={`w-6 h-4 rounded-full transition-colors flex items-center ${
                        acceptance.llm?.require_evidence ? 'bg-purple-400' : 'bg-white/20'
                      }`}
                    >
                      <div className={`w-2.5 h-2.5 bg-white rounded-full shadow-md transform transition-transform ${
                        acceptance.llm?.require_evidence ? 'translate-x-3' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Preview */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-white font-medium flex items-center gap-2">
                  <FontAwesomeIcon icon={faEye} className="text-blue-400" />
                  Geschiktheidspreview
                </h4>
                <p className="text-white/50 text-sm">Test hoe een document zou worden geëvalueerd</p>
              </div>
              <button
                onClick={() => setShowPreview(!showPreview)}
                className="px-3 py-1.5 bg-blue-500/20 text-blue-300 rounded-lg text-sm hover:bg-blue-500/30 transition-colors"
              >
                {showPreview ? 'Verbergen' : 'Tonen'}
              </button>
            </div>

            {showPreview && (
              <div className="space-y-4">
                <textarea
                  value={previewText}
                  onChange={(e) => setPreviewText(e.target.value)}
                  placeholder="Plak voorbeeldtekst hier..."
                  className="w-full h-32 px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white text-sm resize-y"
                />
                <button
                  onClick={() => previewMutation.mutate()}
                  disabled={!previewText || previewMutation.isPending}
                  className="px-4 py-2 bg-blue-500/20 text-blue-300 rounded-lg hover:bg-blue-500/30 disabled:opacity-50 transition-colors"
                >
                  {previewMutation.isPending ? 'Testen...' : 'Test Geschiktheid'}
                </button>

                {previewResult && (
                  <div className={`p-4 rounded-lg ${
                    previewResult.is_eligible
                      ? 'bg-green-500/10 border border-green-500/30'
                      : 'bg-red-500/10 border border-red-500/30'
                  }`}>
                    <div className="flex items-center gap-2 mb-3">
                      <FontAwesomeIcon
                        icon={previewResult.is_eligible ? faCheckCircle : faTimesCircle}
                        className={previewResult.is_eligible ? 'text-green-400' : 'text-red-400'}
                      />
                      <span className={previewResult.is_eligible ? 'text-green-300' : 'text-red-300'}>
                        {previewResult.is_eligible ? 'Document KOMT IN AANMERKING' : 'Document komt NIET in aanmerking'}
                      </span>
                    </div>

                    {/* Computed Signals */}
                    <div className="mb-3">
                      <span className="text-white/60 text-sm">Berekende signalen:</span>
                      <div className="flex flex-wrap gap-2 mt-1">
                        {previewResult.computed_signals.map((s) => (
                          <span key={s.key} className="px-2 py-1 bg-white/10 rounded text-xs text-white/80">
                            {s.label}: {String(s.value)}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Failed Requirements */}
                    {previewResult.failed_requirements.length > 0 && (
                      <div className="mb-2">
                        <span className="text-red-300 text-sm">Niet voldaan aan vereisten:</span>
                        <ul className="list-disc list-inside text-red-200 text-xs">
                          {previewResult.failed_requirements.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Triggered Exclusions */}
                    {previewResult.triggered_exclusions.length > 0 && (
                      <div>
                        <span className="text-red-300 text-sm">Getriggerde uitsluitingen:</span>
                        <ul className="list-disc list-inside text-red-200 text-xs">
                          {previewResult.triggered_exclusions.map((e, i) => (
                            <li key={i}>{e}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-white/10">
        <button
          onClick={() => deleteMutation.mutate()}
          disabled={deleteMutation.isPending}
          className="px-4 py-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
        >
          Reset naar Standaard
        </button>

        <div className="flex items-center gap-3">
          {saveSuccess && (
            <span className="text-green-400 text-sm flex items-center gap-1">
              <FontAwesomeIcon icon={faCheckCircle} />
              Opgeslagen!
            </span>
          )}
          {onClose && (
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white/10 text-white/70 rounded-lg hover:bg-white/20 transition-colors"
            >
              Annuleren
            </button>
          )}
          <button
            onClick={handleSave}
            disabled={updateMutation.isPending || (showAdvanced && !!jsonError)}
            className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-indigo-600 to-indigo-700 text-white rounded-lg hover:from-indigo-500 hover:to-indigo-600 disabled:opacity-50 transition-all"
          >
            <FontAwesomeIcon icon={faSave} />
            {updateMutation.isPending ? 'Opslaan...' : 'Beleid Opslaan'}
          </button>
        </div>
      </div>
    </div>
  );
}
