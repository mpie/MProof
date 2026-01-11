'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPlus, faEdit, faTrash, faSave, faTimes, faSearch, faCode,
  faCheckCircle, faTimesCircle, faCog, faFlask, faQuestionCircle,
  faLightbulb, faChevronDown, faChevronUp
} from '@fortawesome/free-solid-svg-icons';
import {
  listSignals,
  createSignal,
  updateSignal,
  deleteSignal,
  testSignals,
  Signal,
  SignalCreate,
  SignalUpdate,
  SignalTestResponse,
} from '@/lib/api';

export default function SignalsPage() {
  const queryClient = useQueryClient();
  
  // State
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingSignal, setEditingSignal] = useState<Signal | null>(null);
  const [showTestPanel, setShowTestPanel] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [testText, setTestText] = useState('');
  const [testResult, setTestResult] = useState<SignalTestResponse | null>(null);
  
  // Form state
  const [formKey, setFormKey] = useState('');
  const [formLabel, setFormLabel] = useState('');
  const [formDescription, setFormDescription] = useState('');
  const [formSignalType, setFormSignalType] = useState<'boolean' | 'count'>('boolean');
  const [formComputeKind, setFormComputeKind] = useState<'keyword_set' | 'regex_set'>('keyword_set');
  const [formMatchMode, setFormMatchMode] = useState<'any' | 'all'>('any');
  const [formKeywords, setFormKeywords] = useState('');
  const [formPatterns, setFormPatterns] = useState('');

  // Queries
  const { data: signalsData, isLoading } = useQuery({
    queryKey: ['signals'],
    queryFn: listSignals,
  });

  const signals = signalsData?.signals || [];
  const builtinSignals = signals.filter(s => s.source === 'builtin');
  const userSignals = signals.filter(s => s.source === 'user');

  // Mutations
  const createMutation = useMutation({
    mutationFn: createSignal,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['signals'] });
      resetForm();
      setShowCreateForm(false);
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ key, data }: { key: string; data: SignalUpdate }) => updateSignal(key, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['signals'] });
      resetForm();
      setEditingSignal(null);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteSignal,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['signals'] });
    },
  });

  const testMutation = useMutation({
    mutationFn: testSignals,
    onSuccess: (result) => {
      setTestResult(result);
    },
  });

  const resetForm = () => {
    setFormKey('');
    setFormLabel('');
    setFormDescription('');
    setFormSignalType('boolean');
    setFormComputeKind('keyword_set');
    setFormMatchMode('any');
    setFormKeywords('');
    setFormPatterns('');
  };

  const startEdit = (signal: Signal) => {
    setEditingSignal(signal);
    setFormLabel(signal.label);
    setFormDescription(signal.description || '');
    setFormMatchMode(signal.config_json?.match_mode || 'any');
    if (signal.compute_kind === 'keyword_set') {
      setFormKeywords(signal.config_json?.keywords?.join('\n') || '');
    } else {
      setFormPatterns(signal.config_json?.patterns?.join('\n') || '');
    }
  };

  const handleCreate = () => {
    const keywords = formKeywords.split('\n').map(k => k.trim()).filter(Boolean);
    const patterns = formPatterns.split('\n').map(p => p.trim()).filter(Boolean);

    const data: SignalCreate = {
      key: formKey.toLowerCase().replace(/[^a-z0-9_]/g, '_'),
      label: formLabel,
      description: formDescription || undefined,
      signal_type: formSignalType,
      compute_kind: formComputeKind,
      config_json: {
        match_mode: formMatchMode,
        ...(formComputeKind === 'keyword_set' ? { keywords } : { patterns }),
      },
    };

    createMutation.mutate(data);
  };

  const handleUpdate = () => {
    if (!editingSignal) return;

    const keywords = formKeywords.split('\n').map(k => k.trim()).filter(Boolean);
    const patterns = formPatterns.split('\n').map(p => p.trim()).filter(Boolean);

    const data: SignalUpdate = {
      label: formLabel,
      description: formDescription || undefined,
      config_json: {
        match_mode: formMatchMode,
        ...(editingSignal.compute_kind === 'keyword_set' ? { keywords } : { patterns }),
      },
    };

    updateMutation.mutate({ key: editingSignal.key, data });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin w-8 h-8 border-2 border-white/30 border-t-white rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Signalen</h1>
          <p className="text-white/60 mt-1">
            Bepaal welke documenten in aanmerking komen voor een documenttype
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setShowHelp(!showHelp)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-colors ${
              showHelp
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'bg-white/10 text-white hover:bg-white/20'
            }`}
          >
            <FontAwesomeIcon icon={faQuestionCircle} />
            Uitleg
          </button>
          <button
            onClick={() => setShowTestPanel(!showTestPanel)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-colors ${
              showTestPanel
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                : 'bg-white/10 text-white hover:bg-white/20'
            }`}
          >
            <FontAwesomeIcon icon={faFlask} />
            Testen
          </button>
          <button
            onClick={() => { resetForm(); setShowCreateForm(true); }}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-teal-600 to-teal-700 text-white rounded-xl hover:from-teal-500 hover:to-teal-600 transition-all"
          >
            <FontAwesomeIcon icon={faPlus} />
            Nieuw Signaal
          </button>
        </div>
      </div>

      {/* Help Section */}
      {showHelp && (
        <div className="glass-card p-6 border-l-4 border-amber-500/50">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-amber-500/20 rounded-xl">
              <FontAwesomeIcon icon={faLightbulb} className="text-amber-400 text-xl" />
            </div>
            <div className="flex-1 space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Wat zijn signalen?</h3>
                <p className="text-white/70">
                  Signalen zijn eigenschappen die automatisch uit een document worden gehaald. 
                  Denk aan: &quot;Bevat dit document een IBAN?&quot; of &quot;Hoeveel bedragen staan erin?&quot;
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Hoe gebruik je ze?</h3>
                <p className="text-white/70 mb-3">
                  Bij elk documenttype stel je regels in met signalen. Bijvoorbeeld:
                </p>
                <div className="grid md:grid-cols-2 gap-3">
                  <div className="p-3 bg-teal-500/10 border border-teal-500/20 rounded-lg">
                    <div className="text-teal-300 font-medium mb-1">✓ Vereisten</div>
                    <p className="text-white/60 text-sm">
                      &quot;Een bankafschrift moet minimaal 5 transactieregels hebben én een IBAN bevatten&quot;
                    </p>
                  </div>
                  <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                    <div className="text-red-300 font-medium mb-1">✗ Uitsluitingen</div>
                    <p className="text-white/60 text-sm">
                      &quot;Als het document factuurtermen bevat, is het géén bankafschrift&quot;
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Ingebouwde signalen</h3>
                <p className="text-white/70 mb-3">
                  Deze signalen zijn standaard beschikbaar en werken voor alle documenten:
                </p>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2 text-sm">
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <span className="text-blue-300 font-medium">IBAN aanwezig</span>
                    <p className="text-white/50">Ja/nee: bevat IBAN nummer</p>
                  </div>
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <span className="text-blue-300 font-medium">Aantal datums</span>
                    <p className="text-white/50">Telt datums (01-01-2025)</p>
                  </div>
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <span className="text-blue-300 font-medium">Aantal bedragen</span>
                    <p className="text-white/50">Telt geldbedragen (€1.234,56)</p>
                  </div>
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <span className="text-blue-300 font-medium">Transactieregels</span>
                    <p className="text-white/50">Regels met datum + bedrag samen</p>
                  </div>
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <span className="text-blue-300 font-medium">Aantal regels</span>
                    <p className="text-white/50">Totaal niet-lege regels</p>
                  </div>
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <span className="text-blue-300 font-medium">Aantal woorden</span>
                    <p className="text-white/50">Totaal aantal woorden</p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Eigen signalen maken</h3>
                <p className="text-white/70">
                  Maak signalen voor jouw specifieke documenttypes. Bijvoorbeeld een signaal 
                  &quot;Heeft factuurtermen&quot; dat zoekt naar woorden zoals <em>factuur, BTW, vervaldatum</em>. 
                  Of &quot;Heeft bankafschrift termen&quot; dat zoekt naar <em>saldo, transactie, rekeningoverzicht</em>.
                </p>
              </div>

              <div className="pt-2 border-t border-white/10">
                <p className="text-white/50 text-sm flex items-center gap-2">
                  <FontAwesomeIcon icon={faFlask} />
                  Tip: Gebruik de &quot;Testen&quot; knop om te zien welke signalen in een voorbeelddocument worden herkend.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Panel */}
      {showTestPanel && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <FontAwesomeIcon icon={faFlask} className="text-purple-400" />
            Signalen Testen
          </h2>
          <div className="space-y-4">
            <textarea
              value={testText}
              onChange={(e) => setTestText(e.target.value)}
              placeholder="Plak hier voorbeeldtekst om signalen te testen..."
              className="w-full h-32 px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white resize-y"
            />
            <div className="flex items-center gap-4">
              <button
                onClick={() => testMutation.mutate(testText)}
                disabled={!testText || testMutation.isPending}
                className="px-4 py-2 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 disabled:opacity-50 transition-colors"
              >
                {testMutation.isPending ? 'Bezig...' : 'Test Signalen'}
              </button>
              {testResult && (
                <span className="text-white/60 text-sm">
                  {testResult.line_count} regels, {testResult.text_length} karakters
                </span>
              )}
            </div>
            {testResult && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {Object.entries(testResult.signals).map(([key, value]) => {
                  const signal = signals.find(s => s.key === key);
                  const isTrue = value === true || (typeof value === 'number' && value > 0);
                  return (
                    <div
                      key={key}
                      className={`p-3 rounded-lg border ${
                        isTrue
                          ? 'bg-green-500/10 border-green-500/30'
                          : 'bg-white/5 border-white/10'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-white text-sm font-medium">{signal?.label || key}</span>
                        <span className={`font-mono text-sm ${isTrue ? 'text-green-400' : 'text-white/50'}`}>
                          {String(value)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create Form */}
      {showCreateForm && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Nieuw Signaal Maken</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-white/70 text-sm mb-1">Key (unieke identifier)</label>
              <input
                type="text"
                value={formKey}
                onChange={(e) => setFormKey(e.target.value)}
                placeholder="bijv. heeft_btw_nummer"
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-white/70 text-sm mb-1">Label</label>
              <input
                type="text"
                value={formLabel}
                onChange={(e) => setFormLabel(e.target.value)}
                placeholder="bijv. Heeft BTW-nummer"
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block text-white/70 text-sm mb-1">Beschrijving (optioneel)</label>
              <input
                type="text"
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
                placeholder="Korte uitleg wat dit signaal detecteert"
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-white/70 text-sm mb-1">Type</label>
              <select
                value={formSignalType}
                onChange={(e) => setFormSignalType(e.target.value as 'boolean' | 'count')}
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              >
                <option value="boolean" className="bg-gray-800">Boolean (ja/nee)</option>
                <option value="count" className="bg-gray-800">Count (aantal)</option>
              </select>
            </div>
            <div>
              <label className="block text-white/70 text-sm mb-1">Berekeningswijze</label>
              <select
                value={formComputeKind}
                onChange={(e) => setFormComputeKind(e.target.value as 'keyword_set' | 'regex_set')}
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              >
                <option value="keyword_set" className="bg-gray-800">Zoekwoorden</option>
                <option value="regex_set" className="bg-gray-800">Regex patronen</option>
              </select>
            </div>
            <div>
              <label className="block text-white/70 text-sm mb-1">Match Mode</label>
              <select
                value={formMatchMode}
                onChange={(e) => setFormMatchMode(e.target.value as 'any' | 'all')}
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              >
                <option value="any" className="bg-gray-800">Één van (OR)</option>
                <option value="all" className="bg-gray-800">Allemaal (AND)</option>
              </select>
            </div>
            <div className="md:col-span-2">
              {formComputeKind === 'keyword_set' ? (
                <>
                  <label className="block text-white/70 text-sm mb-1">Zoekwoorden (één per regel)</label>
                  <textarea
                    value={formKeywords}
                    onChange={(e) => setFormKeywords(e.target.value)}
                    placeholder="factuur&#10;factuurnummer&#10;btw"
                    className="w-full h-32 px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white font-mono text-sm resize-y"
                  />
                </>
              ) : (
                <>
                  <label className="block text-white/70 text-sm mb-1">Regex patronen (één per regel)</label>
                  <textarea
                    value={formPatterns}
                    onChange={(e) => setFormPatterns(e.target.value)}
                    placeholder="NL\d{2}[A-Z]{4}\d{10}&#10;BTW.*nummer"
                    className="w-full h-32 px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white font-mono text-sm resize-y"
                  />
                </>
              )}
            </div>
          </div>
          <div className="flex justify-end gap-3 mt-4">
            <button
              onClick={() => { resetForm(); setShowCreateForm(false); }}
              className="px-4 py-2 bg-white/10 text-white/70 rounded-lg hover:bg-white/20 transition-colors"
            >
              Annuleren
            </button>
            <button
              onClick={handleCreate}
              disabled={!formKey || !formLabel || createMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-500 disabled:opacity-50 transition-colors"
            >
              <FontAwesomeIcon icon={faSave} />
              {createMutation.isPending ? 'Opslaan...' : 'Opslaan'}
            </button>
          </div>
        </div>
      )}

      {/* Edit Form */}
      {editingSignal && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-white mb-4">
            Signaal Bewerken: {editingSignal.label}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-white/70 text-sm mb-1">Label</label>
              <input
                type="text"
                value={formLabel}
                onChange={(e) => setFormLabel(e.target.value)}
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              />
            </div>
            <div>
              <label className="block text-white/70 text-sm mb-1">Match Mode</label>
              <select
                value={formMatchMode}
                onChange={(e) => setFormMatchMode(e.target.value as 'any' | 'all')}
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              >
                <option value="any" className="bg-gray-800">Één van (OR)</option>
                <option value="all" className="bg-gray-800">Allemaal (AND)</option>
              </select>
            </div>
            <div className="md:col-span-2">
              <label className="block text-white/70 text-sm mb-1">Beschrijving</label>
              <input
                type="text"
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
                className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white"
              />
            </div>
            <div className="md:col-span-2">
              {editingSignal.compute_kind === 'keyword_set' ? (
                <>
                  <label className="block text-white/70 text-sm mb-1">Zoekwoorden (één per regel)</label>
                  <textarea
                    value={formKeywords}
                    onChange={(e) => setFormKeywords(e.target.value)}
                    className="w-full h-32 px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white font-mono text-sm resize-y"
                  />
                </>
              ) : (
                <>
                  <label className="block text-white/70 text-sm mb-1">Regex patronen (één per regel)</label>
                  <textarea
                    value={formPatterns}
                    onChange={(e) => setFormPatterns(e.target.value)}
                    className="w-full h-32 px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white font-mono text-sm resize-y"
                  />
                </>
              )}
            </div>
          </div>
          <div className="flex justify-end gap-3 mt-4">
            <button
              onClick={() => { resetForm(); setEditingSignal(null); }}
              className="px-4 py-2 bg-white/10 text-white/70 rounded-lg hover:bg-white/20 transition-colors"
            >
              Annuleren
            </button>
            <button
              onClick={handleUpdate}
              disabled={!formLabel || updateMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-500 disabled:opacity-50 transition-colors"
            >
              <FontAwesomeIcon icon={faSave} />
              {updateMutation.isPending ? 'Opslaan...' : 'Opslaan'}
            </button>
          </div>
        </div>
      )}

      {/* Built-in Signals */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faCog} className="text-blue-400" />
          Ingebouwde Signalen
          <span className="text-sm font-normal text-white/50">({builtinSignals.length})</span>
        </h2>
        <p className="text-white/60 text-sm mb-4">
          Generieke signalen die automatisch worden berekend. Kunnen niet worden aangepast of verwijderd.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {builtinSignals.map((signal) => (
            <div
              key={signal.key}
              className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-blue-300 font-medium">{signal.label}</span>
                <span className={`text-xs px-2 py-0.5 rounded ${
                  signal.signal_type === 'boolean'
                    ? 'bg-purple-500/20 text-purple-300'
                    : 'bg-amber-500/20 text-amber-300'
                }`}>
                  {signal.signal_type}
                </span>
              </div>
              <p className="text-white/60 text-sm">{signal.description}</p>
              <code className="text-xs text-white/40 mt-2 block">{signal.key}</code>
            </div>
          ))}
        </div>
      </div>

      {/* User-defined Signals */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <FontAwesomeIcon icon={faCode} className="text-teal-400" />
          Eigen Signalen
          <span className="text-sm font-normal text-white/50">({userSignals.length})</span>
        </h2>
        <p className="text-white/60 text-sm mb-4">
          Maak signalen voor jouw specifieke documenttypes met zoekwoorden (bijv. &quot;factuur&quot;, &quot;BTW&quot;) of technische regex patronen.
        </p>
        {userSignals.length === 0 ? (
          <div className="text-center py-8 border border-dashed border-white/20 rounded-xl">
            <p className="text-white/40">Nog geen eigen signalen gemaakt</p>
            <button
              onClick={() => { resetForm(); setShowCreateForm(true); }}
              className="mt-3 text-teal-400 hover:text-teal-300"
            >
              + Maak je eerste signaal
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {userSignals.map((signal) => (
              <div
                key={signal.key}
                className="p-4 bg-white/5 border border-white/10 rounded-xl hover:border-white/20 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-white font-medium">{signal.label}</span>
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        signal.signal_type === 'boolean'
                          ? 'bg-purple-500/20 text-purple-300'
                          : 'bg-amber-500/20 text-amber-300'
                      }`}>
                        {signal.signal_type}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded bg-white/10 text-white/60">
                        {signal.compute_kind === 'keyword_set' ? 'Zoekwoorden' : 'Regex'}
                      </span>
                      {signal.config_json?.match_mode === 'all' && (
                        <span className="text-xs px-2 py-0.5 rounded bg-orange-500/20 text-orange-300">
                          AND
                        </span>
                      )}
                    </div>
                    {signal.description && (
                      <p className="text-white/60 text-sm mb-2">{signal.description}</p>
                    )}
                    <div className="flex flex-wrap gap-1 mt-2">
                      {signal.compute_kind === 'keyword_set' && signal.config_json?.keywords?.slice(0, 5).map((kw, i) => (
                        <span key={i} className="text-xs px-2 py-0.5 bg-white/10 rounded text-white/70">
                          {kw}
                        </span>
                      ))}
                      {signal.compute_kind === 'regex_set' && signal.config_json?.patterns?.slice(0, 3).map((p, i) => (
                        <span key={i} className="text-xs px-2 py-0.5 bg-white/10 rounded text-white/70 font-mono">
                          {p.length > 30 ? p.substring(0, 30) + '...' : p}
                        </span>
                      ))}
                      {((signal.config_json?.keywords?.length || 0) > 5 || (signal.config_json?.patterns?.length || 0) > 3) && (
                        <span className="text-xs text-white/40">
                          +{(signal.config_json?.keywords?.length || signal.config_json?.patterns?.length || 0) - (signal.compute_kind === 'keyword_set' ? 5 : 3)} meer
                        </span>
                      )}
                    </div>
                    <code className="text-xs text-white/40 mt-2 block">{signal.key}</code>
                  </div>
                  <div className="flex gap-2 ml-4">
                    <button
                      onClick={() => startEdit(signal)}
                      className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                    >
                      <FontAwesomeIcon icon={faEdit} />
                    </button>
                    <button
                      onClick={() => {
                        if (confirm(`Signaal "${signal.label}" verwijderen?`)) {
                          deleteMutation.mutate(signal.key);
                        }
                      }}
                      className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
                    >
                      <FontAwesomeIcon icon={faTrash} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
