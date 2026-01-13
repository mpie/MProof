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
        <div className="glass-card p-6 border-l-4 border-amber-500/50 bg-gradient-to-r from-amber-500/5 to-transparent">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-amber-500/20 rounded-xl shrink-0">
              <FontAwesomeIcon icon={faLightbulb} className="text-amber-300 text-xl" />
            </div>
            <div className="flex-1 space-y-5">
              <div>
                <h3 className="text-xl font-bold text-white mb-3">Wat zijn signalen?</h3>
                <p className="text-white/90 text-base leading-relaxed">
                  Signalen zijn <strong className="text-amber-300">automatisch berekende eigenschappen</strong> die uit een document worden gehaald. 
                  Ze beantwoorden vragen zoals: &quot;Bevat dit document een IBAN?&quot; of &quot;Hoeveel bedragen staan erin?&quot;
                </p>
                <p className="text-white/70 text-sm mt-2">
                  Signalen worden gebruikt om te bepalen of een document in aanmerking komt voor een bepaald documenttype.
                </p>
              </div>

              <div>
                <h3 className="text-xl font-bold text-white mb-3">Hoe gebruik je signalen?</h3>
                <p className="text-white/90 text-base mb-4 leading-relaxed">
                  Bij elk documenttype stel je <strong className="text-teal-300">vereisten</strong> en <strong className="text-red-300">uitsluitingen</strong> in met signalen:
                </p>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 bg-teal-500/20 border-2 border-teal-500/40 rounded-xl">
                    <div className="text-teal-200 font-bold mb-2 flex items-center gap-2">
                      <FontAwesomeIcon icon={faCheckCircle} />
                      Vereisten
                    </div>
                    <p className="text-white/90 text-sm leading-relaxed">
                      &quot;Een bankafschrift moet <strong>minimaal 5 transactieregels</strong> hebben <strong>én</strong> een <strong>IBAN bevatten</strong>&quot;
                    </p>
                    <p className="text-white/60 text-xs mt-2">✓ Document moet aan ALLE vereisten voldoen</p>
                  </div>
                  <div className="p-4 bg-red-500/20 border-2 border-red-500/40 rounded-xl">
                    <div className="text-red-200 font-bold mb-2 flex items-center gap-2">
                      <FontAwesomeIcon icon={faTimesCircle} />
                      Uitsluitingen
                    </div>
                    <p className="text-white/90 text-sm leading-relaxed">
                      &quot;Als het document <strong>factuurtermen bevat</strong>, is het <strong>géén</strong> bankafschrift&quot;
                    </p>
                    <p className="text-white/60 text-xs mt-2">✗ Document wordt uitgesloten als signaal matcht</p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Ingebouwde signalen</h3>
                <p className="text-white/70 mb-3">
                  Deze signalen zijn standaard beschikbaar en werken automatisch voor alle documenten. Ze worden altijd berekend zonder dat je ze hoeft te configureren.
                </p>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
                  <div className="p-3 bg-cyan-500/20 border border-cyan-500/40 rounded-lg">
                    <span className="text-cyan-200 font-semibold block mb-1">IBAN aanwezig</span>
                    <p className="text-white/80 text-xs">Boolean: Detecteert of document een IBAN nummer bevat</p>
                  </div>
                  <div className="p-3 bg-emerald-500/20 border border-emerald-500/40 rounded-lg">
                    <span className="text-emerald-200 font-semibold block mb-1">Aantal datums</span>
                    <p className="text-white/80 text-xs">Count: Telt datums in formaat DD-MM-YYYY</p>
                  </div>
                  <div className="p-3 bg-amber-500/20 border border-amber-500/40 rounded-lg">
                    <span className="text-amber-200 font-semibold block mb-1">Aantal bedragen</span>
                    <p className="text-white/80 text-xs">Count: Telt geldbedragen (€X.XXX,XX formaat)</p>
                  </div>
                  <div className="p-3 bg-purple-500/20 border border-purple-500/40 rounded-lg">
                    <span className="text-purple-200 font-semibold block mb-1">Transactieregels</span>
                    <p className="text-white/80 text-xs">Count: Regels met zowel datum als bedrag</p>
                  </div>
                  <div className="p-3 bg-pink-500/20 border border-pink-500/40 rounded-lg">
                    <span className="text-pink-200 font-semibold block mb-1">Aantal regels</span>
                    <p className="text-white/80 text-xs">Count: Totaal aantal niet-lege regels</p>
                  </div>
                  <div className="p-3 bg-indigo-500/20 border border-indigo-500/40 rounded-lg">
                    <span className="text-indigo-200 font-semibold block mb-1">Aantal woorden</span>
                    <p className="text-white/80 text-xs">Count: Totaal aantal woorden (tokens)</p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-xl font-bold text-white mb-3">Eigen signalen maken</h3>
                <p className="text-white/90 text-base mb-3 leading-relaxed">
                  Maak aangepaste signalen voor jouw specifieke documenttypes. Je kunt kiezen tussen:
                </p>
                <div className="grid md:grid-cols-2 gap-3 mb-3">
                  <div className="p-3 bg-purple-500/20 border border-purple-500/40 rounded-lg">
                    <div className="text-purple-200 font-semibold mb-1">Zoekwoorden</div>
                    <p className="text-white/80 text-sm">
                      Zoek naar specifieke woorden zoals <code className="bg-black/30 px-1 rounded">factuur</code>, <code className="bg-black/30 px-1 rounded">BTW</code>, <code className="bg-black/30 px-1 rounded">vervaldatum</code>
                    </p>
                  </div>
                  <div className="p-3 bg-orange-500/20 border border-orange-500/40 rounded-lg">
                    <div className="text-orange-200 font-semibold mb-1">Regex patronen</div>
                    <p className="text-white/80 text-sm">
                      Geavanceerde patronen zoals <code className="bg-black/30 px-1 rounded font-mono">NL\d{2}[A-Z]{4}\d{10}</code> voor IBAN detectie
                    </p>
                  </div>
                </div>
                <p className="text-white/70 text-sm">
                  <strong>Voorbeeld:</strong> Maak een signaal &quot;Heeft factuurtermen&quot; met zoekwoorden: <em>factuur, BTW, vervaldatum, factuurnummer</em>. 
                  Of &quot;Heeft bankafschrift termen&quot; met: <em>saldo, transactie, rekeningoverzicht, bij- en afschrijvingen</em>.
                </p>
              </div>

              <div className="pt-4 border-t-2 border-white/20 bg-white/5 p-4 rounded-lg">
                <p className="text-white/90 text-sm flex items-start gap-3">
                  <FontAwesomeIcon icon={faFlask} className="text-purple-300 text-lg shrink-0 mt-0.5" />
                  <span>
                    <strong className="text-purple-200">Tip:</strong> Gebruik de &quot;Testen&quot; knop om te zien welke signalen in een voorbeelddocument worden herkend. 
                    Plak voorbeeldtekst en zie direct welke signalen matchen!
                  </span>
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Panel */}
      {showTestPanel && (
        <div className="glass-card p-6 border-l-4 border-purple-500/50">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <FontAwesomeIcon icon={faFlask} className="text-purple-300 text-lg" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Signalen Testen</h2>
              <p className="text-white/70 text-sm mt-1">Plak voorbeeldtekst om te zien welke signalen worden herkend</p>
            </div>
          </div>
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
              <div className="space-y-3">
                <div className="text-sm text-white/70 mb-2">
                  Resultaten: {Object.entries(testResult.signals).filter(([_, v]) => v === true || (typeof v === 'number' && v > 0)).length} van {Object.keys(testResult.signals).length} signalen gematcht
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {Object.entries(testResult.signals).map(([key, value]) => {
                    const signal = signals.find(s => s.key === key);
                    const isTrue = value === true || (typeof value === 'number' && value > 0);
                    return (
                      <div
                        key={key}
                        className={`p-4 rounded-xl border-2 transition-all ${
                          isTrue
                            ? 'bg-green-500/20 border-green-500/50 shadow-lg'
                            : 'bg-white/10 border-white/20'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-sm font-semibold ${isTrue ? 'text-green-200' : 'text-white/70'}`}>
                            {signal?.label || key}
                          </span>
                          <span className={`font-mono text-base font-bold px-2 py-1 rounded ${
                            isTrue 
                              ? 'bg-green-500/30 text-green-200 border border-green-500/50' 
                              : 'bg-white/10 text-white/50 border border-white/20'
                          }`}>
                            {String(value)}
                          </span>
                        </div>
                        {signal?.description && (
                          <p className="text-xs text-white/60 mt-1">{signal.description}</p>
                        )}
                      </div>
                    );
                  })}
                </div>
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
      <div className="glass-card p-6 border-l-4 border-cyan-500/50">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-cyan-500/20 rounded-lg">
            <FontAwesomeIcon icon={faCog} className="text-cyan-300 text-lg" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              Ingebouwde Signalen
              <span className="text-sm font-normal text-white/60 bg-white/10 px-2 py-0.5 rounded">({builtinSignals.length})</span>
            </h2>
            <p className="text-white/70 text-sm mt-1">
              Generieke signalen die automatisch worden berekend voor elk document. Deze kunnen niet worden aangepast of verwijderd.
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
          {builtinSignals.map((signal, idx) => {
            const colors = [
              { bg: 'bg-cyan-500/20', border: 'border-cyan-500/40', text: 'text-cyan-200' },
              { bg: 'bg-emerald-500/20', border: 'border-emerald-500/40', text: 'text-emerald-200' },
              { bg: 'bg-amber-500/20', border: 'border-amber-500/40', text: 'text-amber-200' },
              { bg: 'bg-purple-500/20', border: 'border-purple-500/40', text: 'text-purple-200' },
              { bg: 'bg-pink-500/20', border: 'border-pink-500/40', text: 'text-pink-200' },
              { bg: 'bg-indigo-500/20', border: 'border-indigo-500/40', text: 'text-indigo-200' },
            ];
            const color = colors[idx % colors.length];
            return (
              <div
                key={signal.key}
                className={`p-4 ${color.bg} border-2 ${color.border} rounded-xl hover:shadow-lg transition-shadow`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className={`${color.text} font-semibold text-base`}>{signal.label}</span>
                  <span className={`text-xs px-2 py-1 rounded font-medium ${
                    signal.signal_type === 'boolean'
                      ? 'bg-purple-500/30 text-purple-200 border border-purple-500/50'
                      : 'bg-amber-500/30 text-amber-200 border border-amber-500/50'
                  }`}>
                    {signal.signal_type}
                  </span>
                </div>
                <p className="text-white/90 text-sm mb-2 leading-relaxed">{signal.description}</p>
                <code className="text-xs text-white/50 mt-2 block font-mono bg-black/20 px-2 py-1 rounded">{signal.key}</code>
              </div>
            );
          })}
        </div>
      </div>

      {/* User-defined Signals */}
      <div className="glass-card p-6 border-l-4 border-teal-500/50">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-teal-500/20 rounded-lg">
            <FontAwesomeIcon icon={faCode} className="text-teal-300 text-lg" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              Eigen Signalen
              <span className="text-sm font-normal text-white/60 bg-white/10 px-2 py-0.5 rounded">({userSignals.length})</span>
            </h2>
            <p className="text-white/70 text-sm mt-1">
              Maak aangepaste signalen voor jouw specifieke documenttypes. Gebruik zoekwoorden (bijv. &quot;factuur&quot;, &quot;BTW&quot;) of regex patronen voor geavanceerde matching.
            </p>
          </div>
        </div>
        {userSignals.length === 0 ? (
          <div className="text-center py-12 border-2 border-dashed border-teal-500/30 rounded-xl bg-teal-500/5">
            <FontAwesomeIcon icon={faCode} className="text-teal-400/50 text-4xl mb-3" />
            <p className="text-white/70 font-medium mb-1">Nog geen eigen signalen gemaakt</p>
            <p className="text-white/50 text-sm mb-4">Maak je eerste signaal om documenttypes te classificeren</p>
            <button
              onClick={() => { resetForm(); setShowCreateForm(true); }}
              className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-500 transition-colors font-medium"
            >
              + Maak je eerste signaal
            </button>
          </div>
        ) : (
          <div className="space-y-3 mt-4">
            {userSignals.map((signal) => (
              <div
                key={signal.key}
                className="p-5 bg-white/10 border-2 border-teal-500/30 rounded-xl hover:border-teal-500/50 hover:bg-white/15 transition-all shadow-lg"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      <span className="text-white font-semibold text-base">{signal.label}</span>
                      <span className={`text-xs px-2 py-1 rounded font-medium border ${
                        signal.signal_type === 'boolean'
                          ? 'bg-purple-500/30 text-purple-200 border-purple-500/50'
                          : 'bg-amber-500/30 text-amber-200 border-amber-500/50'
                      }`}>
                        {signal.signal_type}
                      </span>
                      <span className="text-xs px-2 py-1 rounded bg-teal-500/30 text-teal-200 border border-teal-500/50 font-medium">
                        {signal.compute_kind === 'keyword_set' ? 'Zoekwoorden' : 'Regex'}
                      </span>
                      {signal.config_json?.match_mode === 'all' && (
                        <span className="text-xs px-2 py-1 rounded bg-orange-500/30 text-orange-200 border border-orange-500/50 font-medium">
                          ALL (AND)
                        </span>
                      )}
                      {signal.config_json?.match_mode === 'any' && (
                        <span className="text-xs px-2 py-1 rounded bg-green-500/30 text-green-200 border border-green-500/50 font-medium">
                          ANY (OR)
                        </span>
                      )}
                    </div>
                    {signal.description && (
                      <p className="text-white/80 text-sm mb-3 leading-relaxed">{signal.description}</p>
                    )}
                    <div className="flex flex-wrap gap-2 mt-3">
                      {signal.compute_kind === 'keyword_set' && signal.config_json?.keywords?.slice(0, 5).map((kw, i) => (
                        <span key={i} className="text-xs px-2.5 py-1 bg-white/20 rounded-md text-white font-medium border border-white/30">
                          {kw}
                        </span>
                      ))}
                      {signal.compute_kind === 'regex_set' && signal.config_json?.patterns?.slice(0, 3).map((p, i) => (
                        <span key={i} className="text-xs px-2.5 py-1 bg-white/20 rounded-md text-white/90 font-mono border border-white/30">
                          {p.length > 30 ? p.substring(0, 30) + '...' : p}
                        </span>
                      ))}
                      {((signal.config_json?.keywords?.length || 0) > 5 || (signal.config_json?.patterns?.length || 0) > 3) && (
                        <span className="text-xs text-white/60 px-2.5 py-1 bg-white/10 rounded-md border border-white/20">
                          +{(signal.config_json?.keywords?.length || signal.config_json?.patterns?.length || 0) - (signal.compute_kind === 'keyword_set' ? 5 : 3)} meer
                        </span>
                      )}
                    </div>
                    <code className="text-xs text-white/50 mt-3 block font-mono bg-black/30 px-2 py-1 rounded border border-white/10">{signal.key}</code>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <button
                      onClick={() => startEdit(signal)}
                      className="p-2.5 text-teal-300 hover:text-teal-200 hover:bg-teal-500/20 rounded-lg transition-colors border border-teal-500/30"
                      title="Bewerken"
                    >
                      <FontAwesomeIcon icon={faEdit} />
                    </button>
                    <button
                      onClick={() => {
                        if (confirm(`Signaal "${signal.label}" verwijderen?`)) {
                          deleteMutation.mutate(signal.key);
                        }
                      }}
                      className="p-2.5 text-red-400 hover:text-red-300 hover:bg-red-500/20 rounded-lg transition-colors border border-red-500/30"
                      title="Verwijderen"
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
