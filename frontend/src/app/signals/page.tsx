'use client';

import { useState, useEffect, useRef } from 'react';
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

  // Auto-test on text change with debounce
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!showTestPanel) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (testText.trim().length < 15) { setTestResult(null); return; }
    debounceRef.current = setTimeout(() => {
      testMutation.mutate(testText);
    }, 500);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [testText, showTestPanel]);

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
        <div className="animate-spin w-8 h-8 border-2 border-slate-300 border-t-slate-600 rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Signalen</h1>
          <p className="text-slate-400 mt-1 text-sm max-w-lg leading-relaxed">
            Signalen zijn zoekregels die het systeem gebruikt om fraude-indicatoren op te sporen.
            Elk signaal zoekt naar specifieke woorden of patronen in documenten en verhoogt de risicoscore als ze gevonden worden.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Uitleg / Testen: exclusive toggle pill */}
          <div className="flex items-center bg-slate-100 border border-slate-200 rounded-xl p-1 gap-1">
            <button
              onClick={() => { setShowHelp(!showHelp); if (!showHelp) setShowTestPanel(false); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                showHelp
                  ? 'bg-amber-400 text-white shadow-sm shadow-amber-200'
                  : 'text-slate-600 hover:text-slate-800 hover:bg-white'
              }`}
            >
              <FontAwesomeIcon icon={faQuestionCircle} className="w-3.5 h-3.5" />
              Uitleg
            </button>
            <button
              onClick={() => { setShowTestPanel(!showTestPanel); if (!showTestPanel) setShowHelp(false); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                showTestPanel
                  ? 'bg-purple-500 text-white shadow-sm shadow-purple-200'
                  : 'text-slate-600 hover:text-slate-800 hover:bg-white'
              }`}
            >
              <FontAwesomeIcon icon={faFlask} className="w-3.5 h-3.5" />
              Testen
            </button>
          </div>
          <button
            onClick={() => { resetForm(); setShowCreateForm(true); }}
            className="flex items-center gap-2 px-4 py-2 sm:py-2.5 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-xl hover:from-[#1ab8b8] hover:to-[#e8a8d8] transition-all font-medium shadow-md shadow-[#FFC1F3]/20 cursor-pointer text-sm whitespace-nowrap"
          >
            <FontAwesomeIcon icon={faPlus} className="w-3.5 h-3.5" />
            Nieuw Signaal
          </button>
        </div>
      </div>

      {/* Help Section */}
      {showHelp && (
        <div className="glass-card p-6 border-l-4 border-amber-500/50 bg-gradient-to-r from-amber-500/5 to-transparent">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-amber-100 rounded-xl shrink-0">
              <FontAwesomeIcon icon={faLightbulb} className="text-amber-500 text-xl" />
            </div>
            <div className="flex-1 space-y-5">
              <div>
                <h3 className="text-xl font-bold text-slate-800 mb-3">Wat zijn signalen?</h3>
                <p className="text-slate-700 text-base leading-relaxed">
                  Signalen zijn <strong className="text-amber-700">automatisch berekende eigenschappen</strong> die uit een document worden gehaald. 
                  Ze beantwoorden vragen zoals: &quot;Bevat dit document een IBAN?&quot; of &quot;Hoeveel bedragen staan erin?&quot;
                </p>
                <p className="text-slate-500 text-sm mt-2">
                  Signalen worden gebruikt om te bepalen of een document in aanmerking komt voor een bepaald documenttype.
                </p>
              </div>

              <div>
                <h3 className="text-xl font-bold text-slate-800 mb-3">Hoe gebruik je signalen?</h3>
                <p className="text-slate-700 text-base mb-4 leading-relaxed">
                  Bij elk documenttype stel je <strong className="text-teal-600">vereisten</strong> en <strong className="text-red-600">uitsluitingen</strong> in met signalen:
                </p>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 bg-teal-50 border-2 border-teal-200 rounded-xl">
                    <div className="text-teal-700 font-bold mb-2 flex items-center gap-2">
                      <FontAwesomeIcon icon={faCheckCircle} />
                      Vereisten
                    </div>
                    <p className="text-slate-700 text-sm leading-relaxed">
                      &quot;Een bankafschrift moet <strong>minimaal 5 transactieregels</strong> hebben <strong>én</strong> een <strong>IBAN bevatten</strong>&quot;
                    </p>
                    <p className="text-slate-500 text-xs mt-2">✓ Document moet aan ALLE vereisten voldoen</p>
                  </div>
                  <div className="p-4 bg-red-50 border-2 border-red-200 rounded-xl">
                    <div className="text-red-700 font-bold mb-2 flex items-center gap-2">
                      <FontAwesomeIcon icon={faTimesCircle} />
                      Uitsluitingen
                    </div>
                    <p className="text-slate-700 text-sm leading-relaxed">
                      &quot;Als het document <strong>factuurtermen bevat</strong>, is het <strong>géén</strong> bankafschrift&quot;
                    </p>
                    <p className="text-slate-500 text-xs mt-2">✗ Document wordt uitgesloten als signaal matcht</p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-slate-800 mb-2">Ingebouwde signalen</h3>
                <p className="text-slate-500 mb-3">
                  Deze signalen zijn standaard beschikbaar en werken automatisch voor alle documenten. Ze worden altijd berekend zonder dat je ze hoeft te configureren.
                </p>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
                  <div className="p-3 bg-cyan-50 border border-cyan-200 rounded-lg">
                    <span className="text-cyan-700 font-semibold block mb-1">IBAN aanwezig</span>
                    <p className="text-slate-600 text-xs">Boolean: Detecteert of document een IBAN nummer bevat</p>
                  </div>
                  <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                    <span className="text-emerald-700 font-semibold block mb-1">Aantal datums</span>
                    <p className="text-slate-600 text-xs">Count: Telt datums in formaat DD-MM-YYYY</p>
                  </div>
                  <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                    <span className="text-amber-700 font-semibold block mb-1">Aantal bedragen</span>
                    <p className="text-slate-600 text-xs">Count: Telt geldbedragen (€X.XXX,XX formaat)</p>
                  </div>
                  <div className="p-3 bg-purple-50 border border-purple-200 rounded-lg">
                    <span className="text-purple-700 font-semibold block mb-1">Transactieregels</span>
                    <p className="text-slate-600 text-xs">Count: Regels met zowel datum als bedrag</p>
                  </div>
                  <div className="p-3 bg-pink-50 border border-pink-200 rounded-lg">
                    <span className="text-pink-700 font-semibold block mb-1">Aantal regels</span>
                    <p className="text-slate-600 text-xs">Count: Totaal aantal niet-lege regels</p>
                  </div>
                  <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                    <span className="text-indigo-700 font-semibold block mb-1">Aantal woorden</span>
                    <p className="text-slate-600 text-xs">Count: Totaal aantal woorden (tokens)</p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-xl font-bold text-slate-800 mb-3">Eigen signalen maken</h3>
                <p className="text-slate-700 text-base mb-3 leading-relaxed">
                  Maak aangepaste signalen voor jouw specifieke documenttypes. Je kunt kiezen tussen:
                </p>
                <div className="grid md:grid-cols-2 gap-3 mb-3">
                  <div className="p-3 bg-purple-50 border border-purple-200 rounded-lg">
                    <div className="text-purple-700 font-semibold mb-1">Zoekwoorden</div>
                    <p className="text-slate-600 text-sm">
                      Zoek naar specifieke woorden zoals <code className="bg-white/80 px-1 rounded">factuur</code>, <code className="bg-white/80 px-1 rounded">BTW</code>, <code className="bg-white/80 px-1 rounded">vervaldatum</code>
                    </p>
                  </div>
                  <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                    <div className="text-orange-700 font-semibold mb-1">Regex patronen</div>
                    <p className="text-slate-600 text-sm">
                      Geavanceerde patronen zoals <code className="bg-white/80 px-1 rounded font-mono">NL\d{2}[A-Z]{4}\d{10}</code> voor IBAN detectie
                    </p>
                  </div>
                </div>
                <p className="text-slate-500 text-sm">
                  <strong>Voorbeeld:</strong> Maak een signaal &quot;Heeft factuurtermen&quot; met zoekwoorden: <em>factuur, BTW, vervaldatum, factuurnummer</em>. 
                  Of &quot;Heeft bankafschrift termen&quot; met: <em>saldo, transactie, rekeningoverzicht, bij- en afschrijvingen</em>.
                </p>
              </div>

              <div className="pt-4 border-t-2 border-slate-300 bg-slate-50 p-4 rounded-lg">
                <p className="text-slate-700 text-sm flex items-start gap-3">
                  <FontAwesomeIcon icon={faFlask} className="text-purple-500 text-lg shrink-0 mt-0.5" />
                  <span>
                    <strong className="text-purple-700">Tip:</strong> Gebruik de &quot;Testen&quot; knop om te zien welke signalen in een voorbeelddocument worden herkend. 
                    Plak voorbeeldtekst en zie direct welke signalen matchen!
                  </span>
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Modal */}
      {showTestPanel && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[150] flex items-center justify-center p-4" onClick={() => { setShowTestPanel(false); setTestText(''); setTestResult(null); }}>
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <FontAwesomeIcon icon={faFlask} className="text-purple-500 w-4 h-4" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-slate-800">Signalen testen</h2>
                  <p className="text-slate-400 text-xs">Resultaten verschijnen automatisch terwijl je typt</p>
                </div>
              </div>
              <button onClick={() => { setShowTestPanel(false); setTestText(''); setTestResult(null); }} className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg">
                <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              <textarea
                autoFocus
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Plak hier voorbeeldtekst..."
                className="w-full h-28 px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 text-sm resize-none focus:outline-none focus:border-purple-400"
              />
              <div className="flex items-center gap-2 min-h-[18px]">
                {testMutation.isPending && (
                  <span className="text-slate-400 text-xs flex items-center gap-1.5">
                    <FontAwesomeIcon icon={faSearch} className="w-3 h-3 animate-pulse" />
                    Analyseren...
                  </span>
                )}
                {testResult && !testMutation.isPending && (
                  <span className="text-slate-400 text-xs">
                    {Object.entries(testResult.signals).filter(([_, v]) => v === true || (typeof v === 'number' && v > 0)).length} van {Object.keys(testResult.signals).length} gematcht
                    <span className="ml-2 text-slate-300">· {testResult.line_count} regels</span>
                  </span>
                )}
                {!testResult && !testMutation.isPending && testText.trim().length > 0 && testText.trim().length < 15 && (
                  <span className="text-slate-300 text-xs">Typ minimaal 15 tekens...</span>
                )}
              </div>
              {testResult && (() => {
                const matched = Object.entries(testResult.signals).filter(([_, v]) => v === true || (typeof v === 'number' && v > 0));
                const unmatched = Object.entries(testResult.signals).filter(([_, v]) => !(v === true || (typeof v === 'number' && v > 0)));
                return (
                  <div className="space-y-4">
                    {matched.length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-green-600 mb-2">Gematcht ({matched.length})</div>
                        <div className="flex flex-wrap gap-2">
                          {matched.map(([key, value]) => {
                            const signal = signals.find(s => s.key === key);
                            return (
                              <div key={key} className="flex items-center gap-1.5 px-3 py-1.5 bg-green-50 border border-green-200 rounded-full text-xs">
                                <FontAwesomeIcon icon={faCheckCircle} className="text-green-500 w-3 h-3 flex-shrink-0" />
                                <span className="text-green-800 font-medium">{signal?.label || key}</span>
                                {typeof value === 'number' && value > 1 && (
                                  <span className="bg-green-200 text-green-700 rounded-full px-1.5 py-0.5 text-[10px] font-bold">{value}×</span>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    {matched.length === 0 && <p className="text-slate-400 text-sm py-2">Geen signalen gematcht in deze tekst</p>}
                    {unmatched.length > 0 && (
                      <div>
                        <div className="text-xs font-medium text-slate-400 mb-2">Niet gematcht ({unmatched.length})</div>
                        <div className="flex flex-wrap gap-1.5">
                          {unmatched.map(([key]) => {
                            const signal = signals.find(s => s.key === key);
                            return (
                              <span key={key} className="px-2 py-1 bg-slate-100 text-slate-400 rounded-full text-xs">
                                {signal?.label || key}
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}

      {/* Create Form */}
      {showCreateForm && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">Nieuw Signaal Maken</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-slate-500 text-sm mb-1">Key (unieke identifier)</label>
              <input
                type="text"
                value={formKey}
                onChange={(e) => setFormKey(e.target.value)}
                placeholder="bijv. heeft_btw_nummer"
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              />
            </div>
            <div>
              <label className="block text-slate-500 text-sm mb-1">Label</label>
              <input
                type="text"
                value={formLabel}
                onChange={(e) => setFormLabel(e.target.value)}
                placeholder="bijv. Heeft BTW-nummer"
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block text-slate-500 text-sm mb-1">Beschrijving (optioneel)</label>
              <input
                type="text"
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
                placeholder="Korte uitleg wat dit signaal detecteert"
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              />
            </div>
            <div>
              <label className="block text-slate-500 text-sm mb-1">Type</label>
              <select
                value={formSignalType}
                onChange={(e) => setFormSignalType(e.target.value as 'boolean' | 'count')}
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              >
                <option value="boolean" className="bg-white">Boolean (ja/nee)</option>
                <option value="count" className="bg-white">Count (aantal)</option>
              </select>
            </div>
            <div>
              <label className="block text-slate-500 text-sm mb-1">Berekeningswijze</label>
              <select
                value={formComputeKind}
                onChange={(e) => setFormComputeKind(e.target.value as 'keyword_set' | 'regex_set')}
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              >
                <option value="keyword_set" className="bg-white">Zoekwoorden</option>
                <option value="regex_set" className="bg-white">Regex patronen</option>
              </select>
            </div>
            <div>
              <label className="block text-slate-500 text-sm mb-1">Match Mode</label>
              <select
                value={formMatchMode}
                onChange={(e) => setFormMatchMode(e.target.value as 'any' | 'all')}
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              >
                <option value="any" className="bg-white">Één van (OR)</option>
                <option value="all" className="bg-white">Allemaal (AND)</option>
              </select>
            </div>
            <div className="md:col-span-2">
              {formComputeKind === 'keyword_set' ? (
                <>
                  <label className="block text-slate-500 text-sm mb-1">Zoekwoorden (één per regel)</label>
                  <textarea
                    value={formKeywords}
                    onChange={(e) => setFormKeywords(e.target.value)}
                    placeholder="factuur&#10;factuurnummer&#10;btw"
                    className="w-full h-32 px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800 font-mono text-sm resize-y"
                  />
                </>
              ) : (
                <>
                  <label className="block text-slate-500 text-sm mb-1">Regex patronen (één per regel)</label>
                  <textarea
                    value={formPatterns}
                    onChange={(e) => setFormPatterns(e.target.value)}
                    placeholder="NL\d{2}[A-Z]{4}\d{10}&#10;BTW.*nummer"
                    className="w-full h-32 px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800 font-mono text-sm resize-y"
                  />
                </>
              )}
            </div>
          </div>
          <div className="flex justify-end gap-3 mt-4">
            <button
              onClick={() => { resetForm(); setShowCreateForm(false); }}
              className="px-4 py-2 bg-slate-100 text-slate-500 rounded-lg hover:bg-slate-200 transition-colors"
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
          <h2 className="text-lg font-semibold text-slate-800 mb-4">
            Signaal Bewerken: {editingSignal.label}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-slate-500 text-sm mb-1">Label</label>
              <input
                type="text"
                value={formLabel}
                onChange={(e) => setFormLabel(e.target.value)}
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              />
            </div>
            <div>
              <label className="block text-slate-500 text-sm mb-1">Match Mode</label>
              <select
                value={formMatchMode}
                onChange={(e) => setFormMatchMode(e.target.value as 'any' | 'all')}
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              >
                <option value="any" className="bg-white">Één van (OR)</option>
                <option value="all" className="bg-white">Allemaal (AND)</option>
              </select>
            </div>
            <div className="md:col-span-2">
              <label className="block text-slate-500 text-sm mb-1">Beschrijving</label>
              <input
                type="text"
                value={formDescription}
                onChange={(e) => setFormDescription(e.target.value)}
                className="w-full px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800"
              />
            </div>
            <div className="md:col-span-2">
              {editingSignal.compute_kind === 'keyword_set' ? (
                <>
                  <label className="block text-slate-500 text-sm mb-1">Zoekwoorden (één per regel)</label>
                  <textarea
                    value={formKeywords}
                    onChange={(e) => setFormKeywords(e.target.value)}
                    className="w-full h-32 px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800 font-mono text-sm resize-y"
                  />
                </>
              ) : (
                <>
                  <label className="block text-slate-500 text-sm mb-1">Regex patronen (één per regel)</label>
                  <textarea
                    value={formPatterns}
                    onChange={(e) => setFormPatterns(e.target.value)}
                    className="w-full h-32 px-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-slate-800 font-mono text-sm resize-y"
                  />
                </>
              )}
            </div>
          </div>
          <div className="flex justify-end gap-3 mt-4">
            <button
              onClick={() => { resetForm(); setEditingSignal(null); }}
              className="px-4 py-2 bg-slate-100 text-slate-500 rounded-lg hover:bg-slate-200 transition-colors"
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
          <div className="p-2 bg-cyan-100 rounded-lg">
            <FontAwesomeIcon icon={faCog} className="text-cyan-600 text-lg" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
              Ingebouwde Signalen
              <span className="text-sm font-normal text-slate-500 bg-slate-100 px-2 py-0.5 rounded">({builtinSignals.length})</span>
            </h2>
            <p className="text-slate-500 text-sm mt-1">
              Generieke signalen die automatisch worden berekend voor elk document. Deze kunnen niet worden aangepast of verwijderd.
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
          {builtinSignals.map((signal, idx) => {
            const colors = [
              { bg: 'bg-cyan-50',   border: 'border-cyan-200',   text: 'text-cyan-700'   },
              { bg: 'bg-emerald-50',border: 'border-emerald-200',text: 'text-emerald-700' },
              { bg: 'bg-amber-50',  border: 'border-amber-200',  text: 'text-amber-700'  },
              { bg: 'bg-purple-50', border: 'border-purple-200', text: 'text-purple-700'  },
              { bg: 'bg-pink-50',   border: 'border-pink-200',   text: 'text-pink-700'   },
              { bg: 'bg-indigo-50', border: 'border-indigo-200', text: 'text-indigo-700'  },
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
                      ? 'bg-purple-100 text-purple-700 border border-purple-200'
                      : 'bg-amber-100 text-amber-700 border border-amber-200'
                  }`}>
                    {signal.signal_type}
                  </span>
                </div>
                <p className="text-slate-700 text-sm mb-2 leading-relaxed">{signal.description}</p>
                <code className="text-xs text-slate-400 mt-2 block font-mono bg-slate-50 px-2 py-1 rounded">{signal.key}</code>
              </div>
            );
          })}
        </div>
      </div>

      {/* User-defined Signals */}
      <div className="glass-card p-6 border-l-4 border-teal-500/50">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-teal-100 rounded-lg">
            <FontAwesomeIcon icon={faCode} className="text-teal-600 text-lg" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
              Eigen Signalen
              <span className="text-sm font-normal text-slate-500 bg-slate-100 px-2 py-0.5 rounded">({userSignals.length})</span>
            </h2>
            <p className="text-slate-500 text-sm mt-1">
              Maak aangepaste signalen voor jouw specifieke documenttypes. Gebruik zoekwoorden (bijv. &quot;factuur&quot;, &quot;BTW&quot;) of regex patronen voor geavanceerde matching.
            </p>
          </div>
        </div>
        {userSignals.length === 0 ? (
          <div className="text-center py-12 border-2 border-dashed border-teal-200 rounded-xl bg-teal-50">
            <FontAwesomeIcon icon={faCode} className="text-teal-400/50 text-4xl mb-3" />
            <p className="text-slate-500 font-medium mb-1">Nog geen eigen signalen gemaakt</p>
            <p className="text-slate-400 text-sm mb-4">Maak je eerste signaal om documenttypes te classificeren</p>
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
                className="p-5 bg-slate-50 border-2 border-teal-200 rounded-xl hover:border-teal-300 hover:bg-slate-100 transition-all shadow-sm"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2 flex-wrap">
                      <span className="text-slate-800 font-semibold text-base">{signal.label}</span>
                      <span className={`text-xs px-2 py-1 rounded font-medium border ${
                        signal.signal_type === 'boolean'
                          ? 'bg-purple-100 text-purple-700 border-purple-200'
                          : 'bg-amber-100 text-amber-700 border-amber-200'
                      }`}>
                        {signal.signal_type}
                      </span>
                      <span className="text-xs px-2 py-1 rounded bg-teal-100 text-teal-700 border border-teal-200 font-medium">
                        {signal.compute_kind === 'keyword_set' ? 'Zoekwoorden' : 'Regex'}
                      </span>
                      {signal.config_json?.match_mode === 'all' && (
                        <span className="text-xs px-2 py-1 rounded bg-orange-100 text-orange-700 border border-orange-200 font-medium">
                          ALL (AND)
                        </span>
                      )}
                      {signal.config_json?.match_mode === 'any' && (
                        <span className="text-xs px-2 py-1 rounded bg-green-100 text-green-700 border border-green-200 font-medium">
                          ANY (OR)
                        </span>
                      )}
                    </div>
                    {signal.description && (
                      <p className="text-slate-600 text-sm mb-3 leading-relaxed">{signal.description}</p>
                    )}
                    <div className="flex flex-wrap gap-2 mt-3">
                      {signal.compute_kind === 'keyword_set' && signal.config_json?.keywords?.slice(0, 5).map((kw, i) => (
                        <span key={i} className="text-xs px-2.5 py-1 bg-slate-200 rounded-md text-slate-800 font-medium border border-slate-300">
                          {kw}
                        </span>
                      ))}
                      {signal.compute_kind === 'regex_set' && signal.config_json?.patterns?.slice(0, 3).map((p, i) => (
                        <span key={i} className="text-xs px-2.5 py-1 bg-slate-200 rounded-md text-slate-700 font-mono border border-slate-300">
                          {p.length > 30 ? p.substring(0, 30) + '...' : p}
                        </span>
                      ))}
                      {((signal.config_json?.keywords?.length || 0) > 5 || (signal.config_json?.patterns?.length || 0) > 3) && (
                        <span className="text-xs text-slate-500 px-2.5 py-1 bg-slate-100 rounded-md border border-slate-300">
                          +{(signal.config_json?.keywords?.length || signal.config_json?.patterns?.length || 0) - (signal.compute_kind === 'keyword_set' ? 5 : 3)} meer
                        </span>
                      )}
                    </div>
                    <code className="text-xs text-slate-400 mt-3 block font-mono bg-white px-2 py-1 rounded border border-slate-200">{signal.key}</code>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <button
                      onClick={() => startEdit(signal)}
                      className="p-2.5 text-teal-600 hover:text-teal-700 hover:bg-teal-100 rounded-lg transition-colors border border-teal-200"
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
                      className="p-2.5 text-red-500 hover:text-red-700 hover:bg-red-100 rounded-lg transition-colors border border-red-200"
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
