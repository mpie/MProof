'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearchPlus, faSearchMinus, faExpand, faChevronLeft, faChevronRight, faHighlighter, faEyeSlash, faChevronDown, faChevronUp } from '@fortawesome/free-solid-svg-icons';

interface EvidenceItem {
  page: number;
  start: number;
  end: number;
  quote: string;
}

interface EvidenceQuote extends EvidenceItem {
  fieldName: string;
}

interface PageEvidence {
  items: EvidenceQuote[];
}

interface PDFViewerWithHighlightsProps {
  url: string;
  evidence?: Record<string, EvidenceItem[]>;
  onClose?: () => void;
}

interface HighlightStyle { bg: string; shadow: string; border: string; cardBg: string; text: string; label: string }

const HIGHLIGHT_STYLES: Record<string, HighlightStyle> = {
  amount:  { bg: 'rgba(34,197,94,0.35)',   shadow: '0 0 0 1px rgba(34,197,94,0.8)',   border: '#16a34a', cardBg: '#dcfce7', text: '#14532d', label: '#15803d' },
  date:    { bg: 'rgba(99,102,241,0.35)',  shadow: '0 0 0 1px rgba(99,102,241,0.8)',  border: '#4f46e5', cardBg: '#e0e7ff', text: '#1e1b4b', label: '#4338ca' },
  email:   { bg: 'rgba(249,115,22,0.35)',  shadow: '0 0 0 1px rgba(249,115,22,0.8)',  border: '#ea580c', cardBg: '#ffedd5', text: '#7c2d12', label: '#c2410c' },
  name:    { bg: 'rgba(236,72,153,0.35)',  shadow: '0 0 0 1px rgba(236,72,153,0.8)',  border: '#db2777', cardBg: '#fce7f3', text: '#831843', label: '#be185d' },
  address: { bg: 'rgba(139,92,246,0.35)',  shadow: '0 0 0 1px rgba(139,92,246,0.8)',  border: '#7c3aed', cardBg: '#ede9fe', text: '#2e1065', label: '#6d28d9' },
  default: { bg: 'rgba(250,204,21,0.45)',  shadow: '0 0 0 1px rgba(234,179,8,0.8)',   border: '#ca8a04', cardBg: '#fef9c3', text: '#713f12', label: '#a16207' },
};

function getHighlightStyle(fieldName: string, quote: string): HighlightStyle {
  const f = fieldName.toLowerCase();
  const q = quote;
  if (/@/.test(q) || /telefoon|phone|tel\b|gsm|mobiel|email|e-mail|kantoor|contact/.test(f))
    return HIGHLIGHT_STYLES.email;
  if (/[€$£¥]|\d{1,3}[.,]\d{3}/.test(q) || /waarde|bedrag|prijs|huur|koop|kosten|rente|markt|lening|schuld|rente|inkomen|salaris/.test(f))
    return HIGHLIGHT_STYLES.amount;
  if (/\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}/.test(q) || /datum|date|peildatum|looptijd/.test(f))
    return HIGHLIGHT_STYLES.date;
  if (/naam|name|persoon|eigenaar|verkoper|koper|adviseur|notaris|taxateur|opdrachtgever/.test(f))
    return HIGHLIGHT_STYLES.name;
  if (/adres|address|straat|locatie|object|postcode|gemeente|stad|wijk/.test(f))
    return HIGHLIGHT_STYLES.address;
  return HIGHLIGHT_STYLES.default;
}

export function PDFViewerWithHighlights({ url, evidence = {} }: PDFViewerWithHighlightsProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [scale, setScale] = useState<number>(1.0);
  const [pageWidth, setPageWidth] = useState<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const [pageEvidence, setPageEvidence] = useState<Map<number, PageEvidence>>(new Map());
  const [showHighlights, setShowHighlights] = useState<boolean>(true);
  const [showEvidenceOverlay, setShowEvidenceOverlay] = useState<boolean>(true);
  const [didJumpToEvidence, setDidJumpToEvidence] = useState<boolean>(false);
  // Items per page that were actually found and highlighted (subset of pageEvidence)
  const [matchedItems, setMatchedItems] = useState<Map<number, EvidenceQuote[]>>(new Map());

  // Set up PDF.js worker on client side only
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Use jsdelivr CDN which is more reliable than unpkg
      pdfjs.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
    }
  }, []);

  // Collect evidence per page. The UI highlights by quote text, not by OCR offsets.
  useEffect(() => {
    const evidencePerPage = new Map<number, PageEvidence>();
    
    Object.entries(evidence).forEach(([fieldName, items]) => {
      const evidenceItems = Array.isArray(items) ? items : items ? [items as EvidenceItem] : [];
      if (evidenceItems.length > 0) {
        evidenceItems.forEach(item => {
          if (item.quote && item.page !== undefined && item.page !== null) {
            // Handle both number and string page values, convert 0-indexed to 1-indexed
            const rawPage = typeof item.page === 'string' ? parseInt(item.page, 10) : item.page;
            const pageNum = rawPage + 1;
            const evidenceItem = { ...item, fieldName };
            
            const existing = evidencePerPage.get(pageNum);
            if (existing) {
              existing.items.push(evidenceItem);
            } else {
              evidencePerPage.set(pageNum, {
                items: [evidenceItem],
              });
            }
          }
        });
      }
    });

    setPageEvidence(evidencePerPage);
    setMatchedItems(new Map());
    setDidJumpToEvidence(false);
  }, [evidence]);

  useEffect(() => {
    if (didJumpToEvidence || pageEvidence.size === 0) return;

    const firstEvidencePage = Array.from(pageEvidence.keys()).sort((a, b) => a - b)[0];
    if (firstEvidencePage && firstEvidencePage !== currentPage) {
      setCurrentPage(firstEvidencePage);
    }
    setDidJumpToEvidence(true);
  }, [currentPage, didJumpToEvidence, pageEvidence]);

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
  }, []);

  // Calculate page width based on container
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.clientWidth - 48; // padding
        setPageWidth(Math.min(containerWidth, 800));
      }
    };
    
    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  const zoomIn = () => setScale(s => Math.min(s + 0.25, 3));
  const zoomOut = () => setScale(s => Math.max(s - 0.25, 0.5));
  const resetZoom = () => setScale(1);

  const goToPrevPage = () => setCurrentPage(p => Math.max(p - 1, 1));
  const goToNextPage = () => setCurrentPage(p => Math.min(p + 1, numPages));

  // Apply highlights by adding a CSS class to matching text spans.
  // This approach avoids absolute-positioning math, overflow:hidden clipping,
  // and z-index stacking issues — the spans are already correctly placed by pdf.js.
  useEffect(() => {
    const pageEv = pageEvidence.get(currentPage);

    function applyHighlights() {
      const pageElement = document.querySelector(`.react-pdf__Page[data-page-number="${currentPage}"]`);
      if (!pageElement) return;

      const textLayer = pageElement.querySelector('.react-pdf__Page__textContent, .textLayer');

      // Clear old highlights (span class, inline styles, and fallback labels)
      pageElement.querySelectorAll('.pdf-highlight-fallback').forEach(el => el.remove());
      if (textLayer) {
        textLayer.querySelectorAll('.pdf-span-highlight').forEach(el => {
          el.classList.remove('pdf-span-highlight');
          (el as HTMLElement).style.removeProperty('background-color');
          (el as HTMLElement).style.removeProperty('box-shadow');
          (el as HTMLElement).style.removeProperty('border-radius');
        });
      }

      if (!showHighlights || !pageEv || pageEv.items.length === 0) {
        return;
      }

      if (!textLayer) {
        // No text layer (scanned PDF without OCR) — show fallback labels and treat all as matched
        setMatchedItems(prev => { const n = new Map(prev); n.set(currentPage, pageEv.items); return n; });
        pageEv.items.slice(0, 4).forEach((item, index) => {
          const fallback = document.createElement('div');
          fallback.className = 'pdf-highlight-fallback';
          fallback.textContent = item.fieldName.replace(/_/g, ' ');
          fallback.style.top = `${12 + index * 30}px`;
          pageElement.appendChild(fallback);
        });
        return;
      }

      // Build span index: each span's position in concatenated page text.
      // Use [role="presentation"] to select only actual text spans — pdfjs wraps text items
      // in markedContent <span> containers (no role, height:0) which must be excluded.
      const spans = Array.from(textLayer.querySelectorAll('[role="presentation"]')) as HTMLSpanElement[];
      type SpanRange = { span: HTMLSpanElement; start: number; end: number };
      const textRanges: SpanRange[] = [];
      let pos = 0;
      spans.forEach(span => {
        const text = span.textContent || '';
        if (!text.trim()) return;
        if (textRanges.length > 0) pos += 1; // space between spans
        textRanges.push({ span, start: pos, end: pos + text.length });
        pos += text.length;
      });

      if (textRanges.length === 0) {
        setMatchedItems(prev => { const n = new Map(prev); n.set(currentPage, pageEv.items); return n; });
        pageEv.items.slice(0, 4).forEach((item, index) => {
          const fallback = document.createElement('div');
          fallback.className = 'pdf-highlight-fallback';
          fallback.textContent = item.fieldName.replace(/_/g, ' ');
          fallback.style.top = `${12 + index * 30}px`;
          pageElement.appendChild(fallback);
        });
        return;
      }

      const pageText = textRanges.map(r => r.span.textContent || '').join(' ');
      const lowerPage = pageText.toLowerCase();

      const usedRanges: Array<{ start: number; end: number }> = [];
      const highlightRanges: Array<{ start: number; end: number; style: HighlightStyle }> = [];

      const overlaps = (s: number, e: number) =>
        usedRanges.some(r => s < r.end && e > r.start);

      const reserve = (s: number, e: number, style: HighlightStyle) => {
        usedRanges.push({ start: s, end: e });
        highlightRanges.push({ start: s, end: e, style });
      };

      const findFirst = (haystack: string, needle: string): number => {
        let i = haystack.indexOf(needle);
        while (i >= 0) {
          if (!overlaps(i, i + needle.length)) return i;
          i = haystack.indexOf(needle, i + 1);
        }
        return -1;
      };

      const normalize = (s: string) =>
        s.replace(/\s+/g, ' ').trim().toLowerCase();

      // Track which items actually match text on this page
      const itemsFoundOnPage: EvidenceQuote[] = [];

      pageEv.items.forEach(item => {
        const quote = item.quote?.trim();
        if (!quote) return;
        const style = getHighlightStyle(item.fieldName, quote);
        const beforeCount = highlightRanges.length;

        // 1. Exact match
        let idx = findFirst(pageText, quote);
        if (idx >= 0) { reserve(idx, idx + quote.length, style); }
        else {
          // 2. Case-insensitive
          idx = findFirst(lowerPage, quote.toLowerCase());
          if (idx >= 0) { reserve(idx, idx + quote.length, style); }
          else {
            // 3. Normalized whitespace
            const normQuote = normalize(quote);
            const normPage = normalize(pageText);
            idx = normPage.indexOf(normQuote);
            if (idx >= 0 && !overlaps(idx, idx + normQuote.length)) {
              reserve(idx, idx + normQuote.length, style);
            } else {
              // 4. Token-level fallback — require at least 2 significant tokens to match
              const tokens = (quote.match(/[A-Za-zÀ-ÿ0-9]{4,}/g) || [])
                .filter((t, i, a) => a.indexOf(t) === i);
              let tokenMatches = 0;
              tokens.forEach(token => {
                const ti = lowerPage.indexOf(token.toLowerCase());
                if (ti >= 0 && !overlaps(ti, ti + token.length)) {
                  reserve(ti, ti + token.length, style);
                  tokenMatches++;
                }
              });
              // If only 1 or 0 tokens matched via fallback, don't count as found
              if (tokenMatches < 2) {
                // Roll back any single-token reserves from this item
                const added = highlightRanges.length - beforeCount;
                if (added > 0) {
                  highlightRanges.splice(beforeCount, added);
                  usedRanges.splice(beforeCount, added);
                }
              }
            }
          }
        }

        if (highlightRanges.length > beforeCount) {
          itemsFoundOnPage.push(item);
        }
      });

      // Update sidebar to only show items that were actually found on this page
      setMatchedItems(prev => {
        const next = new Map(prev);
        next.set(currentPage, itemsFoundOnPage);
        return next;
      });

      // Apply highlight to spans that overlap any highlight range.
      textRanges.forEach(({ span, start, end }) => {
        const match = highlightRanges.find(r => r.start < end && r.end > start);
        if (match) {
          span.classList.add('pdf-span-highlight');
          span.style.backgroundColor = match.style.bg;
          span.style.boxShadow = match.style.shadow;
          span.style.borderRadius = '2px';
        }
      });

      // Don't show fallback labels when text layer exists but quotes weren't found on this page.
    }

    // Poll with rAF until react-pdf has populated role="presentation" spans inside the text layer.
    // The text layer div appears first; spans are added asynchronously — so we wait for both.
    let cancelled = false;
    const deadline = Date.now() + 3000;

    function waitForSpans() {
      if (cancelled) return;
      const pg = document.querySelector(`.react-pdf__Page[data-page-number="${currentPage}"]`);
      const textLayer = pg?.querySelector('.react-pdf__Page__textContent, .textLayer');
      const spans = textLayer?.querySelectorAll('[role="presentation"]');
      if (spans && spans.length > 0) {
        applyHighlights();
      } else if (Date.now() < deadline) {
        requestAnimationFrame(waitForSpans);
      } else {
        // Deadline hit — run anyway (handles scanned PDFs with no text layer)
        applyHighlights();
      }
    }

    requestAnimationFrame(waitForSpans);

    return () => { cancelled = true; };
  }, [currentPage, pageEvidence, showHighlights, scale, pageWidth]);

  const currentPageItems = matchedItems.get(currentPage) || [];

  return (
    <div className="flex flex-col h-full bg-slate-50" ref={containerRef}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-white border-b border-slate-200">
        {/* Page navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={goToPrevPage}
            disabled={currentPage <= 1}
            className="p-2 text-slate-400 hover:text-slate-800 hover:bg-slate-100 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <FontAwesomeIcon icon={faChevronLeft} className="w-4 h-4" />
          </button>
          <span className="text-slate-800 text-sm min-w-[80px] text-center">
            {currentPage} / {numPages}
          </span>
          <button
            onClick={goToNextPage}
            disabled={currentPage >= numPages}
            className="p-2 text-slate-400 hover:text-slate-800 hover:bg-slate-100 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <FontAwesomeIcon icon={faChevronRight} className="w-4 h-4" />
          </button>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={zoomOut}
            className="p-2 text-slate-400 hover:text-slate-800 hover:bg-slate-100 rounded"
            title="Uitzoomen"
          >
            <FontAwesomeIcon icon={faSearchMinus} className="w-4 h-4" />
          </button>
          <span className="text-slate-800 text-sm min-w-[50px] text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            onClick={zoomIn}
            className="p-2 text-slate-400 hover:text-slate-800 hover:bg-slate-100 rounded"
            title="Inzoomen"
          >
            <FontAwesomeIcon icon={faSearchPlus} className="w-4 h-4" />
          </button>
          <button
            onClick={resetZoom}
            className="p-2 text-slate-400 hover:text-slate-800 hover:bg-slate-100 rounded"
            title="Reset zoom"
          >
            <FontAwesomeIcon icon={faExpand} className="w-4 h-4" />
          </button>
        </div>

        {/* Highlight toggle */}
        {pageEvidence.size > 0 && currentPageItems.length > 0 && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowHighlights(!showHighlights)}
              className={`p-2 rounded transition-colors ${
                showHighlights 
                  ? 'text-blue-400 hover:text-blue-500 hover:bg-blue-500/10' 
                  : 'text-slate-400 hover:text-slate-800 hover:bg-slate-100'
              }`}
              title={showHighlights ? 'Highlights verbergen' : 'Highlights tonen'}
            >
              <FontAwesomeIcon icon={showHighlights ? faHighlighter : faEyeSlash} className="w-4 h-4" />
            </button>
            {currentPageItems.length > 0 && (
              <span className="text-slate-400 text-xs">
                {currentPageItems.length}
              </span>
            )}
          </div>
        )}
      </div>

      {/* PDF Content */}
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row overflow-hidden">
        <div className="flex-1 min-h-0 overflow-auto p-4 flex justify-center">
          <Document
            file={url}
            onLoadSuccess={onDocumentLoadSuccess}
            loading={
              <div className="flex items-center justify-center h-64">
                <div className="text-slate-400">PDF laden...</div>
              </div>
            }
            error={
              <div className="flex items-center justify-center h-64">
                <div className="text-red-400">Fout bij laden van PDF</div>
              </div>
            }
          >
            <Page
              pageNumber={currentPage}
              scale={scale}
              width={pageWidth}
              renderTextLayer={true}
              renderAnnotationLayer={false}
          className="shadow-2xl"
            />
          </Document>
        </div>

        {currentPageItems.length > 0 && (
          <aside className="lg:w-80 lg:max-w-80 max-h-48 lg:max-h-none overflow-hidden bg-white border-t lg:border-t-0 lg:border-l border-slate-200">
            <button
              onClick={() => setShowEvidenceOverlay(!showEvidenceOverlay)}
              className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-50 transition-colors"
            >
              <div className="flex items-center gap-2 min-w-0">
                <FontAwesomeIcon icon={faHighlighter} className="text-blue-400 w-3.5 h-3.5 flex-shrink-0" />
                <div className="min-w-0">
                  <div className="text-slate-800 text-sm font-semibold">Evidence op pagina {currentPage}</div>
                  <div className="text-slate-400 text-[11px]">{currentPageItems.length} bron{currentPageItems.length === 1 ? '' : 'nen'}</div>
                </div>
              </div>
              <FontAwesomeIcon
                icon={showEvidenceOverlay ? faChevronUp : faChevronDown}
                className="text-slate-400 w-3 h-3 flex-shrink-0"
              />
            </button>

            {showEvidenceOverlay && (
              <div className="px-4 pb-4 space-y-2 overflow-y-auto max-h-36 lg:max-h-[calc(100%-64px)]">
                {currentPageItems.map((item, i) => {
                  const hs = getHighlightStyle(item.fieldName, item.quote || '');
                  return (
                  <div
                    key={`${item.fieldName}-${i}`}
                    className="rounded-lg p-2.5"
                    style={{ border: `1px solid ${hs.border}60`, background: hs.cardBg }}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: hs.label }} />
                        <span className="text-[10px] uppercase tracking-wide font-semibold truncate" style={{ color: hs.label }}>
                          {item.fieldName.replace(/_/g, ' ')}
                        </span>
                      </div>
                      <span className="text-[10px] flex-shrink-0" style={{ color: hs.label }}>
                        #{i + 1}
                      </span>
                    </div>
                    <p className="text-xs leading-relaxed break-words" style={{ color: hs.text }}>
                      "{item.quote.length > 140 ? `${item.quote.substring(0, 140)}...` : item.quote}"
                    </p>
                  </div>
                  );
                })}
              </div>
            )}
          </aside>
        )}
      </div>

      {/* Page thumbnails / quick nav for pages with confirmed highlights */}
      {matchedItems.size > 0 && Array.from(matchedItems.values()).some(items => items.length > 0) && (
        <div className="px-4 py-2 bg-white border-t border-slate-200">
          <div className="text-slate-400 text-[10px] mb-1">Pagina's met evidence:</div>
          <div className="flex gap-1 flex-wrap">
            {Array.from(matchedItems.entries())
              .filter(([, items]) => items.length > 0)
              .sort(([a], [b]) => a - b)
              .map(([pageNum]) => (
              <button
                key={pageNum}
                onClick={() => setCurrentPage(pageNum)}
                className={`px-2 py-1 text-xs rounded ${
                  pageNum === currentPage
                    ? 'bg-blue-500 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                p{pageNum}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* CSS for highlights */}
      <style jsx global>{`
        .pdf-highlight {
          background-color: rgba(59, 130, 246, 0.5) !important;
          padding: 2px 0;
          border-radius: 2px;
          box-shadow: 0 0 8px rgba(59, 130, 246, 0.6);
        }

        .pdf-highlight-box {
          position: absolute;
          z-index: 8;
          pointer-events: none;
          background-color: rgba(250, 204, 21, 0.42);
          border: 1px solid rgba(250, 204, 21, 0.95);
          border-radius: 3px;
          box-shadow: 0 0 10px rgba(250, 204, 21, 0.45);
        }

        .pdf-highlight-fallback {
          position: absolute;
          right: 12px;
          z-index: 9;
          pointer-events: none;
          max-width: calc(100% - 24px);
          padding: 5px 8px;
          border-radius: 999px;
          background: rgba(37, 99, 235, 0.9);
          color: white;
          font-size: 11px;
          font-weight: 600;
          box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
        }
        
        .react-pdf__Page__textContent {
          opacity: 0.8;
        }
        
        .react-pdf__Page__textContent span {
          pointer-events: auto;
        }
        
        .react-pdf__Page__canvas {
          border-radius: 8px;
        }

        .react-pdf__Page {
          position: relative !important;
        }
      `}</style>
    </div>
  );
}
