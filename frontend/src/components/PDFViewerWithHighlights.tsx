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

export function PDFViewerWithHighlights({ url, evidence = {} }: PDFViewerWithHighlightsProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [scale, setScale] = useState<number>(1.0);
  const [pageWidth, setPageWidth] = useState<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const [pageEvidence, setPageEvidence] = useState<Map<number, PageEvidence>>(new Map());
  const [showHighlights, setShowHighlights] = useState<boolean>(true);
  const [showEvidenceOverlay, setShowEvidenceOverlay] = useState<boolean>(true);
  const [textLayerRenderTick, setTextLayerRenderTick] = useState<number>(0);
  const [didJumpToEvidence, setDidJumpToEvidence] = useState<boolean>(false);

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

  // Apply highlights to text layer after rendering. Prefer quote matching over OCR offsets,
  // because OCR offsets often do not line up with React-PDF's text layer.
  useEffect(() => {
    const pageEv = pageEvidence.get(currentPage);
    
    // Wait for text layer to be rendered
    const timeout = setTimeout(() => {
      const pageElement = document.querySelector(`.react-pdf__Page[data-page-number="${currentPage}"]`);
      if (!pageElement) return;
      
      const textLayer = pageElement.querySelector('.react-pdf__Page__textContent, .textLayer');
      pageElement.querySelectorAll('.pdf-highlight-box, .pdf-highlight-fallback').forEach(element => element.remove());
      if (!textLayer) {
        if (showHighlights && pageEv && pageEv.items.length > 0) {
          pageEv.items.slice(0, 3).forEach((item, index) => {
            const fallback = document.createElement('div');
            fallback.className = 'pdf-highlight-fallback';
            fallback.textContent = item.fieldName.replace(/_/g, ' ');
            fallback.style.top = `${12 + index * 30}px`;
            pageElement.appendChild(fallback);
          });
        }
        return;
      }
      
      // First, remove existing highlights by restoring original text
      const allSpans = textLayer.querySelectorAll('span, mark');
      allSpans.forEach((element: Element) => {
        if (element.tagName === 'MARK' && element.classList.contains('pdf-highlight')) {
          const parent = element.parentNode;
          if (parent) {
            const textNode = document.createTextNode(element.textContent || '');
            parent.replaceChild(textNode, element);
            parent.normalize();
          }
        }
      });
      
      // If highlights are disabled, we're done
      if (!showHighlights || !pageEv || pageEv.items.length === 0) return;
      
      // Collect all text spans and calculate their character positions
      const spans = textLayer.querySelectorAll('span');
      const textRanges: Array<{ span: HTMLSpanElement; start: number; end: number; originalText: string }> = [];
      let cumulativePos = 0;
      
      spans.forEach((span: HTMLSpanElement) => {
        const text = span.textContent || '';
        if (text.trim()) {
          if (textRanges.length > 0) {
            cumulativePos += 1; // Separator between PDF text spans for quote matching.
          }
          const start = cumulativePos;
          const end = cumulativePos + text.length;
          textRanges.push({ span, start, end, originalText: text });
          cumulativePos = end;
        }
      });

      if (textRanges.length === 0) {
        pageEv.items.slice(0, 3).forEach((item, index) => {
          const fallback = document.createElement('div');
          fallback.className = 'pdf-highlight-fallback';
          fallback.textContent = item.fieldName.replace(/_/g, ' ');
          fallback.style.top = `${12 + index * 30}px`;
          pageElement.appendChild(fallback);
        });
        return;
      }

      const pageText = textRanges.map(range => range.originalText).join(' ');
      const normalizeWithMap = (value: string) => {
        let normalized = '';
        const indexMap: number[] = [];
        let previousWasSpace = false;

        Array.from(value).forEach((char, index) => {
          if (/\s/.test(char)) {
            if (!previousWasSpace && normalized.length > 0) {
              normalized += ' ';
              indexMap.push(index);
            }
            previousWasSpace = true;
            return;
          }

          normalized += char.toLowerCase();
          indexMap.push(index);
          previousWasSpace = false;
        });

        return { normalized: normalized.trim(), indexMap };
      };

      const normalizedPage = normalizeWithMap(pageText);
      const usedRanges: Array<{ start: number; end: number }> = [];
      const highlightRanges: Array<{ start: number; end: number }> = [];

      const overlapsUsedRange = (start: number, end: number) =>
        usedRanges.some(range => start < range.end && end > range.start);

      const reserveRange = (start: number, end: number) => {
        usedRanges.push({ start, end });
        highlightRanges.push({ start, end });
      };

      const findAvailableIndex = (text: string, search: string) => {
        let start = text.indexOf(search);
        while (start >= 0) {
          const end = start + search.length;
          if (!overlapsUsedRange(start, end)) return start;
          start = text.indexOf(search, start + 1);
        }

        return -1;
      };

      const findTokenRanges = (quote: string) => {
        const tokens = Array.from(new Set(
          quote
            .match(/[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9.,:/-]*/g)
            ?.map(token => token.trim())
            .filter(token => token.length >= 2) || []
        ));
        const ranges: Array<{ start: number; end: number }> = [];
        const lowerPageText = pageText.toLowerCase();

        tokens.forEach(token => {
          let start = lowerPageText.indexOf(token.toLowerCase());
          while (start >= 0) {
            const end = start + token.length;
            if (!overlapsUsedRange(start, end)) {
              ranges.push({ start, end });
              break;
            }
            start = lowerPageText.indexOf(token.toLowerCase(), start + 1);
          }
        });

        return ranges;
      };

      pageEv.items.forEach(item => {
        const quote = item.quote?.trim();
        let matchStart = -1;
        let matchEnd = -1;

        if (quote) {
          const exactStart = findAvailableIndex(pageText, quote);
          if (exactStart >= 0) {
            matchStart = exactStart;
            matchEnd = exactStart + quote.length;
          } else {
            const lowerStart = findAvailableIndex(pageText.toLowerCase(), quote.toLowerCase());
            if (lowerStart >= 0) {
              matchStart = lowerStart;
              matchEnd = lowerStart + quote.length;
            } else {
              const normalizedQuote = normalizeWithMap(quote).normalized;
              let normalizedStart = normalizedQuote ? normalizedPage.normalized.indexOf(normalizedQuote) : -1;
              while (normalizedStart >= 0) {
                const rawStart = normalizedPage.indexMap[normalizedStart];
                const rawEnd = normalizedPage.indexMap[normalizedStart + normalizedQuote.length - 1] + 1;
                if (!overlapsUsedRange(rawStart, rawEnd)) {
                  matchStart = rawStart;
                  matchEnd = rawEnd;
                  break;
                }

                normalizedStart = normalizedPage.normalized.indexOf(normalizedQuote, normalizedStart + 1);
              }
            }
          }
        } else if (item.start > 0 && item.end > item.start) {
          matchStart = item.start;
          matchEnd = item.end;
        }

        if (matchStart >= 0 && matchEnd > matchStart) {
          reserveRange(matchStart, matchEnd);
        } else if (quote) {
          findTokenRanges(quote).forEach(range => reserveRange(range.start, range.end));
        }
      });

      if (highlightRanges.length === 0) {
        pageEv.items.slice(0, 3).forEach((item, index) => {
          const fallback = document.createElement('div');
          fallback.className = 'pdf-highlight-fallback';
          fallback.textContent = item.fieldName.replace(/_/g, ' ');
          fallback.style.top = `${12 + index * 30}px`;
          pageElement.appendChild(fallback);
        });
        return;
      }

      const pageRect = pageElement.getBoundingClientRect();
      textRanges.forEach(({ span, start, end }) => {
        const localRanges = highlightRanges
          .filter(range => range.start < end && range.end > start)
          .map(range => ({
            start: Math.max(0, range.start - start),
            end: Math.max(0, range.end - start),
          }))
          .sort((a, b) => a.start - b.start);

        if (localRanges.length === 0) return;

        const spanRect = span.getBoundingClientRect();
        if (spanRect.width <= 0 || spanRect.height <= 0) return;

        const highlightBox = document.createElement('div');
        highlightBox.className = 'pdf-highlight-box';
        highlightBox.style.left = `${spanRect.left - pageRect.left}px`;
        highlightBox.style.top = `${spanRect.top - pageRect.top}px`;
        highlightBox.style.width = `${spanRect.width}px`;
        highlightBox.style.height = `${spanRect.height}px`;
        pageElement.appendChild(highlightBox);
      });
    }, 150);
    
    return () => clearTimeout(timeout);
  }, [currentPage, pageEvidence, showHighlights, scale, pageWidth, textLayerRenderTick]);

  const currentPageEv = pageEvidence.get(currentPage);
  const currentPageItems = currentPageEv?.items || [];

  return (
    <div className="flex flex-col h-full bg-gray-900" ref={containerRef}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-white/10">
        {/* Page navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={goToPrevPage}
            disabled={currentPage <= 1}
            className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <FontAwesomeIcon icon={faChevronLeft} className="w-4 h-4" />
          </button>
          <span className="text-white text-sm min-w-[80px] text-center">
            {currentPage} / {numPages}
          </span>
          <button
            onClick={goToNextPage}
            disabled={currentPage >= numPages}
            className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <FontAwesomeIcon icon={faChevronRight} className="w-4 h-4" />
          </button>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={zoomOut}
            className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded"
            title="Uitzoomen"
          >
            <FontAwesomeIcon icon={faSearchMinus} className="w-4 h-4" />
          </button>
          <span className="text-white text-sm min-w-[50px] text-center">
            {Math.round(scale * 100)}%
          </span>
          <button
            onClick={zoomIn}
            className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded"
            title="Inzoomen"
          >
            <FontAwesomeIcon icon={faSearchPlus} className="w-4 h-4" />
          </button>
          <button
            onClick={resetZoom}
            className="p-2 text-white/60 hover:text-white hover:bg-white/10 rounded"
            title="Reset zoom"
          >
            <FontAwesomeIcon icon={faExpand} className="w-4 h-4" />
          </button>
        </div>

        {/* Highlight toggle */}
        {pageEvidence.size > 0 && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowHighlights(!showHighlights)}
              className={`p-2 rounded transition-colors ${
                showHighlights 
                  ? 'text-blue-400 hover:text-blue-300 hover:bg-blue-500/10' 
                  : 'text-white/60 hover:text-white hover:bg-white/10'
              }`}
              title={showHighlights ? 'Highlights verbergen' : 'Highlights tonen'}
            >
              <FontAwesomeIcon icon={showHighlights ? faHighlighter : faEyeSlash} className="w-4 h-4" />
            </button>
            {currentPageItems.length > 0 && (
              <span className="text-white/60 text-xs">
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
                <div className="text-white/60">PDF laden...</div>
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
              onRenderTextLayerSuccess={() => setTextLayerRenderTick(tick => tick + 1)}
          className="shadow-2xl"
            />
          </Document>
        </div>

        {currentPageItems.length > 0 && (
          <aside className="lg:w-80 lg:max-w-80 max-h-48 lg:max-h-none overflow-hidden bg-gray-900/95 border-t lg:border-t-0 lg:border-l border-white/10">
            <button
              onClick={() => setShowEvidenceOverlay(!showEvidenceOverlay)}
              className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-white/5 transition-colors"
            >
              <div className="flex items-center gap-2 min-w-0">
                <FontAwesomeIcon icon={faHighlighter} className="text-blue-400 w-3.5 h-3.5 flex-shrink-0" />
                <div className="min-w-0">
                  <div className="text-white text-sm font-semibold">Evidence op pagina {currentPage}</div>
                  <div className="text-white/45 text-[11px]">{currentPageItems.length} bron{currentPageItems.length === 1 ? '' : 'nen'}</div>
                </div>
              </div>
              <FontAwesomeIcon 
                icon={showEvidenceOverlay ? faChevronUp : faChevronDown} 
                className="text-white/60 w-3 h-3 flex-shrink-0" 
              />
            </button>

            {showEvidenceOverlay && (
              <div className="px-4 pb-4 space-y-2 overflow-y-auto max-h-36 lg:max-h-[calc(100%-64px)]">
                {currentPageItems.map((item, i) => (
                  <div 
                    key={`${item.fieldName}-${i}`}
                    className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-2.5"
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span className="text-[10px] uppercase tracking-wide text-blue-300 font-semibold truncate">
                        {item.fieldName.replace(/_/g, ' ')}
                      </span>
                      <span className="text-[10px] text-white/35 flex-shrink-0">
                        #{i + 1}
                      </span>
                    </div>
                    <p className="text-xs text-white/85 leading-relaxed break-words">
                      "{item.quote.length > 140 ? `${item.quote.substring(0, 140)}...` : item.quote}"
                    </p>
                  </div>
                ))}
              </div>
            )}
          </aside>
        )}
      </div>

      {/* Page thumbnails / quick nav for pages with highlights */}
      {pageEvidence.size > 0 && (
        <div className="px-4 py-2 bg-gray-800 border-t border-white/10">
          <div className="text-white/50 text-[10px] mb-1">Pagina's met evidence:</div>
          <div className="flex gap-1 flex-wrap">
            {Array.from(pageEvidence.keys()).sort((a, b) => a - b).map(pageNum => (
              <button
                key={pageNum}
                onClick={() => setCurrentPage(pageNum)}
                className={`px-2 py-1 text-xs rounded ${
                  pageNum === currentPage 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-white/10 text-white/70 hover:bg-white/20'
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
          background-color: rgba(59, 130, 246, 0.35);
          border: 1px solid rgba(96, 165, 250, 0.9);
          border-radius: 3px;
          box-shadow: 0 0 10px rgba(59, 130, 246, 0.55);
          mix-blend-mode: multiply;
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
