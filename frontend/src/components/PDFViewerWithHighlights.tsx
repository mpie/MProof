'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearchPlus, faSearchMinus, faExpand, faChevronLeft, faChevronRight, faHighlighter, faEye, faEyeSlash, faChevronDown, faChevronUp } from '@fortawesome/free-solid-svg-icons';

interface EvidenceItem {
  page: number;
  start: number;
  end: number;
  quote: string;
}

interface PageEvidence {
  items: EvidenceItem[];
  charRanges: Array<{ start: number; end: number }>;
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

  // Set up PDF.js worker on client side only
  useEffect(() => {
    if (typeof window !== 'undefined') {
      // Use jsdelivr CDN which is more reliable than unpkg
      pdfjs.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
    }
  }, []);

  // Collect evidence per page with start/end positions
  useEffect(() => {
    const evidencePerPage = new Map<number, PageEvidence>();
    
    console.log('[PDFViewer] Processing evidence:', evidence);
    
    Object.entries(evidence).forEach(([fieldName, items]) => {
      if (Array.isArray(items)) {
        items.forEach(item => {
          if (item.quote && item.page !== undefined && item.page !== null && item.start !== undefined && item.end !== undefined) {
            // Handle both number and string page values, convert 0-indexed to 1-indexed
            const rawPage = typeof item.page === 'string' ? parseInt(item.page, 10) : item.page;
            const pageNum = rawPage + 1;
            
            console.log(`[PDFViewer] Field "${fieldName}" -> page ${rawPage} (display: ${pageNum}), start: ${item.start}, end: ${item.end}`);
            
            const existing = evidencePerPage.get(pageNum);
            if (existing) {
              existing.items.push(item);
              existing.charRanges.push({ start: item.start, end: item.end });
            } else {
              evidencePerPage.set(pageNum, {
                items: [item],
                charRanges: [{ start: item.start, end: item.end }]
              });
            }
          }
        });
      }
    });
    
    console.log('[PDFViewer] Pages with evidence:', Array.from(evidencePerPage.keys()));
    setPageEvidence(evidencePerPage);
  }, [evidence]);

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

  // Apply highlights to text layer after rendering using start/end positions
  useEffect(() => {
    const pageEv = pageEvidence.get(currentPage);
    
    // Wait for text layer to be rendered
    const timeout = setTimeout(() => {
      const pageElement = document.querySelector(`.react-pdf__Page[data-page-number="${currentPage}"]`);
      if (!pageElement) return;
      
      const textLayer = pageElement.querySelector('.react-pdf__Page__textContent');
      if (!textLayer) return;
      
      // First, remove all existing highlights by restoring original text
      const allSpans = textLayer.querySelectorAll('span, mark');
      allSpans.forEach((element: Element) => {
        if (element.tagName === 'MARK' && element.classList.contains('pdf-highlight')) {
          // Replace mark with its text content
          const parent = element.parentNode;
          if (parent) {
            const textNode = document.createTextNode(element.textContent || '');
            parent.replaceChild(textNode, element);
            // Normalize to merge adjacent text nodes
            parent.normalize();
          }
        }
      });
      
      // If highlights are disabled, we're done
      if (!showHighlights || !pageEv || pageEv.charRanges.length === 0) return;
      
      // Collect all text spans and calculate their character positions
      const spans = textLayer.querySelectorAll('span');
      const textRanges: Array<{ span: HTMLSpanElement; start: number; end: number; originalText: string }> = [];
      let cumulativePos = 0;
      
      spans.forEach((span: HTMLSpanElement) => {
        const text = span.textContent || '';
        if (text.trim()) {
          const start = cumulativePos;
          const end = cumulativePos + text.length;
          textRanges.push({ span, start, end, originalText: text });
          cumulativePos = end;
        }
      });
      
      // Apply highlights based on evidence ranges
      pageEv.charRanges.forEach(range => {
        textRanges.forEach(({ span, start, end, originalText }) => {
          // Check if this span overlaps with the evidence range
          if (range.start < end && range.end > start) {
            const highlightStart = Math.max(0, range.start - start);
            const highlightEnd = Math.min(originalText.length, range.end - start);
            
            if (highlightStart < highlightEnd && span.textContent === originalText) {
              // Only apply if span hasn't been modified yet
              const before = originalText.substring(0, highlightStart);
              const matched = originalText.substring(highlightStart, highlightEnd);
              const after = originalText.substring(highlightEnd);
              
              // Create highlight wrapper
              const highlightSpan = document.createElement('mark');
              highlightSpan.className = 'pdf-highlight';
              highlightSpan.textContent = matched;
              
              // Replace the span content
              span.textContent = '';
              if (before) span.appendChild(document.createTextNode(before));
              span.appendChild(highlightSpan);
              if (after) span.appendChild(document.createTextNode(after));
            }
          }
        });
      });
    }, 150);
    
    return () => clearTimeout(timeout);
  }, [currentPage, pageEvidence, showHighlights]);

  const currentPageEv = pageEvidence.get(currentPage);
  const currentPageQuotes = currentPageEv?.items.map(item => item.quote) || [];

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
            {currentPageQuotes.length > 0 && (
              <span className="text-white/60 text-xs">
                {currentPageQuotes.length}
              </span>
            )}
          </div>
        )}
      </div>

      {/* PDF Content */}
      <div className="flex-1 overflow-auto p-4 flex justify-center">
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
          <div className="relative">
            <Page
              pageNumber={currentPage}
              scale={scale}
              width={pageWidth}
              renderTextLayer={true}
              renderAnnotationLayer={false}
              className="shadow-2xl"
            />
            
            {/* Evidence overlay - moved to top and made collapsible */}
            {currentPageQuotes.length > 0 && (
              <div className="absolute top-4 left-4 right-4 z-10">
                <div className="bg-blue-500/20 backdrop-blur-sm rounded-lg border border-blue-500/40 overflow-hidden">
                  <button
                    onClick={() => setShowEvidenceOverlay(!showEvidenceOverlay)}
                    className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-blue-500/10 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <FontAwesomeIcon icon={faHighlighter} className="text-blue-400 w-3.5 h-3.5" />
                      <span className="text-white text-xs font-bold drop-shadow-lg" style={{ textShadow: '0 2px 4px rgba(0,0,0,0.8), 0 0 8px rgba(0,0,0,0.6)' }}>
                        Gevonden evidence ({currentPageQuotes.length})
                      </span>
                    </div>
                    <FontAwesomeIcon 
                      icon={showEvidenceOverlay ? faChevronUp : faChevronDown} 
                      className="text-white/60 w-3 h-3" 
                    />
                  </button>
                  {showEvidenceOverlay && (
                    <div className="px-3 pb-3 pt-1 max-h-32 overflow-y-auto">
                      <div className="flex flex-col gap-1.5">
                        {currentPageQuotes.map((quote, i) => (
                          <div 
                            key={i} 
                            className="text-[10px] bg-blue-500/30 text-white px-2 py-1.5 rounded border border-blue-500/40"
                          >
                            <span className="font-medium text-blue-200 drop-shadow-md" style={{ textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}>#{i + 1}:</span>{" "}
                            <span className="text-white/90 drop-shadow-md" style={{ textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}>
                              {quote.length > 60 ? `"${quote.substring(0, 60)}..."` : `"${quote}"`}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </Document>
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
        
        .react-pdf__Page__textContent {
          opacity: 0.8;
        }
        
        .react-pdf__Page__textContent span {
          pointer-events: auto;
        }
        
        .react-pdf__Page__canvas {
          border-radius: 8px;
        }
      `}</style>
    </div>
  );
}
