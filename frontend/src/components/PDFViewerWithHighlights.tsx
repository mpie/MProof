'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearchPlus, faSearchMinus, faExpand, faChevronLeft, faChevronRight } from '@fortawesome/free-solid-svg-icons';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface EvidenceItem {
  page: number;
  start?: number;
  end?: number;
  quote: string;
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
  const [highlightedTexts, setHighlightedTexts] = useState<Map<number, string[]>>(new Map());

  // Collect all quotes per page
  useEffect(() => {
    const quotesPerPage = new Map<number, string[]>();
    
    console.log('[PDFViewer] Processing evidence:', evidence);
    
    Object.entries(evidence).forEach(([fieldName, items]) => {
      if (Array.isArray(items)) {
        items.forEach(item => {
          if (item.quote && item.page !== undefined && item.page !== null) {
            // Handle both number and string page values, convert 0-indexed to 1-indexed
            const rawPage = typeof item.page === 'string' ? parseInt(item.page, 10) : item.page;
            const pageNum = rawPage + 1;
            
            console.log(`[PDFViewer] Field "${fieldName}" -> page ${rawPage} (display: ${pageNum}), quote: "${item.quote.substring(0, 30)}..."`);
            
            const existing = quotesPerPage.get(pageNum) || [];
            if (!existing.includes(item.quote)) {
              existing.push(item.quote);
            }
            quotesPerPage.set(pageNum, existing);
          }
        });
      }
    });
    
    console.log('[PDFViewer] Pages with highlights:', Array.from(quotesPerPage.keys()));
    setHighlightedTexts(quotesPerPage);
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

  // Custom text renderer to highlight evidence
  const textRenderer = useCallback((textItem: { str: string }) => {
    const pageQuotes = highlightedTexts.get(currentPage) || [];
    let result = textItem.str;
    
    // Check if this text contains any of our quotes
    pageQuotes.forEach(quote => {
      // Normalize both strings for comparison
      const normalizedResult = result.toLowerCase();
      const normalizedQuote = quote.toLowerCase().substring(0, 30); // Use first 30 chars for matching
      
      if (normalizedResult.includes(normalizedQuote) || normalizedQuote.includes(normalizedResult)) {
        // Wrap in highlight span
        result = `<mark class="pdf-highlight">${result}</mark>`;
      }
    });
    
    return result;
  }, [currentPage, highlightedTexts]);

  const currentPageQuotes = highlightedTexts.get(currentPage) || [];

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

        {/* Highlight indicator */}
        {currentPageQuotes.length > 0 && (
          <div className="text-blue-400 text-xs">
            {currentPageQuotes.length} highlight{currentPageQuotes.length > 1 ? 's' : ''} op deze pagina
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
              customTextRenderer={textRenderer}
              className="shadow-2xl"
            />
            
            {/* Highlight overlay for current page quotes */}
            {currentPageQuotes.length > 0 && (
              <div className="absolute bottom-4 left-4 right-4 bg-blue-500/20 backdrop-blur-sm rounded-lg p-2 border border-blue-500/40">
                <div className="text-white text-xs font-bold mb-1">Gevonden evidence:</div>
                <div className="flex flex-wrap gap-1">
                  {currentPageQuotes.map((quote, i) => (
                    <span 
                      key={i} 
                      className="text-[9px] bg-blue-500/40 text-white px-1.5 py-0.5 rounded"
                    >
                      "{quote.substring(0, 40)}{quote.length > 40 ? '...' : ''}"
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Document>
      </div>

      {/* Page thumbnails / quick nav for pages with highlights */}
      {highlightedTexts.size > 0 && (
        <div className="px-4 py-2 bg-gray-800 border-t border-white/10">
          <div className="text-white/50 text-[10px] mb-1">Pagina's met evidence:</div>
          <div className="flex gap-1 flex-wrap">
            {Array.from(highlightedTexts.keys()).sort((a, b) => a - b).map(pageNum => (
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
