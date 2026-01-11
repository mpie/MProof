'use client';

import { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faCog, faBook, faExternalLinkAlt, faBars, faTimes, faLayerGroup, faShieldAlt, faCode } from '@fortawesome/free-solid-svg-icons';

const navItems = [
  { href: '/', label: 'Dashboard', icon: faHome },
  { href: '/document-types', label: 'Documenttypes', icon: faLayerGroup },
  { href: '/signals', label: 'Signalen', icon: faCode },
  { href: '/settings', label: 'Instellingen', icon: faCog },
];

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export function Navigation() {
  const pathname = usePathname();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Lock body scroll when menu is open
  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isMobileMenuOpen]);

  return (
    <>
      {/* Desktop Navigation - CSS-based responsive hiding */}
      <nav className="hidden md:flex items-center space-x-1">
        {navItems.map((item) => {
          const isActive = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer ${
                isActive
                  ? 'bg-blue-500/30 text-white shadow-lg border border-blue-400/30'
                  : 'text-white/70 hover:text-white hover:bg-white/10'
              }`}
            >
              <FontAwesomeIcon icon={item.icon} className="w-4 h-4" />
              <span>{item.label}</span>
            </Link>
          );
        })}
        
        {/* Divider */}
        <div className="w-px h-6 bg-white/20 mx-1" />
        
        {/* API Docs Link */}
        <a
          href={`${BACKEND_URL}/docs`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium text-white/70 hover:text-white hover:bg-white/10 transition-all cursor-pointer"
          title="API Documentatie (Swagger)"
        >
          <FontAwesomeIcon icon={faBook} className="w-4 h-4" />
          <span className="hidden lg:inline">API Docs</span>
          <FontAwesomeIcon icon={faExternalLinkAlt} className="w-3 h-3 opacity-50" />
        </a>
      </nav>

      {/* Mobile Menu Button - CSS-based responsive hiding */}
      <button
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        className="flex md:hidden p-2 text-white/70 hover:text-white hover:bg-white/10 rounded-lg transition-all"
        aria-label="Toggle menu"
      >
        <FontAwesomeIcon icon={isMobileMenuOpen ? faTimes : faBars} className="w-5 h-5" />
      </button>

      {/* Mobile Navigation Menu Overlay - only shown on mobile */}
      {isMobileMenuOpen && (
        <>
          {/* Backdrop - shadow overlay covering content below menu */}
          <div 
            className="fixed inset-0 md:hidden"
            style={{ 
              top: '56px',
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              backdropFilter: 'blur(4px)',
              WebkitBackdropFilter: 'blur(4px)',
              zIndex: 9998
            }}
            onClick={() => setIsMobileMenuOpen(false)}
          />
          {/* Menu - solid opaque background */}
          <div 
            className="fixed left-0 right-0 md:hidden"
            style={{
              top: '56px',
              backgroundColor: '#0f172a',
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
              maxHeight: 'calc(100vh - 56px)',
              overflowY: 'auto',
              zIndex: 9999
            }}
          >
            <nav className="flex flex-col p-3 space-y-1">
              {navItems.map((item) => {
                const isActive = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`flex items-center space-x-3 px-4 py-3 rounded-xl text-base font-medium transition-all cursor-pointer ${
                      isActive
                        ? 'bg-blue-500/20 text-white border border-blue-500/30'
                        : 'text-white/80 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    <FontAwesomeIcon icon={item.icon} className="w-5 h-5" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
              
              <div className="h-px bg-white/10 my-2" />
              
              <a
                href={`${BACKEND_URL}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setIsMobileMenuOpen(false)}
                className="flex items-center space-x-3 px-4 py-3 rounded-xl text-base font-medium text-white/70 hover:text-white hover:bg-white/10 transition-all cursor-pointer"
              >
                <FontAwesomeIcon icon={faBook} className="w-5 h-5" />
                <span>API Docs</span>
                <FontAwesomeIcon icon={faExternalLinkAlt} className="w-3 h-3 opacity-50 ml-auto" />
              </a>
            </nav>
          </div>
        </>
      )}
    </>
  );
}
