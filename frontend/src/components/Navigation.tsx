'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faFileAlt, faCog, faBook, faExternalLinkAlt } from '@fortawesome/free-solid-svg-icons';

const navItems = [
  { href: '/', label: 'Dashboard', icon: faHome },
  { href: '/documents', label: 'Documenten', icon: faFileAlt },
  { href: '/document-types', label: 'Configuratie', icon: faCog },
];

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="flex items-center space-x-1">
      {navItems.map((item) => {
        const isActive = pathname === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
              isActive
                ? 'bg-white/20 text-white shadow-lg'
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
        className="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium text-white/70 hover:text-white hover:bg-white/10 transition-all"
        title="API Documentatie (Swagger)"
      >
        <FontAwesomeIcon icon={faBook} className="w-4 h-4" />
        <span>API Docs</span>
        <FontAwesomeIcon icon={faExternalLinkAlt} className="w-3 h-3 opacity-50" />
      </a>
    </nav>
  );
}
