'use client';

import { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faHome, faCog, faBook, faExternalLinkAlt,
  faBars, faTimes, faLayerGroup, faCode,
} from '@fortawesome/free-solid-svg-icons';
import { useAuth } from '@/context/AuthContext';

const BASE_NAV = [
  { href: '/', label: 'Dashboard', icon: faHome, minRole: 'user' },
  { href: '/document-types', label: 'Types', icon: faLayerGroup, minRole: 'admin' },
  { href: '/signals', label: 'Signalen', icon: faCode, minRole: 'admin' },
  { href: '/settings', label: 'Instellingen', icon: faCog, minRole: 'admin' },
] as const;

const ROLE_ORDER = { user: 0, admin: 1, super_admin: 2 } as const;

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export function Navigation() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const { user } = useAuth();
  const userRoleOrder = ROLE_ORDER[(user?.role ?? 'user') as keyof typeof ROLE_ORDER] ?? 0;
  const navItems = BASE_NAV.filter(item => userRoleOrder >= ROLE_ORDER[item.minRole as keyof typeof ROLE_ORDER]);

  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [open]);

  return (
    <>
      {/* Desktop */}
      <nav className="hidden md:flex items-center gap-0.5">
        {navItems.map(item => {
          const active = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                active
                  ? 'bg-[#FFC1F3]/15 text-slate-800 border border-[#FFC1F3]/25'
                  : 'text-slate-400 hover:text-slate-800 hover:bg-slate-100'
              }`}
            >
              <FontAwesomeIcon icon={item.icon} className="w-3.5 h-3.5" />
              <span>{item.label}</span>
            </Link>
          );
        })}
        <div className="w-px h-5 bg-slate-100 mx-1" />
        <a
          href={`${BACKEND_URL}/docs`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-500 hover:bg-slate-100 transition-all"
        >
          <FontAwesomeIcon icon={faBook} className="w-3.5 h-3.5" />
          <span className="hidden lg:inline">API</span>
          <FontAwesomeIcon icon={faExternalLinkAlt} className="w-2.5 h-2.5" />
        </a>
      </nav>

      {/* Mobile toggle */}
      <button
        onClick={() => setOpen(!open)}
        className="flex md:hidden p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-all"
        aria-label="Menu"
      >
        <FontAwesomeIcon icon={open ? faTimes : faBars} className="w-5 h-5" />
      </button>

      {/* Mobile overlay */}
      {open && (
        <>
          <div
            className="fixed inset-0 md:hidden z-[9998]"
            style={{ top: '56px', background: 'rgba(245, 247, 255, 0.7)', backdropFilter: 'blur(4px)' }}
            onClick={() => setOpen(false)}
          />
          <div
            className="fixed left-0 right-0 md:hidden z-[9999]"
            style={{ top: '56px', background: '#f5f7ff', borderBottom: '1px solid rgba(0,0,0,0.08)' }}
          >
            <nav className="flex flex-col p-3 gap-1">
              {navItems.map(item => {
                const active = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    onClick={() => setOpen(false)}
                    className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                      active
                        ? 'bg-[#FFC1F3]/12 text-slate-800 border border-[#FFC1F3]/20'
                        : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'
                    }`}
                  >
                    <FontAwesomeIcon icon={item.icon} className="w-4 h-4" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
              <div className="h-px bg-slate-100 my-1" />
              <a
                href={`${BACKEND_URL}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setOpen(false)}
                className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-all"
              >
                <FontAwesomeIcon icon={faBook} className="w-4 h-4" />
                <span>API Docs</span>
                <FontAwesomeIcon icon={faExternalLinkAlt} className="w-3 h-3 ml-auto opacity-50" />
              </a>
            </nav>
          </div>
        </>
      )}
    </>
  );
}
