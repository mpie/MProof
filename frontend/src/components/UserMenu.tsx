'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faUser, faSignOutAlt, faUsers, faKey, faChevronDown, faSpinner } from '@fortawesome/free-solid-svg-icons';
import { useAuth } from '@/context/AuthContext';

const ROLE_LABELS: Record<string, string> = {
  super_admin: 'Super Admin',
  admin: 'Admin',
  user: 'User',
};

export function UserMenu() {
  const { user, isLoading, logout } = useAuth();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  if (isLoading) {
    return <FontAwesomeIcon icon={faSpinner} className="text-slate-300 w-4 h-4 animate-spin" />;
  }

  if (!user) return null;

  function handleLogout() {
    logout();
    router.push('/login');
  }

  const isAdmin = user.role === 'admin' || user.role === 'super_admin';

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl hover:bg-slate-100 transition-all text-sm"
      >
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-[#22d3d3] to-[#FFC1F3] flex items-center justify-center text-white text-xs font-semibold flex-shrink-0">
          {user.name.charAt(0).toUpperCase()}
        </div>
        <span className="hidden sm:block text-slate-700 font-medium max-w-[120px] truncate">{user.name}</span>
        <FontAwesomeIcon icon={faChevronDown} className={`w-3 h-3 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-52 glass-card py-1.5 shadow-xl z-[200]">
          {/* User info */}
          <div className="px-4 py-2.5 border-b border-slate-100">
            <div className="font-semibold text-slate-800 text-sm truncate">{user.name}</div>
            <div className="text-xs text-slate-500 truncate">{user.email}</div>
            <div className="mt-1">
              <span className="text-xs text-slate-400">{ROLE_LABELS[user.role] ?? user.role}</span>
            </div>
          </div>

          {/* Menu items */}
          <div className="py-1">
            {isAdmin && (
              <Link
                href="/users"
                onClick={() => setOpen(false)}
                className="flex items-center gap-2.5 px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
              >
                <FontAwesomeIcon icon={faUsers} className="w-3.5 h-3.5 text-slate-400" />
                Gebruikers
              </Link>
            )}
            <Link
              href="/settings?tab=api-keys"
              onClick={() => setOpen(false)}
              className="flex items-center gap-2.5 px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
            >
              <FontAwesomeIcon icon={faKey} className="w-3.5 h-3.5 text-slate-400" />
              API Keys
            </Link>
          </div>

          <div className="border-t border-slate-100 pt-1">
            <button
              onClick={handleLogout}
              className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
            >
              <FontAwesomeIcon icon={faSignOutAlt} className="w-3.5 h-3.5" />
              Uitloggen
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
