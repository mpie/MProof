'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPlus, faEdit, faTrash, faKey, faTimes, faSpinner, faCheck, faUser,
  faShieldAlt, faCrown, faToggleOn, faToggleOff,
} from '@fortawesome/free-solid-svg-icons';
import {
  listUsers, createUser, updateUser, deleteUser, resetUserPassword,
  User, CreateUserRequest, UpdateUserRequest,
} from '@/lib/api';
import { useAuth } from '@/context/AuthContext';

const ROLE_LABELS: Record<string, { label: string; color: string; icon: typeof faUser }> = {
  super_admin: { label: 'Super Admin', color: 'text-purple-700 bg-purple-100 border-purple-200', icon: faCrown },
  admin:       { label: 'Admin',       color: 'text-blue-700 bg-blue-100 border-blue-200',     icon: faShieldAlt },
  user:        { label: 'User',        color: 'text-slate-600 bg-slate-100 border-slate-200',  icon: faUser },
};

function RoleBadge({ role }: { role: string }) {
  const r = ROLE_LABELS[role] ?? ROLE_LABELS.user;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${r.color}`}>
      <FontAwesomeIcon icon={r.icon} className="w-3 h-3" />
      {r.label}
    </span>
  );
}

function CreateUserModal({ onClose, canCreateAdmin }: { onClose: () => void; canCreateAdmin: boolean }) {
  const qc = useQueryClient();
  const [form, setForm] = useState<CreateUserRequest>({ email: '', name: '', password: '', role: 'user' });
  const [error, setError] = useState('');

  const mutation = useMutation({
    mutationFn: createUser,
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['users'] }); onClose(); },
    onError: (e: unknown) => setError((e as { response?: { data?: { detail?: string } } }).response?.data?.detail ?? 'Fout bij aanmaken'),
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/30 backdrop-blur-sm">
      <div className="glass-card w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-slate-800">Nieuwe gebruiker</h2>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-all">
            <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
          </button>
        </div>

        {error && <div className="mb-4 px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">{error}</div>}

        <div className="flex flex-col gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Naam</label>
            <input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-[#22d3d3]/40 focus:border-[#22d3d3]/60"
              placeholder="Naam" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">E-mail</label>
            <input type="email" value={form.email} onChange={e => setForm(f => ({ ...f, email: e.target.value }))}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-[#22d3d3]/40 focus:border-[#22d3d3]/60"
              placeholder="email@domein.nl" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Wachtwoord</label>
            <input type="password" value={form.password} onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-[#22d3d3]/40 focus:border-[#22d3d3]/60"
              placeholder="Min. 8 tekens" />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Rol</label>
            <select value={form.role} onChange={e => setForm(f => ({ ...f, role: e.target.value as 'user' | 'admin' | 'super_admin' }))}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-[#22d3d3]/40">
              <option value="user">User</option>
              {canCreateAdmin && <option value="admin">Admin</option>}
              {canCreateAdmin && <option value="super_admin">Super Admin</option>}
            </select>
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button onClick={onClose} className="flex-1 px-4 py-2 rounded-xl border border-slate-200 text-slate-600 text-sm font-medium hover:bg-slate-50 transition-all">
            Annuleren
          </button>
          <button
            onClick={() => mutation.mutate(form)}
            disabled={mutation.isPending || !form.email || !form.name || !form.password}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-xl text-sm font-medium hover:from-[#1ab8b8] hover:to-[#e8a8d8] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {mutation.isPending && <FontAwesomeIcon icon={faSpinner} className="animate-spin w-3.5 h-3.5" />}
            Aanmaken
          </button>
        </div>
      </div>
    </div>
  );
}

function EditUserModal({ user, currentUserRole, onClose }: { user: User; currentUserRole: string; onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState<UpdateUserRequest>({ name: user.name, role: user.role, is_active: user.is_active });
  const [error, setError] = useState('');

  const canChangeRole = currentUserRole === 'super_admin';
  const canEditRole = canChangeRole || (currentUserRole === 'admin' && user.role === 'user');

  const mutation = useMutation({
    mutationFn: (data: UpdateUserRequest) => updateUser(user.id, data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['users'] }); onClose(); },
    onError: (e: unknown) => setError((e as { response?: { data?: { detail?: string } } }).response?.data?.detail ?? 'Fout bij bijwerken'),
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/30 backdrop-blur-sm">
      <div className="glass-card w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-slate-800">Gebruiker bewerken</h2>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-all">
            <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
          </button>
        </div>

        <p className="text-sm text-slate-500 mb-5">{user.email}</p>

        {error && <div className="mb-4 px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">{error}</div>}

        <div className="flex flex-col gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">Naam</label>
            <input value={form.name ?? ''} onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              className="w-full px-3 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-[#22d3d3]/40 focus:border-[#22d3d3]/60" />
          </div>
          {canEditRole && (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Rol</label>
              <select value={form.role ?? user.role} onChange={e => setForm(f => ({ ...f, role: e.target.value as 'user' | 'admin' | 'super_admin' }))}
                className="w-full px-3 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-[#22d3d3]/40">
                <option value="user">User</option>
                <option value="admin">Admin</option>
                {canChangeRole && <option value="super_admin">Super Admin</option>}
              </select>
            </div>
          )}
          <div className="flex items-center justify-between px-3 py-2.5 rounded-xl border border-slate-200 bg-white">
            <span className="text-sm font-medium text-slate-700">Actief</span>
            <button onClick={() => setForm(f => ({ ...f, is_active: !f.is_active }))}
              className={`transition-colors ${form.is_active ? 'text-emerald-500' : 'text-slate-300'}`}>
              <FontAwesomeIcon icon={form.is_active ? faToggleOn : faToggleOff} className="w-6 h-6" />
            </button>
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button onClick={onClose} className="flex-1 px-4 py-2 rounded-xl border border-slate-200 text-slate-600 text-sm font-medium hover:bg-slate-50 transition-all">
            Annuleren
          </button>
          <button
            onClick={() => mutation.mutate(form)}
            disabled={mutation.isPending}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-xl text-sm font-medium hover:from-[#1ab8b8] hover:to-[#e8a8d8] transition-all disabled:opacity-50"
          >
            {mutation.isPending && <FontAwesomeIcon icon={faSpinner} className="animate-spin w-3.5 h-3.5" />}
            Opslaan
          </button>
        </div>
      </div>
    </div>
  );
}

function ResetPasswordModal({ user, onClose }: { user: User; onClose: () => void }) {
  const [password, setPassword] = useState('');
  const [done, setDone] = useState(false);
  const [error, setError] = useState('');

  const mutation = useMutation({
    mutationFn: (pw: string) => resetUserPassword(user.id, pw),
    onSuccess: () => setDone(true),
    onError: (e: unknown) => setError((e as { response?: { data?: { detail?: string } } }).response?.data?.detail ?? 'Fout bij reset'),
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/30 backdrop-blur-sm">
      <div className="glass-card w-full max-w-sm p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-slate-800">Wachtwoord resetten</h2>
          <button onClick={onClose} className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-all">
            <FontAwesomeIcon icon={faTimes} className="w-4 h-4" />
          </button>
        </div>

        {done ? (
          <div className="flex flex-col items-center gap-3 py-4">
            <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center">
              <FontAwesomeIcon icon={faCheck} className="text-emerald-600 text-xl" />
            </div>
            <p className="text-sm text-slate-700 font-medium">Wachtwoord bijgewerkt</p>
            <button onClick={onClose} className="mt-2 px-5 py-2 rounded-xl border border-slate-200 text-slate-600 text-sm font-medium hover:bg-slate-50 transition-all">
              Sluiten
            </button>
          </div>
        ) : (
          <>
            <p className="text-sm text-slate-500 mb-5">{user.email}</p>
            {error && <div className="mb-4 px-3 py-2 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">{error}</div>}
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Nieuw wachtwoord</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)}
                placeholder="Min. 8 tekens"
                className="w-full px-3 py-2 rounded-xl border border-slate-200 bg-white text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-[#22d3d3]/40 focus:border-[#22d3d3]/60" />
            </div>
            <div className="flex gap-3">
              <button onClick={onClose} className="flex-1 px-4 py-2 rounded-xl border border-slate-200 text-slate-600 text-sm font-medium hover:bg-slate-50 transition-all">
                Annuleren
              </button>
              <button
                onClick={() => mutation.mutate(password)}
                disabled={mutation.isPending || password.length < 8}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-xl text-sm font-medium hover:from-[#1ab8b8] hover:to-[#e8a8d8] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {mutation.isPending && <FontAwesomeIcon icon={faSpinner} className="animate-spin w-3.5 h-3.5" />}
                Resetten
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function UsersPage() {
  const { user: currentUser } = useAuth();
  const qc = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  const [editUser, setEditUser] = useState<User | null>(null);
  const [resetUser, setResetUser] = useState<User | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<User | null>(null);

  const { data: users = [], isLoading } = useQuery({ queryKey: ['users'], queryFn: listUsers });

  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteUser(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['users'] }); setDeleteConfirm(null); },
  });

  const isSuperAdmin = currentUser?.role === 'super_admin';
  const isAdmin = currentUser?.role === 'admin' || isSuperAdmin;

  if (!isAdmin) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <p className="text-slate-500 text-sm">Geen toegang</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Gebruikers</h1>
          <p className="text-sm text-slate-500 mt-0.5">Beheer toegang en rollen</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2 sm:py-2.5 bg-gradient-to-r from-[#22d3d3] to-[#FFC1F3] text-white rounded-xl hover:from-[#1ab8b8] hover:to-[#e8a8d8] transition-all font-medium shadow-md shadow-[#FFC1F3]/20 text-sm whitespace-nowrap"
        >
          <FontAwesomeIcon icon={faPlus} className="w-3.5 h-3.5" />
          <span>Nieuwe gebruiker</span>
        </button>
      </div>

      <div className="glass-card overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <FontAwesomeIcon icon={faSpinner} className="text-slate-300 text-2xl animate-spin" />
          </div>
        ) : users.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <FontAwesomeIcon icon={faUser} className="text-3xl mb-3" />
            <p className="text-sm">Geen gebruikers gevonden</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100">
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Naam</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide hidden sm:table-cell">E-mail</th>
                  <th className="text-left px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Rol</th>
                  <th className="text-center px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Status</th>
                  <th className="text-right px-5 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Acties</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u, i) => (
                  <tr key={u.id} className={`${i > 0 ? 'border-t border-slate-50' : ''} hover:bg-slate-50/50 transition-colors`}>
                    <td className="px-5 py-3.5">
                      <div className="font-medium text-slate-800">{u.name}</div>
                      <div className="text-xs text-slate-400 sm:hidden">{u.email}</div>
                    </td>
                    <td className="px-5 py-3.5 text-slate-600 hidden sm:table-cell">{u.email}</td>
                    <td className="px-5 py-3.5"><RoleBadge role={u.role} /></td>
                    <td className="px-5 py-3.5 text-center">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${u.is_active ? 'text-emerald-700 bg-emerald-50 border-emerald-200' : 'text-slate-500 bg-slate-100 border-slate-200'}`}>
                        {u.is_active ? 'Actief' : 'Inactief'}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center justify-end gap-1">
                        <button onClick={() => setEditUser(u)} title="Bewerken"
                          className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-all">
                          <FontAwesomeIcon icon={faEdit} className="w-3.5 h-3.5" />
                        </button>
                        <button onClick={() => setResetUser(u)} title="Wachtwoord resetten"
                          className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all">
                          <FontAwesomeIcon icon={faKey} className="w-3.5 h-3.5" />
                        </button>
                        {isSuperAdmin && u.id !== currentUser?.user_id && (
                          <button onClick={() => setDeleteConfirm(u)} title="Verwijderen"
                            className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all">
                            <FontAwesomeIcon icon={faTrash} className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {deleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/30 backdrop-blur-sm">
          <div className="glass-card w-full max-w-sm p-6">
            <h3 className="text-base font-semibold text-slate-800 mb-2">Gebruiker verwijderen</h3>
            <p className="text-sm text-slate-600 mb-5">
              Weet je zeker dat je <strong>{deleteConfirm.name}</strong> wilt verwijderen? Dit kan niet ongedaan worden gemaakt.
            </p>
            <div className="flex gap-3">
              <button onClick={() => setDeleteConfirm(null)} className="flex-1 px-4 py-2 rounded-xl border border-slate-200 text-slate-600 text-sm font-medium hover:bg-slate-50 transition-all">
                Annuleren
              </button>
              <button
                onClick={() => deleteMutation.mutate(deleteConfirm.id)}
                disabled={deleteMutation.isPending}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-500 hover:bg-red-600 text-white rounded-xl text-sm font-medium transition-all disabled:opacity-50"
              >
                {deleteMutation.isPending && <FontAwesomeIcon icon={faSpinner} className="animate-spin w-3.5 h-3.5" />}
                Verwijderen
              </button>
            </div>
          </div>
        </div>
      )}

      {showCreate && <CreateUserModal onClose={() => setShowCreate(false)} canCreateAdmin={isSuperAdmin} />}
      {editUser && <EditUserModal user={editUser} currentUserRole={currentUser?.role ?? 'user'} onClose={() => setEditUser(null)} />}
      {resetUser && <ResetPasswordModal user={resetUser} onClose={() => setResetUser(null)} />}
    </div>
  );
}
