import React, { useEffect, useState } from 'react';
import { adminApi } from '../../api/admin';
import Card from '../../components/ui/Card';
import Button from '../../components/ui/Button';
import Input from '../../components/ui/Input';
import Badge from '../../components/ui/Badge';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import type { AdminUser, AdminUserCreate, AdminUserUpdate, Hospital } from '../../types/admin';
import { UserRole } from '../../types/enums';

const AdminUsers: React.FC = () => {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [creating, setCreating] = useState(false);
  const [editId, setEditId] = useState<string | null>(null);
  const [editData, setEditData] = useState<AdminUserUpdate>({});
  const [saving, setSaving] = useState(false);

  // Create form
  const [cEmail, setCEmail] = useState('');
  const [cPassword, setCPassword] = useState('');
  const [cName, setCName] = useState('');
  const [cPhone, setCPhone] = useState('');
  const [cRole, setCRole] = useState<UserRole>(UserRole.Patient);
  const [cHospitalId, setCHospitalId] = useState('');

  // Hospitals for dropdown
  const [hospitals, setHospitals] = useState<Hospital[]>([]);

  const loadUsers = () => {
    setLoading(true);
    adminApi.getUsers().then(setUsers).catch(() => setError('Failed to load users.')).finally(() => setLoading(false));
  };

  useEffect(() => {
    loadUsers();
    adminApi.getHospitals().then(setHospitals).catch(() => { /* non-critical, dropdown will be empty */ });
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();

    // Guard: hospital_id is required
    if (!cHospitalId.trim()) {
      setError('Hospital ID is required.');
      return;
    }

    setCreating(true);
    setError('');
    try {
      const data: AdminUserCreate = {
        email: cEmail,
        password: cPassword,
        full_name: cName,
        phone: cPhone || undefined,
        role: cRole,
        hospital_id: cHospitalId,
      };
      await adminApi.createUser(data);

      // Success: refresh list, reset form, close panel
      loadUsers();
      setCEmail('');
      setCPassword('');
      setCName('');
      setCPhone('');
      setCRole(UserRole.Patient);
      setCHospitalId('');
      setShowCreate(false);
    } catch (err: unknown) {
      console.error('Create user failed:', (err as { response?: { data?: unknown } })?.response?.data || err);

      // Safely extract a string from whatever FastAPI sends back
      const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      let msg: string;
      if (typeof detail === 'string') {
        msg = detail;
      } else if (Array.isArray(detail)) {
        // FastAPI 422 validation errors: [{msg, loc, type}, ...]
        msg = detail
          .map((d: { msg?: string; loc?: (string | number)[] }) => {
            const field = d.loc ? d.loc.filter((l) => l !== 'body').join(' → ') : '';
            return field ? `${field}: ${d.msg ?? ''}` : (d.msg ?? '');
          })
          .join(' | ');
      } else if (detail && typeof detail === 'object') {
        msg = JSON.stringify(detail);
      } else {
        msg = 'Failed to create user.';
      }

      setError(msg);
    } finally {
      setCreating(false);
    }
  };

  const handleUpdate = async (userId: string) => {
    setSaving(true);
    setError('');
    try {
      await adminApi.updateUser(userId, editData);
      setEditId(null);
      setEditData({});
      loadUsers();
    } catch { setError('Failed to update user.'); }
    finally { setSaving(false); }
  };

  const roleBadge = (role: UserRole) => {
    const v = { [UserRole.Patient]: 'primary', [UserRole.Doctor]: 'success', [UserRole.Admin]: 'warning' } as const;
    return <Badge variant={v[role]}>{role}</Badge>;
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">User Management</h1>
          <p className="text-text-secondary mt-1">{users.length} users total</p>
        </div>
        <Button onClick={() => setShowCreate(!showCreate)}>
          {showCreate ? 'Cancel' : 'Create User'}
        </Button>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      {/* Create form */}
      {showCreate && (
        <Card>
          <h2 className="text-base font-semibold text-text-primary mb-4">Create New User</h2>
          <form onSubmit={handleCreate} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Input label="Full Name" required value={cName} onChange={(e) => setCName(e.target.value)} />
              <Input label="Email" type="email" required value={cEmail} onChange={(e) => setCEmail(e.target.value)} />
              <Input label="Password" type="password" required value={cPassword} onChange={(e) => setCPassword(e.target.value)} />
              <Input label="Phone" value={cPhone} onChange={(e) => setCPhone(e.target.value)} />
              <div>
                <label className="block text-sm font-medium text-text-primary mb-1.5">Role</label>
                <select className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-medora-500" value={cRole} onChange={(e) => setCRole(e.target.value as UserRole)}>
                  <option value="patient">Patient</option>
                  <option value="doctor">Doctor</option>
                  <option value="admin">Admin</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-text-primary mb-1.5">
                  Hospital <span className="text-red-500">*</span>
                </label>
                <select
                  required
                  className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-medora-500 disabled:opacity-50"
                  value={cHospitalId}
                  onChange={(e) => setCHospitalId(e.target.value)}
                >
                  <option value="">— Select a hospital —</option>
                  {hospitals.map((h) => (
                    <option key={h.id} value={h.id}>{h.name}</option>
                  ))}
                </select>
                {hospitals.length === 0 && (
                  <p className="mt-1 text-xs text-text-tertiary">No hospitals found. A hospital is created automatically on first signup.</p>
                )}
              </div>
            </div>
            <div className="flex justify-end">
              <Button type="submit" loading={creating}>Create</Button>
            </div>
          </form>
        </Card>
      )}

      {/* Users table */}
      <Card padding="none">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface-tertiary">
                <th className="text-left px-5 py-3 font-medium text-text-secondary">Name</th>
                <th className="text-left px-5 py-3 font-medium text-text-secondary">Email</th>
                <th className="text-left px-5 py-3 font-medium text-text-secondary">Role</th>
                <th className="text-left px-5 py-3 font-medium text-text-secondary">Active</th>
                <th className="text-left px-5 py-3 font-medium text-text-secondary">Created</th>
                <th className="px-5 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-light">
              {users.map((u) => (
                <tr key={u.id} className="hover:bg-surface-tertiary transition-colors">
                  <td className="px-5 py-3 font-medium text-text-primary">
                    {editId === u.id ? (
                      <Input value={editData.full_name ?? u.full_name} onChange={(e) => setEditData({ ...editData, full_name: e.target.value })} />
                    ) : u.full_name}
                  </td>
                  <td className="px-5 py-3 text-text-secondary">{u.email}</td>
                  <td className="px-5 py-3">
                    {editId === u.id ? (
                      <select className="rounded-lg border border-border px-2 py-1 text-sm" value={editData.role ?? u.role} onChange={(e) => setEditData({ ...editData, role: e.target.value as UserRole })}>
                        <option value="patient">patient</option>
                        <option value="doctor">doctor</option>
                        <option value="admin">admin</option>
                      </select>
                    ) : roleBadge(u.role)}
                  </td>
                  <td className="px-5 py-3">
                    {editId === u.id ? (
                      <select className="rounded-lg border border-border px-2 py-1 text-sm" value={String(editData.is_active ?? u.is_active)} onChange={(e) => setEditData({ ...editData, is_active: e.target.value === 'true' })}>
                        <option value="true">Active</option>
                        <option value="false">Inactive</option>
                      </select>
                    ) : (
                      <Badge variant={u.is_active ? 'success' : 'default'}>{u.is_active ? 'Active' : 'Inactive'}</Badge>
                    )}
                  </td>
                  <td className="px-5 py-3 text-text-tertiary text-xs">{new Date(u.created_at).toLocaleDateString()}</td>
                  <td className="px-5 py-3">
                    {editId === u.id ? (
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => handleUpdate(u.id)} loading={saving}>Save</Button>
                        <Button size="sm" variant="ghost" onClick={() => { setEditId(null); setEditData({}); }}>Cancel</Button>
                      </div>
                    ) : (
                      <Button size="sm" variant="ghost" onClick={() => { setEditId(u.id); setEditData({}); }}>Edit</Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

export default AdminUsers;
