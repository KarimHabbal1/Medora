import React, { useEffect, useState } from 'react';
import { patientApi } from '../../api/patient';
import Card from '../../components/ui/Card';
import Input from '../../components/ui/Input';
import Button from '../../components/ui/Button';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import type { PatientProfile, PatientUpdate } from '../../types/patient';

const PatientProfilePage: React.FC = () => {
  const [profile, setProfile] = useState<PatientProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [form, setForm] = useState<PatientUpdate>({
    date_of_birth: null,
    sex: null,
    height_cm: null,
    weight_kg: null,
  });

  useEffect(() => {
    const load = async () => {
      try {
        const data = await patientApi.getProfile();
        setProfile(data);
        setForm({
          date_of_birth: data.date_of_birth,
          sex: data.sex,
          height_cm: data.height_cm,
          weight_kg: data.weight_kg,
        });
      } catch {
        setError('Failed to load profile.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const updated = await patientApi.updateProfile(form);
      setProfile(updated);
      setSuccess('Profile updated successfully.');
    } catch {
      setError('Failed to update profile.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Your Profile</h1>
        <p className="text-text-secondary mt-1">Update your personal information</p>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}
      {success && (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-700">
          {success}
        </div>
      )}

      <Card>
        <form onSubmit={handleSave} className="space-y-5">
          {/* Read-only info */}
          {profile && (
            <div className="pb-4 border-b border-border">
              <p className="text-xs text-text-tertiary mb-1">Patient ID</p>
              <p className="text-sm font-mono text-text-secondary">{profile.id}</p>
            </div>
          )}

          <Input
            label="Date of Birth"
            type="date"
            value={form.date_of_birth || ''}
            onChange={(e) => setForm({ ...form, date_of_birth: e.target.value || null })}
          />

          <div>
            <label className="block text-sm font-medium text-text-primary mb-1.5">Sex</label>
            <select
              className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary transition-colors focus:outline-none focus:ring-2 focus:ring-medora-500 focus:border-medora-500 hover:border-medora-300"
              value={form.sex || ''}
              onChange={(e) => setForm({ ...form, sex: e.target.value || null })}
            >
              <option value="">Select...</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Height (cm)"
              type="number"
              placeholder="170"
              value={form.height_cm ?? ''}
              onChange={(e) =>
                setForm({ ...form, height_cm: e.target.value ? Number(e.target.value) : null })
              }
            />
            <Input
              label="Weight (kg)"
              type="number"
              placeholder="70"
              value={form.weight_kg ?? ''}
              onChange={(e) =>
                setForm({ ...form, weight_kg: e.target.value ? Number(e.target.value) : null })
              }
            />
          </div>

          <div className="flex justify-end">
            <Button type="submit" loading={saving}>
              Save Changes
            </Button>
          </div>
        </form>
      </Card>
    </div>
  );
};

export default PatientProfilePage;
