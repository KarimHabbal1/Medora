import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { patientApi } from '../../api/patient';
import Card from '../../components/ui/Card';
import Button from '../../components/ui/Button';
import TextArea from '../../components/ui/TextArea';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import type { MedicalHistory, MedicalHistoryUpdate } from '../../types/patient';

function parseList(field: Record<string, unknown> | null): string[] {
  if (!field) return [];
  if (Array.isArray(field.items)) return field.items as string[];
  if (Array.isArray(field)) return field as string[];
  return [];
}

function toList(items: string[]): Record<string, unknown> {
  return { items: items.filter((i) => i.trim()) };
}

const MedicalHistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const [history, setHistory] = useState<MedicalHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [chronic, setChronic] = useState('');
  const [medications, setMedications] = useState('');
  const [allergies, setAllergies] = useState('');
  const [surgeries, setSurgeries] = useState('');
  const [familyHistory, setFamilyHistory] = useState('');
  const [smokingStatus, setSmokingStatus] = useState('');
  const [pregnancyStatus, setPregnancyStatus] = useState('');
  const [notes, setNotes] = useState('');

  useEffect(() => {
    patientApi.getMedicalHistory().then((data) => {
      setHistory(data);
      setChronic(parseList(data.chronic_conditions).join(', '));
      setMedications(parseList(data.medications).join(', '));
      setAllergies(parseList(data.allergies).join(', '));
      setSurgeries(parseList(data.surgeries).join(', '));
      setFamilyHistory(parseList(data.family_history).join(', '));
      setSmokingStatus(data.smoking_status || '');
      setPregnancyStatus(data.pregnancy_status || '');
      setNotes(data.additional_notes || '');
    }).catch(() => setError('Failed to load medical history.'))
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async (skip = false) => {
    setSaving(true); setError(''); setSuccess('');
    const payload: MedicalHistoryUpdate = skip ? { skipped: true } : {
      chronic_conditions: toList(chronic.split(',').map((s) => s.trim())),
      medications: toList(medications.split(',').map((s) => s.trim())),
      allergies: toList(allergies.split(',').map((s) => s.trim())),
      surgeries: toList(surgeries.split(',').map((s) => s.trim())),
      family_history: toList(familyHistory.split(',').map((s) => s.trim())),
      smoking_status: smokingStatus || null,
      pregnancy_status: pregnancyStatus || null,
      additional_notes: notes || null,
      skipped: false,
    };
    try {
      const updated = await patientApi.updateMedicalHistory(payload);
      setHistory(updated);
      setSuccess(skip ? 'Medical history skipped.' : 'Medical history saved.');
    } catch { setError('Failed to save medical history.'); }
    finally { setSaving(false); }
  };

  if (loading) return <LoadingSpinner />;

  const selectCls = 'w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-medora-500 hover:border-medora-300';

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      <div>
        <button onClick={() => navigate(-1)} className="text-medora-600 hover:text-medora-700 text-sm font-medium mb-4">← Back</button>
        <h1 className="text-2xl font-bold text-text-primary">Medical History</h1>
        <p className="text-text-secondary mt-1">Separate multiple items with commas.</p>
      </div>
      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}
      {success && <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-700">{success}</div>}
      {history?.skipped && <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-700">You previously skipped. You can update anytime.</div>}
      <Card>
        <div className="space-y-5">
          <TextArea label="Chronic Conditions" placeholder="e.g., Diabetes, Hypertension" value={chronic} onChange={(e) => setChronic(e.target.value)} />
          <TextArea label="Current Medications" placeholder="e.g., Metformin 500mg" value={medications} onChange={(e) => setMedications(e.target.value)} />
          <TextArea label="Allergies" placeholder="e.g., Penicillin, Shellfish" value={allergies} onChange={(e) => setAllergies(e.target.value)} />
          <TextArea label="Past Surgeries" placeholder="e.g., Appendectomy (2018)" value={surgeries} onChange={(e) => setSurgeries(e.target.value)} />
          <TextArea label="Family History" placeholder="e.g., Heart disease (father)" value={familyHistory} onChange={(e) => setFamilyHistory(e.target.value)} />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-primary mb-1.5">Smoking Status</label>
              <select className={selectCls} value={smokingStatus} onChange={(e) => setSmokingStatus(e.target.value)}>
                <option value="">Select...</option>
                <option value="never">Never smoked</option>
                <option value="former">Former smoker</option>
                <option value="current">Current smoker</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-text-primary mb-1.5">Pregnancy Status</label>
              <select className={selectCls} value={pregnancyStatus} onChange={(e) => setPregnancyStatus(e.target.value)}>
                <option value="">Select...</option>
                <option value="not_applicable">Not applicable</option>
                <option value="not_pregnant">Not pregnant</option>
                <option value="pregnant">Pregnant</option>
                <option value="postpartum">Postpartum</option>
              </select>
            </div>
          </div>
          <TextArea label="Additional Notes" placeholder="Any other relevant info..." value={notes} onChange={(e) => setNotes(e.target.value)} />
          <div className="flex items-center justify-between pt-2">
            <Button variant="ghost" onClick={() => handleSave(true)} disabled={saving}>Skip for now</Button>
            <Button onClick={() => handleSave(false)} loading={saving}>Save Medical History</Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default MedicalHistoryPage;
