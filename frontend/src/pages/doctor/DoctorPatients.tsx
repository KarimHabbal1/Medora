import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { doctorApi } from '../../api/doctor';
import Card from '../../components/ui/Card';
import Input from '../../components/ui/Input';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import EmptyState from '../../components/ui/EmptyState';
import type { PatientSummary } from '../../types/doctor';

const DoctorPatients: React.FC = () => {
  const navigate = useNavigate();
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');

  useEffect(() => {
    doctorApi.getPatients().then(setPatients).catch(() => setError('Failed to load patients.')).finally(() => setLoading(false));
  }, []);

  const filtered = patients.filter((p) => p.full_name.toLowerCase().includes(search.toLowerCase()));

  if (loading) return <LoadingSpinner />;

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Patients</h1>
        <p className="text-text-secondary mt-1">Your assigned patients</p>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      <Input placeholder="Search by name..." value={search} onChange={(e) => setSearch(e.target.value)} icon={<svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" /></svg>} />

      {filtered.length === 0 ? (
        <EmptyState title="No patients found" description={search ? 'Try a different search term' : 'No patients assigned yet'} />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {filtered.map((p) => (
            <Card key={p.id} hover onClick={() => navigate(`/doctor/patients/${p.id}`)}>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-medora-100 flex items-center justify-center text-medora-700 font-semibold text-sm">
                  {p.full_name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase()}
                </div>
                <div>
                  <p className="text-sm font-medium text-text-primary">{p.full_name}</p>
                  <p className="text-xs text-text-tertiary">
                    {p.last_triage ? `Last triage: ${new Date(p.last_triage).toLocaleDateString()}` : 'No triage yet'}
                  </p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default DoctorPatients;
