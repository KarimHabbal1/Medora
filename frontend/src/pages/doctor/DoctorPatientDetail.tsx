import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { doctorApi } from '../../api/doctor';
import Card from '../../components/ui/Card';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import EmptyState from '../../components/ui/EmptyState';
import UrgencyBadge from '../../components/doctor/UrgencyBadge';
import type { PatientDetail, ReportSummary } from '../../types/doctor';

const DoctorPatientDetail: React.FC = () => {
  const { patientId } = useParams<{ patientId: string }>();
  const navigate = useNavigate();
  const [patient, setPatient] = useState<PatientDetail | null>(null);
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!patientId) return;
    const load = async () => {
      try {
        const [p, r] = await Promise.all([doctorApi.getPatient(patientId), doctorApi.getPatientReports(patientId)]);
        setPatient(p);
        setReports(r);
      } catch { setError('Failed to load patient data.'); }
      finally { setLoading(false); }
    };
    load();
  }, [patientId]);

  if (loading) return <LoadingSpinner />;

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      {patient && (
        <>
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-full bg-medora-100 flex items-center justify-center text-medora-700 font-bold text-lg">
              {patient.full_name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase()}
            </div>
            <div>
              <h1 className="text-2xl font-bold text-text-primary">{patient.full_name}</h1>
              <p className="text-sm text-text-secondary">Patient Details</p>
            </div>
          </div>

          {/* Demographics */}
          <Card>
            <h2 className="text-base font-semibold text-text-primary mb-4">Demographics</h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-text-tertiary">Date of Birth</p>
                <p className="text-sm font-medium text-text-primary">{patient.date_of_birth || '—'}</p>
              </div>
              <div>
                <p className="text-xs text-text-tertiary">Sex</p>
                <p className="text-sm font-medium text-text-primary capitalize">{patient.sex || '—'}</p>
              </div>
              <div>
                <p className="text-xs text-text-tertiary">Height</p>
                <p className="text-sm font-medium text-text-primary">{patient.height_cm ? `${patient.height_cm} cm` : '—'}</p>
              </div>
              <div>
                <p className="text-xs text-text-tertiary">Weight</p>
                <p className="text-sm font-medium text-text-primary">{patient.weight_kg ? `${patient.weight_kg} kg` : '—'}</p>
              </div>
            </div>
          </Card>

          {/* Medical History — not available from this endpoint */}
          <Card>
            <h2 className="text-base font-semibold text-text-primary mb-2">Medical History</h2>
            <p className="text-sm text-text-tertiary italic">Medical history unavailable from this view.</p>
          </Card>
        </>
      )}

      {/* Report History */}
      <Card>
        <h2 className="text-base font-semibold text-text-primary mb-4">Report History</h2>
        {reports.length === 0 ? (
          <EmptyState title="No reports" description="No reports available for this patient" />
        ) : (
          <div className="space-y-3">
            {reports.map((r) => (
              <div key={r.id} className="flex items-center justify-between p-3 rounded-lg border border-border-light hover:bg-surface-tertiary transition-colors cursor-pointer" onClick={() => navigate(`/doctor/reports/${r.id}`)}>
                <div>
                  <p className="text-sm font-medium text-text-primary">Clinical Report</p>
                  <p className="text-xs text-text-tertiary">{new Date(r.generated_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</p>
                </div>
                <UrgencyBadge level={r.urgency_level} />
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};

export default DoctorPatientDetail;
