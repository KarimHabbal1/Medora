import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { doctorApi } from '../../api/doctor';
import Card from '../../components/ui/Card';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import EmptyState from '../../components/ui/EmptyState';
import UrgencyBadge from '../../components/doctor/UrgencyBadge';
import type { ReportSummary } from '../../types/doctor';

const DoctorReports: React.FC = () => {
  const navigate = useNavigate();
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    doctorApi.getReports().then(setReports).catch(() => setError('Failed to load reports.')).finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner />;

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Clinical Reports</h1>
        <p className="text-text-secondary mt-1">AI-generated triage reports for review</p>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      {reports.length === 0 ? (
        <EmptyState title="No reports yet" description="Reports will appear here when patients complete triage sessions" />
      ) : (
        <Card padding="none">
          <div className="divide-y divide-border">
            {reports.map((r) => (
              <div key={r.id} className="flex items-center justify-between px-5 py-4 hover:bg-surface-tertiary transition-colors cursor-pointer" onClick={() => navigate(`/doctor/reports/${r.id}`)}>
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-full bg-medora-100 flex items-center justify-center text-medora-700 font-semibold text-xs">
                    {r.patient_name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase()}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-text-primary">{r.patient_name}</p>
                    <p className="text-xs text-text-tertiary">{new Date(r.generated_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}</p>
                  </div>
                </div>
                <UrgencyBadge level={r.urgency_level} />
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default DoctorReports;
