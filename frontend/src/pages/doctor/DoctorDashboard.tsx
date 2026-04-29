import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { doctorApi } from '../../api/doctor';
import Card from '../../components/ui/Card';
import Button from '../../components/ui/Button';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import EmptyState from '../../components/ui/EmptyState';
import UrgencyBadge from '../../components/doctor/UrgencyBadge';
import type { DoctorDashboardData, ReportSummary } from '../../types/doctor';

const DoctorDashboard: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [dashboard, setDashboard] = useState<DoctorDashboardData | null>(null);
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const [d, r] = await Promise.all([doctorApi.getDashboard(), doctorApi.getReports()]);
        setDashboard(d);
        setReports(r);
      } catch { setError('Failed to load dashboard.'); }
      finally { setLoading(false); }
    };
    load();
  }, []);

  if (loading) return <LoadingSpinner />;

  const recent = reports.slice(0, 5);

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      <div>
        <h1 className="text-2xl font-bold text-text-primary">Welcome, Dr. {user?.full_name?.split(' ').slice(-1)[0]}</h1>
        <p className="text-text-secondary mt-1">Here's your clinical overview</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card hover onClick={() => navigate('/doctor/patients')}>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-medora-50 flex items-center justify-center">
              <svg className="h-5 w-5 text-medora-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z" /></svg>
            </div>
            <div>
              <p className="text-2xl font-bold text-text-primary">{dashboard?.total_patients ?? 0}</p>
              <p className="text-xs text-text-secondary">Assigned Patients</p>
            </div>
          </div>
        </Card>

        <Card hover onClick={() => navigate('/doctor/reports')}>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-amber-50 flex items-center justify-center">
              <svg className="h-5 w-5 text-warning" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" /></svg>
            </div>
            <div>
              <p className="text-2xl font-bold text-text-primary">{dashboard?.pending_reports ?? 0}</p>
              <p className="text-xs text-text-secondary">Pending Reports</p>
            </div>
          </div>
        </Card>

        <Card hover onClick={() => navigate('/doctor/reports')}>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-teal-50 flex items-center justify-center">
              <svg className="h-5 w-5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>
            </div>
            <div>
              <p className="text-2xl font-bold text-text-primary">{reports.length}</p>
              <p className="text-xs text-text-secondary">Total Reports</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Recent Reports */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Recent Reports</h2>
          <Button variant="ghost" size="sm" onClick={() => navigate('/doctor/reports')}>View All</Button>
        </div>
        {recent.length === 0 ? (
          <EmptyState title="No reports yet" description="Reports will appear here once patients complete triage sessions" />
        ) : (
          <div className="space-y-3">
            {recent.map((r) => (
              <div key={r.id} className="flex items-center justify-between p-3 rounded-lg border border-border-light hover:bg-surface-tertiary transition-colors cursor-pointer" onClick={() => navigate(`/doctor/reports/${r.id}`)}>
                <div>
                  <p className="text-sm font-medium text-text-primary">{r.patient_name}</p>
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

export default DoctorDashboard;
