import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { patientApi } from '../../api/patient';
import { triageApi } from '../../api/triage';
import Card from '../../components/ui/Card';
import Button from '../../components/ui/Button';
import Badge from '../../components/ui/Badge';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import EmptyState from '../../components/ui/EmptyState';
import type { PatientProfile } from '../../types/patient';
import type { TriageSession } from '../../types/triage';
import { TriageSessionStatus } from '../../types/enums';

const PatientDashboard: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [profile, setProfile] = useState<PatientProfile | null>(null);
  const [sessions, setSessions] = useState<TriageSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const [profileData, sessionsData] = await Promise.all([
          patientApi.getProfile(),
          triageApi.getSessions(),
        ]);
        setProfile(profileData);
        setSessions(sessionsData);
      } catch {
        setError('Failed to load dashboard data.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return <LoadingSpinner />;

  const profileComplete = profile?.date_of_birth && profile?.sex;
  const recentSessions = sessions.slice(0, 5);

  const statusBadge = (status: TriageSessionStatus) => {
    switch (status) {
      case TriageSessionStatus.Active:
        return <Badge variant="primary">Active</Badge>;
      case TriageSessionStatus.Completed:
        return <Badge variant="success">Completed</Badge>;
      case TriageSessionStatus.Cancelled:
        return <Badge variant="default">Cancelled</Badge>;
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      {/* Welcome */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary">
          Welcome back, {user?.full_name?.split(' ')[0]} 👋
        </h1>
        <p className="text-text-secondary mt-1">Here&apos;s an overview of your health journey</p>
      </div>

      {/* Quick cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Profile card */}
        <Card hover onClick={() => navigate('/patient/profile')}>
          <div className="flex items-center gap-3 mb-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${profileComplete ? 'bg-emerald-50' : 'bg-amber-50'}`}>
              <svg className={`h-5 w-5 ${profileComplete ? 'text-success' : 'text-warning'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-text-primary">Profile</p>
              <p className="text-xs text-text-secondary">
                {profileComplete ? 'Complete' : 'Incomplete — tap to update'}
              </p>
            </div>
          </div>
          <Badge variant={profileComplete ? 'success' : 'warning'} size="sm">
            {profileComplete ? '✓ Complete' : 'Action needed'}
          </Badge>
        </Card>

        {/* Medical history card */}
        <Card hover onClick={() => navigate('/patient/medical-history')}>
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg bg-medora-50 flex items-center justify-center">
              <svg className="h-5 w-5 text-medora-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-text-primary">Medical History</p>
              <p className="text-xs text-text-secondary">View or update your records</p>
            </div>
          </div>
          <Badge variant="primary" size="sm">View →</Badge>
        </Card>

        {/* Start triage */}
        <Card className="bg-gradient-to-br from-medora-500 to-teal-500 border-0 text-white">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-lg bg-white/20 flex items-center justify-center">
              <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium">New Triage Session</p>
              <p className="text-xs text-white/80">Describe your symptoms</p>
            </div>
          </div>
          <Button
            variant="secondary"
            size="sm"
            className="bg-white text-medora-700 hover:bg-white/90 border-0"
            onClick={() => navigate('/patient/triage')}
          >
            Start Now →
          </Button>
        </Card>
      </div>

      {/* Recent sessions */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Recent Triage Sessions</h2>
          <Button variant="ghost" size="sm" onClick={() => navigate('/patient/triage')}>
            View All
          </Button>
        </div>

        {recentSessions.length === 0 ? (
          <EmptyState
            title="No sessions yet"
            description="Start a new triage session to get an AI-powered assessment"
            actionLabel="Start Triage"
            onAction={() => navigate('/patient/triage')}
          />
        ) : (
          <div className="space-y-3">
            {recentSessions.map((session) => (
              <div
                key={session.id}
                className="flex items-center justify-between p-3 rounded-lg border border-border-light hover:bg-surface-tertiary transition-colors cursor-pointer"
                onClick={() => navigate(`/patient/triage/${session.id}`)}
              >
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg bg-medora-50 flex items-center justify-center">
                    <svg className="h-4 w-4 text-medora-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-text-primary">
                      {session.chief_complaint || 'Triage Session'}
                    </p>
                    <p className="text-xs text-text-tertiary">
                      {new Date(session.started_at).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </p>
                  </div>
                </div>
                {statusBadge(session.status)}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};

export default PatientDashboard;
