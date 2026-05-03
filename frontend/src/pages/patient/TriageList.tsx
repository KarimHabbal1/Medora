import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { triageApi } from '../../api/triage';
import Card from '../../components/ui/Card';
import Button from '../../components/ui/Button';
import Badge from '../../components/ui/Badge';
import Input from '../../components/ui/Input';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import EmptyState from '../../components/ui/EmptyState';
import type { PatientTriageSession } from '../../types/triage';
import { TriageSessionStatus } from '../../types/enums';

const TriageList: React.FC = () => {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<PatientTriageSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [complaint, setComplaint] = useState('');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    triageApi.getSessions().then(setSessions).catch(() => setError('Failed to load sessions.')).finally(() => setLoading(false));
  }, []);

  const handleCreate = async () => {
    setCreating(true);
    try {
      const session = await triageApi.createSession({ chief_complaint: complaint || undefined });
      navigate(`/patient/triage/${session.id}`);
    } catch { setError('Failed to create session.'); }
    finally { setCreating(false); }
  };

  const statusBadge = (s: TriageSessionStatus) => {
    const map = { [TriageSessionStatus.Active]: 'primary', [TriageSessionStatus.Completed]: 'success', [TriageSessionStatus.Cancelled]: 'default' } as const;
    return <Badge variant={map[s]}>{s}</Badge>;
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Triage Sessions</h1>
          <p className="text-text-secondary mt-1">Your AI-assisted health assessments</p>
        </div>
        <Button onClick={() => setShowModal(true)}>New Session</Button>
      </div>

      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30" onClick={() => setShowModal(false)}>
          <div className="w-full max-w-md mx-4" onClick={(e) => e.stopPropagation()}>
          <Card className="shadow-modal">
            <h2 className="text-lg font-semibold text-text-primary mb-2">Start New Session</h2>
            <p className="text-sm text-text-secondary mb-4">Briefly describe your main concern (optional).</p>
            <Input placeholder="e.g., Chest pain for 2 days" value={complaint} onChange={(e) => setComplaint(e.target.value)} />
            <div className="flex justify-end gap-2 mt-4">
              <Button variant="secondary" onClick={() => setShowModal(false)}>Cancel</Button>
              <Button onClick={handleCreate} loading={creating}>Start</Button>
            </div>
          </Card>
          </div>
        </div>
      )}

      {sessions.length === 0 ? (
        <EmptyState title="No sessions yet" description="Start a new triage session to begin" actionLabel="New Session" onAction={() => setShowModal(true)} />
      ) : (
        <div className="space-y-3">
          {sessions.map((s) => (
            <Card key={s.id} hover onClick={() => navigate(`/patient/triage/${s.id}`)} padding="sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-text-primary">{s.chief_complaint || 'Triage Session'}</p>
                  <p className="text-xs text-text-tertiary">{new Date(s.started_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}</p>
                </div>
                {statusBadge(s.status)}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default TriageList;
