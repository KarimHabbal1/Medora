import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { doctorApi } from '../../api/doctor';
import Card from '../../components/ui/Card';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import UrgencyBadge from '../../components/doctor/UrgencyBadge';
import ReportSectionCard from '../../components/doctor/ReportSectionCard';
import FeedbackForm from '../../components/doctor/FeedbackForm';
import type { ClinicalReport } from '../../types/triage';

const DoctorReportDetail: React.FC = () => {
  const { reportId } = useParams<{ reportId: string }>();
  const [report, setReport] = useState<ClinicalReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!reportId) return;
    doctorApi.getReport(reportId).then(setReport).catch(() => setError('Failed to load report.')).finally(() => setLoading(false));
  }, [reportId]);

  if (loading) return <LoadingSpinner />;
  if (!report) return <ErrorAlert message={error || 'Report not found.'} />;

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Clinical Report</h1>
          <p className="text-sm text-text-secondary mt-1">
            Generated {new Date(report.generated_at).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
        <UrgencyBadge level={report.urgency_level} size="md" />
      </div>

      {/* Summary */}
      <Card className="bg-medora-50/50 border-medora-200">
        <h3 className="text-sm font-semibold text-medora-800 mb-2">Summary</h3>
        <p className="text-sm text-text-primary leading-relaxed">{report.summary_text}</p>
      </Card>

      {/* Sections */}
      <div className="space-y-4">
        <ReportSectionCard title="Presenting Complaints" content={report.presenting_complaints} />
        <ReportSectionCard title="History of Presenting Complaint" content={report.history_of_presenting_complaint} />
        <ReportSectionCard title="Suspected Conditions" content={report.suspected_conditions} />
        <ReportSectionCard title="Triggered Red Flags" content={report.triggered_red_flags} variant={report.triggered_red_flags ? 'warning' : 'default'} />
        <ReportSectionCard title="Recommended Action" content={report.recommended_action} />
        <ReportSectionCard title="Specialty Routing" content={report.specialty_routing} />
        <ReportSectionCard title="Suggested Workup" content={report.suggested_workup} />
        <ReportSectionCard title="Key Exam Findings" content={report.key_exam_findings} />
        <ReportSectionCard title="Admission Criteria" content={report.admission_criteria} />
        <ReportSectionCard title="Referral Criteria" content={report.referral_criteria} />
      </div>

      {/* Escalation */}
      {report.escalation_message && (
        <Card className="border-red-200 bg-red-50/50">
          <h3 className="text-sm font-semibold text-red-800 mb-2">⚠ Escalation Message</h3>
          <p className="text-sm text-red-700">{report.escalation_message}</p>
        </Card>
      )}

      {/* Meta */}
      <Card>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-xs text-text-tertiary">Model Version</p>
            <p className="font-medium text-text-primary">{report.model_version || '—'}</p>
          </div>
          <div>
            <p className="text-xs text-text-tertiary">External Escalation</p>
            <p className="font-medium text-text-primary">{report.external_escalation_completed ? 'Yes' : 'No'}</p>
          </div>
        </div>
      </Card>

      {/* Feedback */}
      <Card>
        <FeedbackForm reportId={report.id} />
      </Card>
    </div>
  );
};

export default DoctorReportDetail;
