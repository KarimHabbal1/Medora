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

      {/* Web Search Evidence */}
      {report.web_search_results && (
        <Card className="border-teal-200 bg-teal-50/50">
          <h3 className="text-sm font-semibold text-teal-800 mb-3">Web Search Evidence (External Sources)</h3>
          <div className="space-y-3">
            <div>
              <p className="text-xs font-semibold text-text-secondary">Web Diagnosis</p>
              <p className="text-sm text-text-primary font-medium">
                {(report.web_search_results as Record<string, unknown>).primary_diagnosis as string || 'N/A'}
                {(report.web_search_results as Record<string, unknown>).confidence && (
                  <span className="ml-2 text-xs text-teal-600">({(report.web_search_results as Record<string, unknown>).confidence as string} confidence)</span>
                )}
              </p>
            </div>
            {(report.web_search_results as Record<string, unknown>).evidence_summary && (
              <div>
                <p className="text-xs font-semibold text-text-secondary">Evidence Summary</p>
                <p className="text-sm text-text-primary">{(report.web_search_results as Record<string, unknown>).evidence_summary as string}</p>
              </div>
            )}
            {Array.isArray((report.web_search_results as Record<string, unknown>).key_findings) && ((report.web_search_results as Record<string, unknown>).key_findings as Array<Record<string, string>>).length > 0 && (
              <div>
                <p className="text-xs font-semibold text-text-secondary mb-1">Key Findings</p>
                <ul className="list-disc list-inside space-y-1">
                  {((report.web_search_results as Record<string, unknown>).key_findings as Array<Record<string, string>>).map((f, i) => (
                    <li key={i} className="text-sm text-text-primary">
                      {f.claim}
                      {f.source && <span className="text-xs text-teal-600 ml-1">— {f.source}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {Array.isArray((report.web_search_results as Record<string, unknown>).sources) && ((report.web_search_results as Record<string, unknown>).sources as Array<Record<string, string>>).length > 0 && (
              <div>
                <p className="text-xs font-semibold text-text-secondary mb-1">Sources</p>
                <ul className="space-y-1">
                  {((report.web_search_results as Record<string, unknown>).sources as Array<Record<string, string>>).map((s, i) => (
                    <li key={i} className="text-xs text-teal-700">
                      [{s.domain}] {s.title}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Card>
      )}

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
