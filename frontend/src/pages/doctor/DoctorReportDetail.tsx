import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import Markdown from 'react-markdown';
import { doctorApi } from '../../api/doctor';
import Card from '../../components/ui/Card';
import LoadingSpinner from '../../components/ui/LoadingSpinner';
import ErrorAlert from '../../components/ui/ErrorAlert';
import UrgencyBadge from '../../components/doctor/UrgencyBadge';
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

  const ws = report.web_search_results as Record<string, unknown> | null;

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
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

      {/* Escalation banner */}
      {report.escalation_message && (
        <Card className="border-red-200 bg-red-50/50">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg">⚠</span>
            <h3 className="text-sm font-semibold text-red-800">Escalation</h3>
          </div>
          <p className="text-sm text-red-700">{report.escalation_message}</p>
        </Card>
      )}

      {/* Side-by-side: Textbook Diagnosis + Web Search Diagnosis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Textbook Diagnosis */}
        <Card className="bg-medora-50/50 border-medora-200">
          <h3 className="text-xs font-semibold text-medora-600 uppercase tracking-wider mb-2">Textbook Diagnosis (RAG)</h3>
          <div className="text-sm text-text-primary leading-relaxed prose prose-sm max-w-none">
            <Markdown>{report.summary_text}</Markdown>
          </div>
        </Card>

        {/* Web Search Diagnosis */}
        <Card className={ws ? 'bg-teal-50/50 border-teal-200' : 'bg-gray-50/50 border-gray-200'}>
          <h3 className="text-xs font-semibold text-teal-600 uppercase tracking-wider mb-2">Web Search Diagnosis</h3>
          {ws ? (
            <div className="space-y-3">
              <div>
                <p className="text-lg font-semibold text-text-primary">
                  {ws.primary_diagnosis as string || 'N/A'}
                </p>
                {ws.confidence ? (
                  <span className="text-xs font-medium text-teal-600">({String(ws.confidence)} confidence)</span>
                ) : null}
              </div>
              {ws.evidence_summary ? (
                <p className="text-sm text-text-primary leading-relaxed">{String(ws.evidence_summary)}</p>
              ) : null}
              {Array.isArray(ws.key_findings) && (ws.key_findings as Array<Record<string, string>>).length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-text-secondary mb-1">Key Findings</p>
                  <ul className="list-disc list-inside space-y-1">
                    {(ws.key_findings as Array<Record<string, string>>).map((f, i) => (
                      <li key={i} className="text-sm text-text-primary">
                        {f.claim}
                        {f.source && <span className="text-xs text-teal-600 ml-1">— {f.source}</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {Array.isArray(ws.differential_diagnoses) && (ws.differential_diagnoses as string[]).length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-text-secondary mb-1">Differentials</p>
                  <p className="text-sm text-text-primary">{(ws.differential_diagnoses as string[]).join(', ')}</p>
                </div>
              )}
              {Array.isArray(ws.sources) && (ws.sources as Array<Record<string, string>>).length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-text-secondary mb-1">Sources</p>
                  <ul className="space-y-0.5">
                    {(ws.sources as Array<Record<string, string>>).map((s, i) => (
                      <li key={i} className="text-xs text-teal-700">[{s.domain}] {s.title}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-text-tertiary italic">Web search not available for this report.</p>
          )}
        </Card>
      </div>

      {/* Clinical Reasoning */}
      {report.history_of_presenting_complaint && (
        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-2">Clinical Reasoning</h3>
          <div className="text-sm text-text-primary leading-relaxed prose prose-sm max-w-none">
            <Markdown>{typeof report.history_of_presenting_complaint === 'string'
              ? report.history_of_presenting_complaint
              : typeof report.history_of_presenting_complaint === 'object' && report.history_of_presenting_complaint !== null
                ? (report.history_of_presenting_complaint as Record<string, unknown>).reasoning as string || ''
                : ''}</Markdown>
          </div>
        </Card>
      )}

      {/* Red Flags */}
      {report.triggered_red_flags && typeof report.triggered_red_flags === 'object' && (
        (() => {
          const flags = report.triggered_red_flags as Record<string, unknown>;
          const flagList = (flags.flags || []) as Array<Record<string, string>>;
          if (flagList.length === 0) return null;
          return (
            <Card className="border-amber-200 bg-amber-50/50">
              <h3 className="text-sm font-semibold text-amber-800 mb-2">Red Flags</h3>
              <ul className="list-disc list-inside space-y-1">
                {flagList.map((f, i) => (
                  <li key={i} className="text-sm text-amber-900">
                    {typeof f === 'string' ? f : f.flag || JSON.stringify(f)}
                    {f.urgency && <span className="text-xs text-amber-600 ml-1">({f.urgency})</span>}
                  </li>
                ))}
              </ul>
            </Card>
          );
        })()
      )}

      {/* Recommended Actions — merged recommended_action + suggested_workup */}
      {(report.recommended_action || report.suggested_workup) && (
        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-2">Recommended Actions</h3>
          <div className="text-sm text-text-primary leading-relaxed prose prose-sm max-w-none">
            {report.recommended_action && (
              <Markdown>{typeof report.recommended_action === 'string'
                ? report.recommended_action
                : ''}</Markdown>
            )}
            {report.suggested_workup && (
              <>
                <p className="text-xs font-semibold text-text-secondary mt-3 mb-1">Suggested Workup</p>
                <Markdown>{typeof report.suggested_workup === 'string'
                  ? report.suggested_workup
                  : typeof report.suggested_workup === 'object' && report.suggested_workup !== null
                    ? (report.suggested_workup as Record<string, unknown>).workup as string || ''
                    : ''}</Markdown>
              </>
            )}
          </div>
        </Card>
      )}

      {/* Specialty Routing — only if present */}
      {report.specialty_routing && typeof report.specialty_routing === 'object' && (report.specialty_routing as Record<string, unknown>).routing && (
        <Card>
          <h3 className="text-sm font-semibold text-text-secondary mb-2">Specialty Routing</h3>
          <p className="text-sm text-text-primary">{(report.specialty_routing as Record<string, unknown>).routing as string}</p>
        </Card>
      )}

      {/* Feedback */}
      <Card>
        <FeedbackForm reportId={report.id} />
      </Card>
    </div>
  );
};

export default DoctorReportDetail;
