import React, { useState } from 'react';
import { doctorApi } from '../../api/doctor';
import Button from '../ui/Button';
import TextArea from '../ui/TextArea';
import ErrorAlert from '../ui/ErrorAlert';
import { DoctorFeedbackRating, FeedbackCategory } from '../../types/enums';

interface FeedbackFormProps {
  reportId: string;
  onSubmitted?: () => void;
}

const FeedbackForm: React.FC<FeedbackFormProps> = ({ reportId, onSubmitted }) => {
  const [rating, setRating] = useState<DoctorFeedbackRating | null>(null);
  const [category, setCategory] = useState<FeedbackCategory | ''>('');
  const [correction, setCorrection] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async () => {
    if (!rating) return;
    setSubmitting(true);
    setError('');
    try {
      await doctorApi.submitFeedback(reportId, {
        rating,
        correction_text: correction || undefined,
        feedback_category: (category as FeedbackCategory) || undefined,
      });
      setSubmitted(true);
      onSubmitted?.();
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Failed to submit feedback.';
      setError(msg);
    } finally { setSubmitting(false); }
  };

  if (submitted) {
    return (
      <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-700 text-center">
        ✓ Feedback submitted successfully
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-base font-semibold text-text-primary">Doctor Feedback</h3>
      {error && <ErrorAlert message={error} onDismiss={() => setError('')} />}

      {/* Rating */}
      <div>
        <p className="text-sm font-medium text-text-primary mb-2">Rating</p>
        <div className="flex gap-3">
          <button
            onClick={() => setRating(DoctorFeedbackRating.ThumbsUp)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${rating === DoctorFeedbackRating.ThumbsUp ? 'border-emerald-300 bg-emerald-50 text-emerald-700' : 'border-border text-text-secondary hover:bg-surface-tertiary'}`}
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M6.633 10.5c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 012.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 00.322-1.672V3a.75.75 0 01.75-.75A2.25 2.25 0 0116.5 4.5c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 01-2.649 7.521c-.388.482-.987.729-1.605.729H14.23c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 00-1.423-.23H5.904M14.25 9h2.25M5.904 18.75c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 01-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 10.203 4.167 9.75 5 9.75h1.053c.472 0 .745.556.5.96a8.958 8.958 0 00-1.302 4.665c0 1.194.232 2.333.654 3.375z" /></svg>
            Thumbs Up
          </button>
          <button
            onClick={() => setRating(DoctorFeedbackRating.ThumbsDown)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${rating === DoctorFeedbackRating.ThumbsDown ? 'border-red-300 bg-red-50 text-red-700' : 'border-border text-text-secondary hover:bg-surface-tertiary'}`}
          >
            <svg className="h-5 w-5 rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M6.633 10.5c.806 0 1.533-.446 2.031-1.08a9.041 9.041 0 012.861-2.4c.723-.384 1.35-.956 1.653-1.715a4.498 4.498 0 00.322-1.672V3a.75.75 0 01.75-.75A2.25 2.25 0 0116.5 4.5c0 1.152-.26 2.243-.723 3.218-.266.558.107 1.282.725 1.282h3.126c1.026 0 1.945.694 2.054 1.715.045.422.068.85.068 1.285a11.95 11.95 0 01-2.649 7.521c-.388.482-.987.729-1.605.729H14.23c-.483 0-.964-.078-1.423-.23l-3.114-1.04a4.501 4.501 0 00-1.423-.23H5.904M14.25 9h2.25M5.904 18.75c.083.205.173.405.27.602.197.4-.078.898-.523.898h-.908c-.889 0-1.713-.518-1.972-1.368a12 12 0 01-.521-3.507c0-1.553.295-3.036.831-4.398C3.387 10.203 4.167 9.75 5 9.75h1.053c.472 0 .745.556.5.96a8.958 8.958 0 00-1.302 4.665c0 1.194.232 2.333.654 3.375z" /></svg>
            Thumbs Down
          </button>
        </div>
      </div>

      {/* Category */}
      <div>
        <label className="block text-sm font-medium text-text-primary mb-1.5">Category (optional)</label>
        <select
          className="w-full rounded-lg border border-border bg-white px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-medora-500"
          value={category}
          onChange={(e) => setCategory(e.target.value as FeedbackCategory | '')}
        >
          <option value="">Select category...</option>
          <option value="wrong_urgency">Wrong Urgency</option>
          <option value="wrong_diagnosis">Wrong Diagnosis</option>
          <option value="missing_info">Missing Info</option>
          <option value="unsafe_response">Unsafe Response</option>
          <option value="irrelevant_sources">Irrelevant Sources</option>
          <option value="other">Other</option>
        </select>
      </div>

      {/* Correction text */}
      <TextArea label="Correction / Notes (optional)" placeholder="Describe what should be corrected..." value={correction} onChange={(e) => setCorrection(e.target.value)} />

      <Button onClick={handleSubmit} loading={submitting} disabled={!rating}>Submit Feedback</Button>
    </div>
  );
};

export default FeedbackForm;
