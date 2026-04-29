import { DoctorFeedbackRating, FeedbackCategory } from './enums';

export interface DoctorDashboardData {
  total_patients: number;
  pending_reports: number;
}

export interface PatientSummary {
  id: string;
  user_id: string;
  full_name: string;
  last_triage: string | null;
}

export interface PatientDetail {
  id: string;
  user_id: string;
  full_name: string;
  date_of_birth: string | null;
  sex: string | null;
  height_cm: number | null;
  weight_kg: number | null;
}

export interface ReportSummary {
  id: string;
  session_id: string;
  patient_id: string;
  patient_name: string;
  generated_at: string;
  urgency_level: string;
}

export interface FeedbackCreate {
  rating: DoctorFeedbackRating;
  correction_text?: string | null;
  feedback_category?: FeedbackCategory | null;
}
