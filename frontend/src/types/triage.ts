import {
  TriageSessionStatus,
  UrgencyLevel,
  EscalationType,
  ChatRetentionPolicy,
  MessageSender,
  MessageType,
  AgentPhase,
} from './enums';

export interface TriageSession {
  id: string;
  patient_id: string;
  doctor_id: string;
  hospital_id: string;
  status: TriageSessionStatus;
  chief_complaint: string | null;
  detected_symptoms: Record<string, unknown> | null;
  urgency_level: UrgencyLevel;
  escalation_type: EscalationType;
  chat_retention_policy: ChatRetentionPolicy;
  agent_phase: AgentPhase | null;
  started_at: string;
  ended_at: string | null;
}

/**
 * Patient-safe session response — excludes urgency, symptoms, and diagnosis data.
 * This is what patient-facing endpoints return.
 */
export interface PatientTriageSession {
  id: string;
  status: TriageSessionStatus;
  chief_complaint: string | null;
  agent_phase: AgentPhase | null;
  started_at: string;
  ended_at: string | null;
}

export interface TriageSessionCreate {
  chief_complaint?: string | null;
}

export interface Message {
  id: string;
  session_id: string;
  sender: MessageSender;
  content: string;
  message_type: MessageType;
  is_persisted_after_summary: boolean;
  is_visible_to_doctor: boolean;
  is_deleted: boolean;
  created_at: string;
}

export interface MessageCreate {
  content: string;
}

export interface ClinicalReport {
  id: string;
  session_id: string;
  patient_id: string;
  doctor_id: string;
  presenting_complaints: unknown;
  history_of_presenting_complaint: unknown;
  summary_text: string;
  suspected_conditions: unknown;
  triggered_red_flags: unknown;
  urgency_level: UrgencyLevel;
  recommended_action: string;
  specialty_routing: unknown;
  suggested_workup: unknown;
  key_exam_findings: unknown;
  admission_criteria: unknown;
  referral_criteria: unknown;
  external_escalation_completed: boolean;
  escalation_message: string | null;
  visible_to_patient: boolean;
  model_version: string | null;
  diagnosis_mode: string | null;
  diagnosis_pass_count: number | null;
  chunks_used_count: number | null;
  generated_at: string;
}

/**
 * Session phase response from GET /triage/sessions/{id}/phase
 */
export interface SessionPhase {
  phase: AgentPhase | null;
  is_escalated: boolean;
}

/**
 * Feedback case from the FeedbackStore (doctor-facing).
 */
export interface FeedbackCase {
  case_id: string;
  patient_name: string;
  timestamp: string;
  symptoms: string[];
  urgency: string;
  intake_summary: Record<string, unknown>;
  clinical_picture: Record<string, unknown>;
  diagnosis_report: Record<string, unknown>;
  system_primary_diagnosis: string;
  review_status: 'pending' | 'confirmed' | 'rejected';
  doctor_decision: string | null;
  doctor_diagnosis: string | null;
  doctor_notes: string | null;
}

/**
 * Feedback statistics from GET /doctor/feedback/statistics
 */
export interface FeedbackStats {
  total_cases: number;
  confirmed: number;
  rejected: number;
  pending: number;
}
