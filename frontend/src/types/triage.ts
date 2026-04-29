import {
  TriageSessionStatus,
  UrgencyLevel,
  EscalationType,
  ChatRetentionPolicy,
  MessageSender,
  MessageType,
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
  generated_at: string;
}
