import { ConsentType } from './enums';

export interface PatientProfile {
  id: string;
  user_id: string;
  hospital_id: string;
  assigned_doctor_id: string | null;
  date_of_birth: string | null;
  sex: string | null;
  height_cm: number | null;
  weight_kg: number | null;
  created_at: string;
}

export interface PatientUpdate {
  date_of_birth?: string | null;
  sex?: string | null;
  height_cm?: number | null;
  weight_kg?: number | null;
}

export interface MedicalHistory {
  id: string;
  patient_id: string;
  chronic_conditions: Record<string, unknown> | null;
  medications: Record<string, unknown> | null;
  allergies: Record<string, unknown> | null;
  surgeries: Record<string, unknown> | null;
  family_history: Record<string, unknown> | null;
  smoking_status: string | null;
  pregnancy_status: string | null;
  additional_notes: string | null;
  skipped: boolean;
  updated_at: string;
}

export interface MedicalHistoryUpdate {
  chronic_conditions?: Record<string, unknown> | null;
  medications?: Record<string, unknown> | null;
  allergies?: Record<string, unknown> | null;
  surgeries?: Record<string, unknown> | null;
  family_history?: Record<string, unknown> | null;
  smoking_status?: string | null;
  pregnancy_status?: string | null;
  additional_notes?: string | null;
  skipped?: boolean | null;
}

export interface Consent {
  id: string;
  patient_id: string;
  consent_type: ConsentType;
  accepted: boolean;
  accepted_at: string;
}

export interface ConsentCreate {
  consent_type: ConsentType;
  accepted: boolean;
}
