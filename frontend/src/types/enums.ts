// Using const objects instead of enums for erasableSyntaxOnly compatibility

export const UserRole = {
  Patient: 'patient',
  Doctor: 'doctor',
  Admin: 'admin',
} as const;
export type UserRole = (typeof UserRole)[keyof typeof UserRole];

export const RegistrationMethod = {
  AdminCreated: 'admin_created',
  SelfSignup: 'self_signup',
} as const;
export type RegistrationMethod = (typeof RegistrationMethod)[keyof typeof RegistrationMethod];

export const TriageSessionStatus = {
  Active: 'active',
  Completed: 'completed',
  Cancelled: 'cancelled',
} as const;
export type TriageSessionStatus = (typeof TriageSessionStatus)[keyof typeof TriageSessionStatus];

export const UrgencyLevel = {
  Routine: 'routine',
  Urgent: 'urgent',
  Emergency: 'emergency',
  Unknown: 'unknown',
} as const;
export type UrgencyLevel = (typeof UrgencyLevel)[keyof typeof UrgencyLevel];

export const EscalationType = {
  None: 'none',
  EmergencyCall: 'emergency_call',
  ComplexDiagnosisAgent: 'complex_diagnosis_agent',
} as const;
export type EscalationType = (typeof EscalationType)[keyof typeof EscalationType];

export const ChatRetentionPolicy = {
  SummaryOnly: 'summary_only',
  KeepFullHistory: 'keep_full_history',
} as const;
export type ChatRetentionPolicy = (typeof ChatRetentionPolicy)[keyof typeof ChatRetentionPolicy];

export const MessageSender = {
  Patient: 'patient',
  IntakeAgent: 'intake_agent',
  RagAgent: 'rag_agent',
  System: 'system',
} as const;
export type MessageSender = (typeof MessageSender)[keyof typeof MessageSender];

export const MessageType = {
  Text: 'text',
  Question: 'question',
  Answer: 'answer',
  Warning: 'warning',
  Summary: 'summary',
  StreamDelta: 'stream_delta',
} as const;
export type MessageType = (typeof MessageType)[keyof typeof MessageType];

export const ConsentType = {
  MedicalDisclaimer: 'medical_disclaimer',
  DataStorage: 'data_storage',
  AiAssistance: 'ai_assistance',
  ChatHistoryStorage: 'chat_history_storage',
} as const;
export type ConsentType = (typeof ConsentType)[keyof typeof ConsentType];

export const DoctorFeedbackRating = {
  ThumbsUp: 'thumbs_up',
  ThumbsDown: 'thumbs_down',
} as const;
export type DoctorFeedbackRating = (typeof DoctorFeedbackRating)[keyof typeof DoctorFeedbackRating];

export const FeedbackCategory = {
  WrongUrgency: 'wrong_urgency',
  WrongDiagnosis: 'wrong_diagnosis',
  MissingInfo: 'missing_info',
  UnsafeResponse: 'unsafe_response',
  IrrelevantSources: 'irrelevant_sources',
  Other: 'other',
} as const;
export type FeedbackCategory = (typeof FeedbackCategory)[keyof typeof FeedbackCategory];
