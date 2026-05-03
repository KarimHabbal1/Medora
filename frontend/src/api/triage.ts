import apiClient from './client';
import type {
  PatientTriageSession,
  TriageSessionCreate,
  Message,
  MessageCreate,
  SessionPhase,
} from '../types/triage';

export const triageApi = {
  createSession: async (data: TriageSessionCreate): Promise<PatientTriageSession> => {
    const response = await apiClient.post<PatientTriageSession>('/triage/sessions', data);
    return response.data;
  },

  getSessions: async (): Promise<PatientTriageSession[]> => {
    const response = await apiClient.get<PatientTriageSession[]>('/triage/sessions');
    return response.data;
  },

  getSession: async (sessionId: string): Promise<PatientTriageSession> => {
    const response = await apiClient.get<PatientTriageSession>(`/triage/sessions/${sessionId}`);
    return response.data;
  },

  getSessionPhase: async (sessionId: string): Promise<SessionPhase> => {
    const response = await apiClient.get<SessionPhase>(`/triage/sessions/${sessionId}/phase`);
    return response.data;
  },

  sendMessage: async (sessionId: string, data: MessageCreate): Promise<Message> => {
    const response = await apiClient.post<Message>(
      `/triage/sessions/${sessionId}/message`,
      data
    );
    return response.data;
  },

  getMessages: async (sessionId: string): Promise<Message[]> => {
    const response = await apiClient.get<Message[]>(
      `/triage/sessions/${sessionId}/messages`
    );
    return response.data;
  },

  endSession: async (sessionId: string): Promise<{ message: string; report_id: string }> => {
    const response = await apiClient.post<{ message: string; report_id: string }>(
      `/triage/sessions/${sessionId}/end`
    );
    return response.data;
  },
};
