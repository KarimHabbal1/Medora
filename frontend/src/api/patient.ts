import apiClient from './client';
import type {
  PatientProfile,
  PatientUpdate,
  MedicalHistory,
  MedicalHistoryUpdate,
  Consent,
  ConsentCreate,
} from '../types/patient';

export const patientApi = {
  getProfile: async (): Promise<PatientProfile> => {
    const response = await apiClient.get<PatientProfile>('/patients/me');
    return response.data;
  },

  updateProfile: async (data: PatientUpdate): Promise<PatientProfile> => {
    const response = await apiClient.patch<PatientProfile>('/patients/me', data);
    return response.data;
  },

  getMedicalHistory: async (): Promise<MedicalHistory> => {
    const response = await apiClient.get<MedicalHistory>('/patients/me/medical-history');
    return response.data;
  },

  updateMedicalHistory: async (data: MedicalHistoryUpdate): Promise<MedicalHistory> => {
    const response = await apiClient.put<MedicalHistory>('/patients/me/medical-history', data);
    return response.data;
  },

  getConsents: async (): Promise<Consent[]> => {
    const response = await apiClient.get<Consent[]>('/patients/me/consents');
    return response.data;
  },

  grantConsent: async (data: ConsentCreate): Promise<Consent> => {
    const response = await apiClient.post<Consent>('/patients/me/consents', data);
    return response.data;
  },
};
