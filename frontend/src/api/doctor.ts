import apiClient from './client';
import type {
  DoctorDashboardData,
  PatientSummary,
  PatientDetail,
  ReportSummary,
  FeedbackCreate,
} from '../types/doctor';
import type { ClinicalReport } from '../types/triage';

export const doctorApi = {
  getDashboard: async (): Promise<DoctorDashboardData> => {
    const response = await apiClient.get<DoctorDashboardData>('/doctor/dashboard');
    return response.data;
  },

  getPatients: async (): Promise<PatientSummary[]> => {
    const response = await apiClient.get<PatientSummary[]>('/doctor/patients');
    return response.data;
  },

  getPatient: async (patientId: string): Promise<PatientDetail> => {
    const response = await apiClient.get<PatientDetail>(`/doctor/patients/${patientId}`);
    return response.data;
  },

  getPatientReports: async (patientId: string): Promise<ReportSummary[]> => {
    const response = await apiClient.get<ReportSummary[]>(
      `/doctor/patients/${patientId}/reports`
    );
    return response.data;
  },

  getReports: async (): Promise<ReportSummary[]> => {
    const response = await apiClient.get<ReportSummary[]>('/doctor/reports');
    return response.data;
  },

  getReport: async (reportId: string): Promise<ClinicalReport> => {
    const response = await apiClient.get<ClinicalReport>(`/doctor/reports/${reportId}`);
    return response.data;
  },

  submitFeedback: async (
    reportId: string,
    data: FeedbackCreate
  ): Promise<{ message: string }> => {
    const response = await apiClient.post<{ message: string }>(
      `/doctor/reports/${reportId}/feedback`,
      data
    );
    return response.data;
  },
};
