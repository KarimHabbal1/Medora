import apiClient from './client';
import type { AdminUser, AdminUserCreate, AdminUserUpdate, Hospital, AssignDoctorRequest } from '../types/admin';

export const adminApi = {
  createUser: async (data: AdminUserCreate): Promise<AdminUser> => {
    const response = await apiClient.post<AdminUser>('/admin/users', data);
    return response.data;
  },

  getUsers: async (): Promise<AdminUser[]> => {
    const response = await apiClient.get<AdminUser[]>('/admin/users');
    return response.data;
  },

  updateUser: async (userId: string, data: AdminUserUpdate): Promise<AdminUser> => {
    const response = await apiClient.patch<AdminUser>(`/admin/users/${userId}`, data);
    return response.data;
  },

  getHospitals: async (): Promise<Hospital[]> => {
    const response = await apiClient.get<Hospital[]>('/admin/hospitals');
    return response.data;
  },

  assignDoctor: async (data: AssignDoctorRequest): Promise<{ message: string }> => {
    const response = await apiClient.post<{ message: string }>('/admin/assign-doctor', data);
    return response.data;
  },
};
