import apiClient from './client';
import type { AdminUser, AdminUserCreate, AdminUserUpdate, Hospital } from '../types/admin';

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
};
