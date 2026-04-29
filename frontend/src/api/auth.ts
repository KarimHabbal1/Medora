import apiClient from './client';
import type { User, Token, UserCreate, ChangePassword } from '../types/auth';

export const authApi = {
  signup: async (data: UserCreate): Promise<User> => {
    const response = await apiClient.post<User>('/auth/signup', data);
    return response.data;
  },

  signin: async (email: string, password: string): Promise<Token> => {
    // Backend uses OAuth2PasswordRequestForm which expects form-data
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

    const response = await apiClient.post<Token>('/auth/signin', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    return response.data;
  },

  getMe: async (): Promise<User> => {
    const response = await apiClient.get<User>('/auth/me');
    return response.data;
  },

  logout: async (): Promise<void> => {
    await apiClient.post('/auth/logout');
  },

  changePassword: async (data: ChangePassword): Promise<void> => {
    await apiClient.patch('/auth/change-password', data);
  },
};
