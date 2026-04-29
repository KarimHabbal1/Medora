import React, { createContext, useState, useEffect, useCallback } from 'react';
import { authApi } from '../api/auth';
import type { User, UserCreate } from '../types/auth';
import { UserRole } from '../types/enums';

interface AuthContextType {
  user: User | null;
  token: string | null;
  loading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (data: UserCreate) => Promise<void>;
  logout: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  loading: true,
  isAuthenticated: false,
  login: async () => {},
  signup: async () => {},
  logout: async () => {},
});

function getRoleRedirectPath(role: UserRole): string {
  switch (role) {
    case UserRole.Patient:
      return '/patient/dashboard';
    case UserRole.Doctor:
      return '/doctor/dashboard';
    case UserRole.Admin:
      return '/admin/users';
    default:
      return '/login';
  }
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(
    localStorage.getItem('medora_token')
  );
  const [loading, setLoading] = useState(true);

  const fetchUser = useCallback(async () => {
    if (!token) {
      setLoading(false);
      return;
    }
    try {
      const userData = await authApi.getMe();
      setUser(userData);
    } catch {
      localStorage.removeItem('medora_token');
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  const login = async (email: string, password: string) => {
    const tokenData = await authApi.signin(email, password);
    localStorage.setItem('medora_token', tokenData.access_token);
    setToken(tokenData.access_token);

    const userData = await authApi.getMe();
    setUser(userData);

    window.location.href = getRoleRedirectPath(userData.role);
  };

  const signup = async (data: UserCreate) => {
    await authApi.signup(data);
    // Auto-login after signup
    await login(data.email, data.password);
  };

  const logout = async () => {
    try {
      await authApi.logout();
    } catch {
      // Ignore logout API errors
    }
    localStorage.removeItem('medora_token');
    setToken(null);
    setUser(null);
    window.location.href = '/login';
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        isAuthenticated: !!user && !!token,
        login,
        signup,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
