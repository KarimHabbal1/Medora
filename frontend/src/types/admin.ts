import { UserRole } from './enums';

export interface AdminUser {
  id: string;
  hospital_id: string;
  full_name: string;
  email: string;
  phone: string | null;
  role: UserRole;
  is_active: boolean;
  registration_method: string;
  created_at: string;
}

export interface AdminUserCreate {
  email: string;
  password: string;
  full_name: string;
  phone?: string | null;
  role: UserRole;
  hospital_id: string;
}

export interface AdminUserUpdate {
  email?: string | null;
  full_name?: string | null;
  phone?: string | null;
  role?: UserRole | null;
  is_active?: boolean | null;
}

export interface Hospital {
  id: string;
  name: string;
}
