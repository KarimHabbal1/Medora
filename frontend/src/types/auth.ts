import { UserRole, RegistrationMethod } from './enums';

export interface User {
  id: string;
  hospital_id: string;
  full_name: string;
  email: string;
  phone: string | null;
  role: UserRole;
  is_active: boolean;
  registration_method: RegistrationMethod;
  created_at: string;
}

export interface Token {
  access_token: string;
  token_type: string;
  refresh_token?: string | null;
}

export interface UserCreate {
  email: string;
  password: string;
  full_name: string;
  phone?: string | null;
}

export interface UserLogin {
  email: string;
  password: string;
}

export interface ChangePassword {
  old_password: string;
  new_password: string;
}
