// types/auth.ts
export interface User {
  id: number;
  email: string;
  username: string;
  role: "admin" | "viewer" | "tecnico";
  is_active: boolean;
  created_at: string;
  updated_at?: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface UserCreate {
  email: string;
  username: string;
  password: string;
  role: "admin" | "viewer" | "tecnico";
}

export interface UserUpdate {
  email?: string;
  username?: string;
  role?: "admin" | "viewer" | "tecnico";
  is_active?: boolean;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

export interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
}
