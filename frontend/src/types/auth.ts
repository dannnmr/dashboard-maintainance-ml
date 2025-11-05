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

// Alert types
export interface Alert {
  id: number;
  equipment_id: string;
  equipment_name?: string;
  title: string;
  message: string;
  severity: "critical" | "warning" | "info";
  status: "active" | "acknowledged" | "resolved";
  alert_type: string;
  source: string;
  comments?: string;
  validation_status?: "validated" | "false_positive" | "investigating";
  prediction_id?: number;
  anomaly_score?: number;
  confidence_score?: number;
  acknowledged_by_user_id?: number;
  resolved_by_user_id?: number;
  created_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  updated_at?: string;
}

export interface AlertSummary {
  total_alerts: number;
  critical_alerts: number;
  warning_alerts: number;
  info_alerts: number;
  active_alerts: number;
  acknowledged_alerts: number;
  resolved_alerts: number;
}

export interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
}
