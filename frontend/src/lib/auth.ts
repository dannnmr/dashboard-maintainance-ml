// lib/auth.ts
import api from "./api";
import {
  User,
  LoginRequest,
  LoginResponse,
  UserCreate,
  UserUpdate,
  ChangePasswordRequest,
  Alert,
  AlertSummary,
} from "../types/auth";

export const authService = {
  // Login
  async login(email: string, password: string): Promise<LoginResponse> {
    const response = await api.post<LoginResponse>("/auth/login", {
      email,
      password,
    });
    return response.data;
  },

  // Get current user
  async getCurrentUser(): Promise<User> {
    const response = await api.get<User>("/auth/me");
    return response.data;
  },

  // Logout
  async logout(): Promise<void> {
    await api.post("/auth/logout");
  },

  // User management (admin only)
  async getUsers(skip = 0, limit = 100): Promise<User[]> {
    const response = await api.get<User[]>(
      `/users?skip=${skip}&limit=${limit}`
    );
    return response.data;
  },

  async getUser(userId: number): Promise<User> {
    const response = await api.get<User>(`/users/${userId}`);
    return response.data;
  },

  async createUser(userData: UserCreate): Promise<User> {
    const response = await api.post<User>("/auth/register", userData);
    return response.data;
  },

  async updateUser(userId: number, userData: UserUpdate): Promise<User> {
    const response = await api.put<User>(`/users/${userId}`, userData);
    return response.data;
  },

  async changePassword(
    userId: number,
    passwordData: ChangePasswordRequest
  ): Promise<void> {
    await api.post(`/users/${userId}/change-password`, passwordData);
  },

  async deleteUser(userId: number): Promise<void> {
    await api.delete(`/users/${userId}`);
  },

  // Alert management
  async getAlerts(skip = 0, limit = 100): Promise<Alert[]> {
    const response = await api.get<Alert[]>(
      `/alerts?skip=${skip}&limit=${limit}`
    );
    return response.data;
  },

  async getAlertSummary(): Promise<AlertSummary> {
    const response = await api.get<AlertSummary>("/alerts/summary");
    return response.data;
  },

  async getActiveAlerts(): Promise<Alert[]> {
    const response = await api.get<Alert[]>("/alerts/active");
    return response.data;
  },

  async getAlert(alertId: number): Promise<Alert> {
    const response = await api.get<Alert>(`/alerts/${alertId}`);
    return response.data;
  },

  async acknowledgeAlert(alertId: number): Promise<Alert> {
    const response = await api.post<Alert>(`/alerts/acknowledge/${alertId}`);
    return response.data;
  },

  async resolveAlert(alertId: number): Promise<Alert> {
    const response = await api.post<Alert>(`/alerts/resolve/${alertId}`);
    return response.data;
  },

  async generateAlertsFromPredictions(
    hoursBack = 24
  ): Promise<{ message: string }> {
    const response = await api.post<{ message: string }>(
      `/alerts/generate-from-predictions?hours_back=${hoursBack}`
    );
    return response.data;
  },

  async updateAlertComments(
    alertId: number,
    comments: string,
    validationStatus?: "validated" | "false_positive" | "investigating"
  ): Promise<Alert> {
    const response = await api.put<Alert>(`/alerts/${alertId}/comments`, {
      comments,
      validation_status: validationStatus,
    });
    return response.data;
  },
};
