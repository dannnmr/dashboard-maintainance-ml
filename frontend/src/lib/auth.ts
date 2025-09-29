// lib/auth.ts
import api from "./api";
import {
  User,
  LoginRequest,
  LoginResponse,
  UserCreate,
  UserUpdate,
  ChangePasswordRequest,
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
};
