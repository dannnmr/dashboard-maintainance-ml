// contexts/AuthContext.tsx
"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import Cookies from "js-cookie";
import { User, AuthContextType } from "../types/auth";
import { authService } from "../lib/auth";
import toast from "react-hot-toast";

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize auth state from cookies
  useEffect(() => {
    const initAuth = () => {
      try {
        const savedToken = Cookies.get("token");
        const savedUser = Cookies.get("user");

        console.log("Init auth - Token:", savedToken ? "exists" : "none");
        console.log("Init auth - User:", savedUser ? "exists" : "none");

        if (savedToken && savedUser) {
          setToken(savedToken);
          setUser(JSON.parse(savedUser));
          console.log("Auth initialized from cookies");
        } else {
          console.log("No auth data in cookies");
        }
      } catch (error) {
        console.error("Error initializing auth:", error);
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      console.log("Starting login for:", email);
      const response = await authService.login(email, password);
      console.log("Login response received:", response);

      // Save token to cookies immediately
      Cookies.set("token", response.access_token, { expires: 7 });
      console.log("Token saved to cookies");

      const userData = await authService.getCurrentUser();
      console.log("User data received:", userData);

      setToken(response.access_token);
      setUser(userData);

      // Save user to cookies
      Cookies.set("user", JSON.stringify(userData), { expires: 7 });

      console.log("Auth state updated and saved to cookies");

      toast.success("Login successful!");
    } catch (error: any) {
      console.error("Login error:", error);
      const message = error.response?.data?.detail || "Login failed";
      toast.error(message);
      throw error;
    }
  };

  const logout = async () => {
    try {
      await authService.logout();
    } catch (error) {
      // Even if logout fails on server, clear local state
      console.error("Logout error:", error);
    } finally {
      setToken(null);
      setUser(null);
      Cookies.remove("token");
      Cookies.remove("user");
      toast.success("Logged out successfully");
    }
  };

  const value: AuthContextType = {
    user,
    token,
    login,
    logout,
    isLoading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
