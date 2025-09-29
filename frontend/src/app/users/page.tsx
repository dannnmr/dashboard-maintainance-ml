// app/users/page.tsx
"use client";

import React from "react";
import ProtectedRoute from "../../components/ProtectedRoute";
import Layout from "../../components/Layout";
import UserManagement from "../../components/UserManagement";

export default function UsersPage() {
  return (
    <ProtectedRoute requiredRole="admin">
      <Layout>
        <UserManagement />
      </Layout>
    </ProtectedRoute>
  );
}
