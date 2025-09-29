// app/dashboard/page.tsx
"use client";

import React from "react";
import ProtectedRoute from "../../components/ProtectedRoute";
import Layout from "../../components/Layout";
import DashboardContent from "../../components/DashboardContent";

export default function DashboardPage() {
  return (
    <ProtectedRoute>
      <Layout>
        <DashboardContent />
      </Layout>
    </ProtectedRoute>
  );
}
