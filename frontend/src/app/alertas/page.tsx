// app/alertas/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import ProtectedRoute from "../../components/ProtectedRoute";
import Layout from "../../components/Layout";
import { authService } from "../../lib/auth";
import { Alert, AlertSummary, User } from "../../types/auth";
import toast from "react-hot-toast";

export default function AlertasPage() {
  const [summary, setSummary] = useState<AlertSummary | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [editingAlert, setEditingAlert] = useState<Alert | null>(null);
  const [comments, setComments] = useState("");
  const [validationStatus, setValidationStatus] = useState("");

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setIsLoading(true);
      const [summaryData, alertsData, me] = await Promise.all([
        authService.getAlertSummary(),
        authService.getActiveAlerts(),
        authService.getCurrentUser(),
      ]);
      setCurrentUser(me);
      // Debug: ver datos crudos
      if (process.env.NODE_ENV !== "production") {
        // eslint-disable-next-line no-console
        console.log("/alerts/summary =>", summaryData);
        // eslint-disable-next-line no-console
        console.log("/alerts/active =>", alertsData);
      }

      setSummary(summaryData);
      const normalized = (alertsData ?? []).map((a: any) => ({
        ...a,
        status:
          typeof a.status === "string"
            ? a.status.toLowerCase()
            : a.status?.value?.toLowerCase?.() || a.status,
        severity:
          typeof a.severity === "string"
            ? a.severity.toLowerCase()
            : a.severity?.value?.toLowerCase?.() || a.severity,
      }));
      if (process.env.NODE_ENV !== "production") {
        // eslint-disable-next-line no-console
        console.log("normalized alerts =>", normalized);
      }
      setAlerts(normalized);
    } catch (error: any) {
      const message = error?.response?.data?.detail || "Error cargando alertas";
      toast.error(message);
      console.error("Error loading alerts:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAcknowledge = async (alertId: number) => {
    try {
      await authService.acknowledgeAlert(alertId);
      toast.success("Alerta confirmada");
      loadData();
    } catch (error: any) {
      toast.error("Error confirmando alerta");
    }
  };

  const handleResolve = async (alertId: number) => {
    try {
      await authService.resolveAlert(alertId);
      toast.success("Alerta rechazada");
      loadData();
    } catch (error: any) {
      toast.error("Error rechazando alerta");
    }
  };

  const handleUpdateComments = async (
    alertId: number,
    comments: string,
    validationStatus?: string
  ) => {
    try {
      const trimmed = (comments || "").trim();
      const payloadComments = trimmed.length > 0 ? trimmed : "";
      if (!payloadComments && !validationStatus) {
        toast("Nada que actualizar", { icon: "ℹ️" });
        return;
      }
      await authService.updateAlertComments(
        alertId,
        payloadComments,
        validationStatus as any
      );
      toast.success("Comentarios actualizados");
      loadData();
    } catch (error: any) {
      const d = error?.response?.data?.detail;
      const message =
        (typeof d === "string" && d) ||
        (Array.isArray(d) && (d[0]?.msg || d[0]?.detail)) ||
        error?.response?.data?.message ||
        error?.message ||
        "Error actualizando comentarios";
      // eslint-disable-next-line no-console
      console.error("update comments error:", error?.response?.data || error);
      toast.error(message);
    }
  };

  const handleEditComments = (alert: Alert) => {
    setEditingAlert(alert);
    setComments(alert.comments || "");
    setValidationStatus(alert.validation_status || "");
  };

  const handleSaveComments = async () => {
    if (!editingAlert) return;

    await handleUpdateComments(editingAlert.id, comments, validationStatus);
    setEditingAlert(null);
    setComments("");
    setValidationStatus("");
  };

  const handleCancelEdit = () => {
    setEditingAlert(null);
    setComments("");
    setValidationStatus("");
  };

  const handleGenerateAlerts = async () => {
    try {
      setIsGenerating(true);
      const result = await authService.generateAlertsFromPredictions(24);
      toast.success(result.message);
      loadData();
    } catch (error: any) {
      toast.error("Error generando alertas");
    } finally {
      setIsGenerating(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "text-red-600 bg-red-50 border-red-200";
      case "warning":
        return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "info":
        return "text-blue-600 bg-blue-50 border-blue-200";
      default:
        return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "critical":
        return (
          <path
            fillRule="evenodd"
            d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        );
      case "warning":
        return (
          <path
            fillRule="evenodd"
            d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
            clipRule="evenodd"
          />
        );
      case "info":
        return (
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            clipRule="evenodd"
          />
        );
      default:
        return null;
    }
  };

  if (isLoading) {
    return (
      <ProtectedRoute>
        <Layout>
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600"></div>
          </div>
        </Layout>
      </ProtectedRoute>
    );
  }

  return (
    <ProtectedRoute>
      <Layout>
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Alertas</h1>
              <p className="mt-1 text-sm text-gray-500">
                Sistema de alertas y notificaciones
              </p>
              <p className="mt-1 text-xs text-green-600">
                Las alertas se generan automáticamente cuando se detectan
                anomalías
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Alertas Críticas */}
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                    <svg
                      className="w-5 h-5 text-red-600"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    Críticas
                  </h3>
                  <p className="text-3xl font-bold text-red-600">
                    {summary?.critical_alerts || 0}
                  </p>
                </div>
              </div>
            </div>

            {/* Alertas de Advertencia */}
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
                    <svg
                      className="w-5 h-5 text-yellow-600"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    Advertencia
                  </h3>
                  <p className="text-3xl font-bold text-yellow-600">
                    {summary?.warning_alerts || 0}
                  </p>
                </div>
              </div>
            </div>

            {/* Alertas Informativas */}
            <div className="bg-white shadow rounded-lg p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                    <svg
                      className="w-5 h-5 text-blue-600"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    Informativas
                  </h3>
                  <p className="text-3xl font-bold text-blue-600">
                    {summary?.info_alerts || 0}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Lista de Alertas */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-900">
                Alertas Activas ({summary?.active_alerts || 0})
              </h2>
            </div>
            <div className="p-6">
              {alerts.length === 0 ? (
                <div className="text-center py-12">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <h3 className="mt-2 text-sm font-medium text-gray-900">
                    No hay alertas activas
                  </h3>
                  <p className="mt-1 text-sm text-gray-500">
                    El sistema está funcionando normalmente.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {alerts.map((alert) => (
                    <div
                      key={alert.id}
                      className="border border-gray-200 rounded-lg p-4"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3">
                            <span
                              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getSeverityColor(
                                alert.severity
                              )}`}
                            >
                              <svg
                                className="w-3 h-3 mr-1"
                                fill="currentColor"
                                viewBox="0 0 20 20"
                              >
                                {getSeverityIcon(alert.severity)}
                              </svg>
                              {alert.severity.toUpperCase()}
                            </span>
                            <span className="text-sm text-gray-500">
                              {new Date(alert.created_at).toLocaleString()}
                            </span>
                          </div>
                          <h3 className="mt-2 text-lg font-medium text-gray-900">
                            {alert.title}
                          </h3>
                          <p className="mt-1 text-sm text-gray-600">
                            {alert.message}
                          </p>
                          {alert.anomaly_score && (
                            <div className="mt-2 text-xs text-gray-500">
                              Score: {alert.anomaly_score.toFixed(3)} |
                              Confianza:{" "}
                              {alert.confidence_score?.toFixed(3) || "N/A"}
                            </div>
                          )}

                          {/* Mostrar comentarios y estado de validación */}
                          {alert.comments && (
                            <div className="mt-3 p-3 bg-gray-50 rounded-md">
                              <div className="text-xs text-gray-600 mb-1">
                                <strong>Comentarios:</strong>
                              </div>
                              <div className="text-sm text-gray-800">
                                {alert.comments}
                              </div>
                            </div>
                          )}

                          {alert.validation_status && (
                            <div className="mt-2">
                              <span
                                className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                                  alert.validation_status === "validated"
                                    ? "bg-green-100 text-green-800"
                                    : alert.validation_status ===
                                      "false_positive"
                                    ? "bg-red-100 text-red-800"
                                    : "bg-yellow-100 text-yellow-800"
                                }`}
                              >
                                {alert.validation_status === "validated"
                                  ? "Validado"
                                  : alert.validation_status === "false_positive"
                                  ? " Falso Positivo"
                                  : " En Observación"}
                              </span>
                            </div>
                          )}
                        </div>
                        <div className="flex space-x-2 ml-4">
                          {alert.status === "active" && (
                            <>
                              <button
                                onClick={() => handleAcknowledge(alert.id)}
                                className="inline-flex items-center px-3 py-1 border border-transparent text-xs font-medium rounded text-white bg-green-600 hover:bg-green-700"
                              >
                                Confirmar
                              </button>
                              <button
                                onClick={() => handleResolve(alert.id)}
                                className="inline-flex items-center px-3 py-1 border border-gray-300 shadow-sm text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50"
                              >
                                Rechazar
                              </button>
                              <button
                                onClick={() => handleEditComments(alert)}
                                className="inline-flex items-center px-3 py-1 border border-gray-300 text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50"
                              >
                                Comentarios
                              </button>
                            </>
                          )}
                          {alert.status === "acknowledged" && (
                            <button
                              onClick={() => handleResolve(alert.id)}
                              className="inline-flex items-center px-3 py-1 border border-gray-300 shadow-sm text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50"
                            >
                              Rechazar
                            </button>
                          )}
                          {alert.status === "resolved" && (
                            <span className="inline-flex items-center px-3 py-1 border border-transparent text-xs font-medium rounded text-green-800 bg-green-100">
                              Rechazada
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </Layout>

      {/* Modal de Comentarios */}
      {editingAlert && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center p-4 sm:items-center bg-black/50 backdrop-blur-sm"
          onClick={handleCancelEdit}
          aria-modal="true"
          role="dialog"
          aria-labelledby="alert-comments-title"
        >
          <div
            className="relative w-full max-w-md rounded-md bg-white shadow-lg border"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-5">
              <h3
                id="alert-comments-title"
                className="text-lg font-medium text-gray-900 mb-4"
              >
                Comentarios y Validación - {editingAlert.title}
              </h3>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Estado de Validación
                </label>
                <select
                  value={validationStatus}
                  onChange={(e) => setValidationStatus(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Seleccionar estado</option>
                  <option value="validated">Validado</option>
                  <option value="false_positive">Falso Positivo</option>
                  <option value="investigating">En Observación</option>
                </select>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Comentarios y Observaciones
                </label>
                <textarea
                  value={comments}
                  onChange={(e) => setComments(e.target.value)}
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Agregar comentarios, observaciones o acciones tomadas..."
                />
              </div>

              <div className="flex justify-end space-x-3">
                <button
                  onClick={handleCancelEdit}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 rounded-md"
                >
                  Cancelar
                </button>
                <button
                  onClick={handleSaveComments}
                  className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md"
                >
                  Guardar
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </ProtectedRoute>
  );
}
