// app/predicciones/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import ProtectedRoute from "../../components/ProtectedRoute";
import Layout from "../../components/Layout";
import { useAuth } from "../../contexts/AuthContext";
import Cookies from "js-cookie";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from "recharts";

interface Prediction {
  id?: number;
  index?: number;
  user_id?: number;
  input_type?: string;
  input_source?: string;
  input_features?: Record<string, any>;
  input_metadata?: Record<string, any> | null;
  prediction_results?: Record<string, any> | null;
  anomaly_detected?: boolean | null;
  confidence_score?: number | null;
  anomaly_score?: number | null;
  score?: number;
  label?: string;
  horizon_shift?: number;
  predicted_future_state?: string;
  prediction_time?: string;
  status?: string;
  processing_time_ms?: number | null;
  error_message?: string | null;
  model_version?: string | null;
  created_at?: string;
  updated_at?: string | null;
}

export default function PrediccionesPage() {
  const { user } = useAuth();
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<any>(null);
  const [selected, setSelected] = useState<Prediction | null>(null);

  // Procesar datos para gráficas
  const getAnomalyTrendData = () => {
    return predictions.slice(-10).map((pred, index) => ({
      name: `P${index + 1}`,
      score: pred.score || pred.anomaly_score || 0,
      timestamp: pred.created_at
        ? new Date(pred.created_at).toLocaleDateString()
        : `Pred ${index + 1}`,
    }));
  };

  const getStatusDistribution = () => {
    const statusCount = predictions.reduce((acc, pred) => {
      const status = pred.label || pred.status || "unknown";
      acc[status] = (acc[status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(statusCount).map(([name, value]) => ({
      name,
      value,
      fill:
        name === "ANOMALY"
          ? "#ef4444"
          : name === "NORMAL"
          ? "#10b981"
          : "#6b7280",
    }));
  };

  const getScoreDistribution = () => {
    const ranges = [
      { name: "Bajo (0-0.3)", min: 0, max: 0.3, count: 0, fill: "#10b981" },
      {
        name: "Medio (0.3-0.7)",
        min: 0.3,
        max: 0.7,
        count: 0,
        fill: "#f59e0b",
      },
      { name: "Alto (0.7-1.0)", min: 0.7, max: 1.0, count: 0, fill: "#ef4444" },
    ];

    predictions.forEach((pred) => {
      const score = pred.score || pred.anomaly_score || 0;
      const range = ranges.find((r) => score >= r.min && score < r.max);
      if (range) range.count++;
    });

    return ranges;
  };

  useEffect(() => {
    fetchPredictions();
    fetchStats();
  }, []);

  // Close modal with ESC
  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected]);

  const fetchPredictions = async () => {
    try {
      const token = Cookies.get("token");
      console.log(
        "Fetching predictions with token:",
        token ? "exists" : "none"
      );

      const response = await fetch(
        `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
        }/simple/predictions`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );

      console.log("Predictions response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("Predictions data:", data);
        // Extract the results array from the response
        setPredictions(data.results || []);
      } else {
        const errorText = await response.text();
        console.error("Predictions error:", errorText);
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchStats = async () => {
    try {
      const token = Cookies.get("token");
      console.log("Fetching stats with token:", token ? "exists" : "none");

      const response = await fetch(
        `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
        }/simple/stats`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        }
      );

      console.log("Stats response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("Stats data:", data);
        setStats(data);
      } else {
        const errorText = await response.text();
        console.error("Stats error:", errorText);
      }
    } catch (error) {
      console.error("Error fetching stats:", error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800";
      case "failed":
        return "bg-red-100 text-red-800";
      case "processing":
        return "bg-yellow-100 text-yellow-800";
      case "pending":
        return "bg-gray-100 text-gray-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <ProtectedRoute>
        <Layout>
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
          </div>
        </Layout>
      </ProtectedRoute>
    );
  }

  return (
    <ProtectedRoute>
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Predicciones</h1>
            <p className="mt-1 text-sm text-gray-500">
              Historial de predicciones del modelo de anomalías
            </p>
          </div>

          {/* Modern Dashboard Grid */}
          {stats && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              {/* Left Column - Main Content */}
              <div className="lg:col-span-2 space-y-6">
                {/* Trend Chart */}
                <div className="bg-white shadow-lg rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-semibold text-gray-900">
                      Tendencia de Anomalías
                    </h3>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <span className="text-sm text-gray-600">
                        Score de Anomalía
                      </span>
                    </div>
                  </div>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={getAnomalyTrendData()}>
                        <defs>
                          <linearGradient
                            id="colorScore"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="5%"
                              stopColor="#3b82f6"
                              stopOpacity={0.3}
                            />
                            <stop
                              offset="95%"
                              stopColor="#3b82f6"
                              stopOpacity={0.05}
                            />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
                        <YAxis stroke="#6b7280" fontSize={12} domain={[0, 1]} />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#ffffff",
                            border: "1px solid #e5e7eb",
                            borderRadius: "8px",
                            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="score"
                          stroke="#3b82f6"
                          strokeWidth={2}
                          fillOpacity={1}
                          fill="url(#colorScore)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Predictions Table */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-900">
                Historial de Predicciones
              </h2>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Tipo
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Estado
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Anomalía
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Tiempo (ms)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Fecha
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Acciones
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {predictions.length === 0 ? (
                    <tr>
                      <td
                        colSpan={8}
                        className="px-6 py-12 text-center text-gray-500"
                      >
                        No hay predicciones disponibles
                      </td>
                    </tr>
                  ) : (
                    predictions.map((prediction, index) => (
                      <tr key={prediction.index || index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          #{prediction.index || index}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800">
                            {prediction.input_type || "real-time"}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span
                            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(
                              prediction.status || prediction.label || "unknown"
                            )}`}
                          >
                            {prediction.status || prediction.label}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {prediction.anomaly_detected !== null ? (
                            <span
                              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                prediction.anomaly_detected
                                  ? "bg-red-100 text-red-800"
                                  : "bg-green-100 text-green-800"
                              }`}
                            >
                              {prediction.anomaly_detected ? "Sí" : "No"}
                            </span>
                          ) : (
                            <span
                              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                prediction.label === "ANOMALY"
                                  ? "bg-red-100 text-red-800"
                                  : "bg-green-100 text-green-800"
                              }`}
                            >
                              {prediction.label === "ANOMALY" ? "Sí" : "No"}
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {prediction.confidence_score
                            ? prediction.confidence_score.toFixed(4)
                            : prediction.score
                            ? prediction.score.toFixed(4)
                            : "-"}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {prediction.processing_time_ms ||
                            prediction.horizon_shift ||
                            "-"}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {prediction.created_at
                            ? formatDate(prediction.created_at)
                            : "Ahora"}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button
                            onClick={() => setSelected(prediction)}
                            className="text-blue-600 hover:text-blue-900"
                          >
                            Ver detalles
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        {/* Details Modal */}
        {selected && (
          <div
            className="fixed inset-0 z-50 flex items-start justify-center p-4 sm:items-center bg-black/50 backdrop-blur-sm"
            onClick={() => setSelected(null)}
            role="dialog"
            aria-modal="true"
          >
            <div
              className="relative w-full max-w-2xl rounded-md bg-white shadow-lg border"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Detalle de la Predicción #{selected.index ?? selected.id}
                  </h3>
                  <button
                    onClick={() => setSelected(null)}
                    className="text-gray-500 hover:text-gray-700"
                    aria-label="Cerrar"
                  >
                    ✕
                  </button>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-gray-500">Estado</div>
                    <div className="font-medium">
                      {selected.status || selected.label || "-"}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Fecha</div>
                    <div className="font-medium">
                      {selected.created_at
                        ? new Date(selected.created_at).toLocaleString()
                        : "-"}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Anomalía</div>
                    <div className="font-medium">
                      {selected.anomaly_detected ?? selected.label === "ANOMALY"
                        ? "Sí"
                        : "No"}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Score</div>
                    <div className="font-medium">
                      {selected.anomaly_score ?? selected.score ?? "-"}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Confianza</div>
                    <div className="font-medium">
                      {selected.confidence_score ?? "-"}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Modelo</div>
                    <div className="font-medium">
                      {selected.model_version ?? "-"}
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 mb-1">Resultados (JSON)</div>
                  <pre className="overflow-auto text-xs bg-gray-50 p-3 rounded border max-h-64">
                    {JSON.stringify(
                      selected.prediction_results ?? selected,
                      null,
                      2
                    )}
                  </pre>
                </div>
                <div className="flex justify-end">
                  <button
                    onClick={() => setSelected(null)}
                    className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md"
                  >
                    Cerrar
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </Layout>
    </ProtectedRoute>
  );
}
