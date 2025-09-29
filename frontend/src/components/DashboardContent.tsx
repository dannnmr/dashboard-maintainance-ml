// components/DashboardContent.tsx
"use client";

import React, { useEffect, useState } from "react";
import {
  fetchFeatures,
  getMaintenanceResults,
  PredictItem,
  PredictResponse,
} from "../lib/api";
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
  RadialBarChart,
  RadialBar,
} from "recharts";

export default function DashboardContent() {
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [featureOrder, setFeatureOrder] = useState<string[]>([]);
  const [modelVersion, setModelVersion] = useState<string>("unknown");
  const [results, setResults] = useState<PredictItem[]>([]);
  const [dataInfo, setDataInfo] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictResponse[]>(
    []
  );
  const ready = featureOrder.length > 0;

  // Procesar datos para gráficas
  const getAnomalyTrendData = () => {
    return results.slice(-10).map((result, index) => ({
      name: `P${index + 1}`,
      score: result.score || 0,
      timestamp: result.prediction_time || `Pred ${index + 1}`,
    }));
  };

  const getStatusDistribution = () => {
    const statusCount = results.reduce((acc, result) => {
      const status = result.label || "unknown";
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

    results.forEach((result) => {
      const score = result.score || 0;
      const range = ranges.find((r) => score >= r.min && score < r.max);
      if (range) range.count++;
    });

    return ranges;
  };

  const getFeatureImportanceData = () => {
    // Si tenemos información de características, podemos mostrar su importancia
    return featureOrder.slice(0, 8).map((feature, index) => ({
      name: feature.length > 15 ? feature.substring(0, 15) + "..." : feature,
      importance: Math.random() * 100, // Placeholder - en un caso real vendría del modelo
      fill: `hsl(${200 + index * 20}, 70%, 50%)`,
    }));
  };

  // Load features on mount
  useEffect(() => {
    (async () => {
      try {
        const f = await fetchFeatures();
        setFeatureOrder(f.feature_order);
        setModelVersion(f.model_version);
      } catch (e) {
        console.error(e);
        alert("No se pudo cargar /features del backend.");
      }
    })();
  }, []);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(async () => {
      await getETLResults();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Load real transformer data on startup
  useEffect(() => {
    if (ready) {
      getETLResults();
    }
  }, [ready]);

  const getETLResults = async () => {
    setLoading(true);
    try {
      const data = await getMaintenanceResults();
      setResults(data.results);
      setDataInfo(data.data_info);
      setLastUpdate(new Date());

      // Add to prediction history
      setPredictionHistory((prev) => [data, ...prev.slice(0, 9)]); // Keep last 10 predictions

      console.log("📊 Datos del ETL:", data);
    } catch (e: any) {
      console.error(e);
      alert(`Error al obtener resultados del ETL: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  };

  const hasResults = results.length > 0;
  const currentResult = results[0];
  const isAnomaly = currentResult?.label === "ANOMALY";

  return (
    <main className="min-h-screen bg-white">
      {/* Header */}
      <div className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-8">
          <div className="flex justify-between items-center py-8">
            <div>
              <h1 className="text-2xl font-medium text-gray-900">
                Transformer Future Prediction Dashboard
              </h1>
              <p className="text-sm text-gray-500 mt-2">
                Predicting transformer state{" "}
                {dataInfo?.horizon_shift
                  ? `${Math.round(dataInfo.horizon_shift / 24)} days ahead`
                  : "15 days ahead"}{" "}
                • Model: {modelVersion} •{" "}
                {lastUpdate
                  ? `Last update: ${lastUpdate.toLocaleTimeString()}`
                  : "Loading data..."}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-8 py-12">
        {!ready && (
          <div className="text-center py-16">
            <div className="animate-spin h-8 w-8 border-2 border-gray-300 border-t-gray-600 rounded-full mx-auto mb-4"></div>
            <p className="text-gray-500">
              Loading transformer monitoring system...
            </p>
          </div>
        )}

        {ready && (
          <div className="space-y-8">
            {/* Future Prediction Alert */}

            {/* Prediction Details - Compact Grid */}
            {hasResults && (
              <div className="space-y-6">
                {/* Header */}

                {/* Charts Row */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  {/* Health Index Gauge */}
                  <div className="bg-white rounded-xl shadow-sm p-4 flex flex-col items-center">
                    <div className="relative w-20 h-20 mb-3">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadialBarChart
                          cx="50%"
                          cy="50%"
                          innerRadius="60%"
                          outerRadius="85%"
                          data={[
                            {
                              name: "Health",
                              value: Math.max(
                                0,
                                Math.min(
                                  100,
                                  (1 - (currentResult?.score || 0)) * 100
                                )
                              ),
                              fill:
                                (1 - (currentResult?.score || 0)) * 100 > 70
                                  ? "#10b981"
                                  : (1 - (currentResult?.score || 0)) * 100 > 40
                                  ? "#f59e0b"
                                  : "#ef4444",
                            },
                          ]}
                          startAngle={90}
                          endAngle={-270}
                        >
                          <RadialBar
                            dataKey="value"
                            cornerRadius={100}
                            background={{ fill: "#f3f4f6" }}
                          />
                        </RadialBarChart>
                      </ResponsiveContainer>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-lg font-bold text-gray-900">
                          {Math.round(
                            Math.max(
                              0,
                              Math.min(
                                100,
                                (1 - (currentResult?.score || 0)) * 100
                              )
                            )
                          )}
                        </span>
                      </div>
                    </div>
                    <span className="text-xs text-gray-700 font-medium mb-2">
                      Health Index
                    </span>
                    <div
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        currentResult?.predicted_future_state === "CRITICO"
                          ? "bg-red-50 text-red-700"
                          : "bg-green-50 text-green-700"
                      }`}
                    >
                      {currentResult?.predicted_future_state || "NORMAL"}
                    </div>
                  </div>

                  {/* Status Distribution Pie Chart */}
                  <div className="bg-white rounded-xl shadow-sm p-4">
                    <h3 className="text-sm font-medium text-gray-900 mb-3">
                      Estado de Predicciones
                    </h3>
                    <div className="h-32">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={getStatusDistribution()}
                            cx="50%"
                            cy="50%"
                            innerRadius={25}
                            outerRadius={45}
                            paddingAngle={2}
                            dataKey="value"
                          >
                            {getStatusDistribution().map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#ffffff",
                              border: "1px solid #e5e7eb",
                              borderRadius: "8px",
                              boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                            }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-2 space-y-1">
                      {getStatusDistribution().map((item, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between text-xs"
                        >
                          <div className="flex items-center">
                            <div
                              className="w-2 h-2 rounded-full mr-2"
                              style={{ backgroundColor: item.fill }}
                            ></div>
                            <span className="text-gray-700 font-medium">
                              {item.name}
                            </span>
                          </div>
                          <span className="font-medium text-gray-900">
                            {item.value}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Score Distribution Bar Chart */}
                  <div className="bg-white rounded-xl shadow-sm p-4">
                    <h3 className="text-sm font-medium text-gray-900 mb-3">
                      Distribución de Scores
                    </h3>
                    <div className="h-32">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={getScoreDistribution()}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#e5e7eb"
                          />
                          <XAxis
                            dataKey="name"
                            stroke="#6b7280"
                            fontSize={10}
                            tick={{ fontSize: 10 }}
                          />
                          <YAxis
                            stroke="#6b7280"
                            fontSize={10}
                            tick={{ fontSize: 10 }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#ffffff",
                              border: "1px solid #e5e7eb",
                              borderRadius: "8px",
                              boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                              fontSize: "12px",
                            }}
                          />
                          <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                            {getScoreDistribution().map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                {/* Info Cards Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Model Configuration */}
                  <div className="bg-white rounded-xl shadow-sm p-4">
                    <h3 className="text-sm font-medium text-gray-900 mb-3">
                      Configuración del Modelo
                    </h3>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Versión:
                        </span>
                        <span className="font-medium">{modelVersion}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Lookback:
                        </span>
                        <span className="font-medium">
                          {dataInfo?.lookback_window || "N/A"}h
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Horizonte:
                        </span>
                        <span className="font-medium">
                          {dataInfo?.horizon_shift
                            ? `${Math.round(dataInfo.horizon_shift / 24)}d`
                            : "15d"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          AE Only:
                        </span>
                        <span className="font-medium">
                          {dataInfo?.ae_only_mode ? "Sí" : "No"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Threshold:
                        </span>
                        <span className="font-medium">
                          {dataInfo?.threshold_used?.toFixed(3) || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Total Rows:
                        </span>
                        <span className="font-medium">
                          {dataInfo?.total_rows || "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Current Prediction Details */}
                  <div className="bg-white rounded-xl shadow-sm p-4">
                    <h3 className="text-sm font-medium text-gray-900 mb-3">
                      Predicción Actual
                    </h3>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Index:
                        </span>
                        <span className="font-medium">
                          {currentResult?.index}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Score:
                        </span>
                        <span className="font-medium">
                          {currentResult?.score?.toFixed(6)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Estado:
                        </span>
                        <span
                          className={`font-medium ${
                            currentResult?.predicted_future_state === "CRITICO"
                              ? "text-red-600"
                              : "text-green-600"
                          }`}
                        >
                          {currentResult?.predicted_future_state || "NORMAL"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Tiempo:
                        </span>
                        <span className="font-medium">
                          {currentResult?.prediction_time ||
                            `${
                              dataInfo?.horizon_shift
                                ? Math.round(dataInfo.horizon_shift / 24)
                                : 15
                            } days ahead`}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-700 font-medium">
                          Confidence:
                        </span>
                        <span className="font-medium">
                          {currentResult
                            ? currentResult.label === "ANOMALY"
                              ? `${(
                                  (currentResult.score /
                                    (dataInfo?.threshold_used || 1)) *
                                  100
                                ).toFixed(1)}%`
                              : `${(
                                  (1 -
                                    currentResult.score /
                                      (dataInfo?.threshold_used || 1)) *
                                  100
                                ).toFixed(1)}%`
                            : "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Prediction History Row */}
                {predictionHistory.length > 0 && (
                  <div className="bg-white rounded-xl shadow-sm p-4">
                    <h3 className="text-sm font-medium text-gray-900 mb-3">
                      Historial de Predicciones
                    </h3>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-xs">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-1">Tiempo</th>
                            <th className="text-left py-1">Score</th>
                            <th className="text-left py-1">Estado</th>
                          </tr>
                        </thead>
                        <tbody>
                          {predictionHistory.slice(0, 8).map((pred, i) => (
                            <tr key={i} className="border-b border-gray-100">
                              <td className="py-1 text-gray-600">
                                {new Date().toLocaleTimeString()}
                              </td>
                              <td className="py-1 font-medium">
                                {pred.results[0]?.score.toFixed(4)}
                              </td>
                              <td className="py-1">
                                <span
                                  className={`px-2 py-0.5 rounded text-xs ${
                                    pred.results[0]?.predicted_future_state ===
                                    "CRITICO"
                                      ? "bg-red-100 text-red-700"
                                      : "bg-green-100 text-green-700"
                                  }`}
                                >
                                  {pred.results[0]?.predicted_future_state ||
                                    "NORMAL"}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
