"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import ProtectedRoute from "../../components/ProtectedRoute";
import Layout from "../../components/Layout";
import { useAuth } from "../../contexts/AuthContext";

interface Transformer {
  id: string;
  name: string;
  status: string;
  location: string;
  lastPrediction: string;
  model: string;
  predictionHorizon: string;
}

interface TransformerHistory {
  transformer_id: string;
  name: string;
  data_range: {
    start: string;
    end: string;
    total_records: number;
    months_available: number;
  };
  historical_data: Array<{
    timestamp: string;
    temp_oil: number;
    temp_ambient: number;
    voltage: number;
    current_load: number;
    power_apparent: number;
    tap_position: number;
    estado_operacional: string;
    nivel_severidad: number;
    temp_hot_spot: number;
    gradient_hot_oil: number;
  }>;
  current_measurements: {
    timestamp?: string;
    temp_oil?: number;
    temp_ambient?: number;
    voltage?: number;
    current_load?: number;
    power_apparent?: number;
    tap_position?: number;
    estado_operacional?: string;
    nivel_severidad?: number;
    temp_hot_spot?: number;
    gradient_hot_oil?: number;
  };
  data_info: {
    sample_interval: string;
    total_samples: number;
    features_available: string[];
  };
}

export default function TransformadoresPage() {
  const [transformers, setTransformers] = useState<Transformer[]>([]);
  const [selectedTransformer, setSelectedTransformer] =
    useState<Transformer | null>(null);
  const [history, setHistory] = useState<TransformerHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [currentView, setCurrentView] = useState<"list" | "details">("list");
  const [filters, setFilters] = useState({
    dateRange: {
      from: "",
      to: "",
    },
    temperatureRange: {
      min: "",
      max: "",
    },
    currentRange: {
      min: "",
      max: "",
    },
    powerRange: {
      min: "",
      max: "",
    },
  });
  const [showFilters, setShowFilters] = useState(false);
  const { user } = useAuth();

  useEffect(() => {
    fetchTransformers();
  }, []);

  const fetchTransformers = async () => {
    try {
      const response = await fetch("http://localhost:8000/simple/transformers");
      if (response.ok) {
        const data = await response.json();
        setTransformers(data.transformers || []);
      } else {
        console.error("Failed to fetch transformers");
      }
    } catch (error) {
      console.error("Error fetching transformers:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchTransformerHistory = async (transformerId: string) => {
    setLoadingHistory(true);
    try {
      const response = await fetch(
        `http://localhost:8000/simple/transformers/${transformerId}/history`
      );
      if (response.ok) {
        const data = await response.json();
        setHistory(data);
      } else {
        console.error("Failed to fetch transformer history");
      }
    } catch (error) {
      console.error("Error fetching transformer history:", error);
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleTransformerSelect = (transformer: Transformer) => {
    setSelectedTransformer(transformer);
    setCurrentView("details");
    fetchTransformerHistory(transformer.id);
  };

  const handleBackToList = () => {
    setCurrentView("list");
    setSelectedTransformer(null);
    setHistory(null);
    setShowFilters(false);
  };

  const handleFilterChange = (
    filterType: string,
    field: string,
    value: string
  ) => {
    setFilters((prev) => ({
      ...prev,
      [filterType]: {
        ...prev[filterType as keyof typeof prev],
        [field]: value,
      },
    }));
  };

  const clearFilters = () => {
    setFilters({
      dateRange: { from: "", to: "" },
      temperatureRange: { min: "", max: "" },
      currentRange: { min: "", max: "" },
      powerRange: { min: "", max: "" },
    });
  };

  const applyFilters = (data: any[]) => {
    return data.filter((item) => {
      // Date range filter
      if (filters.dateRange.from || filters.dateRange.to) {
        const itemDate = new Date(item.timestamp);
        const fromDate = filters.dateRange.from
          ? new Date(filters.dateRange.from)
          : null;
        const toDate = filters.dateRange.to
          ? new Date(filters.dateRange.to)
          : null;

        if (fromDate && itemDate < fromDate) return false;
        if (toDate && itemDate > toDate) return false;
      }

      // Temperature range filter (using temp_oil)
      if (filters.temperatureRange.min || filters.temperatureRange.max) {
        const minTemp = filters.temperatureRange.min
          ? parseFloat(filters.temperatureRange.min)
          : null;
        const maxTemp = filters.temperatureRange.max
          ? parseFloat(filters.temperatureRange.max)
          : null;

        if (minTemp && item.temp_oil < minTemp) return false;
        if (maxTemp && item.temp_oil > maxTemp) return false;
      }

      // Current range filter
      if (filters.currentRange.min || filters.currentRange.max) {
        const minCurrent = filters.currentRange.min
          ? parseFloat(filters.currentRange.min)
          : null;
        const maxCurrent = filters.currentRange.max
          ? parseFloat(filters.currentRange.max)
          : null;

        if (minCurrent && item.current_load < minCurrent) return false;
        if (maxCurrent && item.current_load > maxCurrent) return false;
      }

      // Power range filter
      if (filters.powerRange.min || filters.powerRange.max) {
        const minPower = filters.powerRange.min
          ? parseFloat(filters.powerRange.min)
          : null;
        const maxPower = filters.powerRange.max
          ? parseFloat(filters.powerRange.max)
          : null;

        if (minPower && item.power_apparent < minPower) return false;
        if (maxPower && item.power_apparent > maxPower) return false;
      }

      return true;
    });
  };

  const hasActiveFilters = () => {
    return (
      filters.dateRange.from ||
      filters.dateRange.to ||
      filters.temperatureRange.min ||
      filters.temperatureRange.max ||
      filters.currentRange.min ||
      filters.currentRange.max ||
      filters.powerRange.min ||
      filters.powerRange.max
    );
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "ACTIVE":
        return "text-green-600 bg-green-100";
      case "NORMAL":
        return "text-green-600 bg-green-100";
      case "ANOMALY":
        return "text-red-600 bg-red-100";
      case "CRITICO":
        return "text-red-600 bg-red-100";
      default:
        return "text-gray-600 bg-gray-100";
    }
  };

  if (loading) {
    return (
      <ProtectedRoute>
        <Layout>
          <div className="min-h-screen bg-gray-50 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-4 text-gray-600">Cargando transformadores...</p>
            </div>
          </div>
        </Layout>
      </ProtectedRoute>
    );
  }

  return (
    <ProtectedRoute>
      <Layout>
        <div className="min-h-screen bg-gray-0">
          <div className="w-full">
            {/* Header */}
            <div className="py-6">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">
                    Transformadores
                  </h1>
                  <p className="mt-2 text-gray-600">
                    Gestión y monitoreo de transformadores
                  </p>
                </div>
                <Link
                  href="/predicciones"
                  className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
                >
                  Ver Predicciones
                </Link>
              </div>
            </div>

            {/* Conditional Rendering based on current view */}
            {currentView === "list" ? (
              // List View
              <div>
                {/* Transformers List - Full Width */}
                <div className="bg-white shadow rounded-lg">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-900">
                      Transformadores Disponibles
                    </h2>
                  </div>
                  <div className="p-4">
                    {transformers.length === 0 ? (
                      <p className="text-gray-500 text-center py-8">
                        No hay transformadores disponibles
                      </p>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {transformers.map((transformer) => (
                          <div
                            key={transformer.id}
                            className="p-4 border border-gray-200 rounded-lg cursor-pointer transition-all duration-200 hover:border-blue-300 hover:shadow-md bg-white hover:bg-blue-50/30"
                            onClick={() => handleTransformerSelect(transformer)}
                          >
                            <div className="flex items-start justify-between mb-4">
                              <div className="flex-1">
                                <h3 className="text-lg font-semibold text-gray-900 mb-1">
                                  {transformer.name}
                                </h3>
                                <p className="text-sm text-gray-600 mb-2">
                                  {transformer.location}
                                </p>
                              </div>
                              <span
                                className={`px-3 py-1 text-xs font-medium rounded-full ${getStatusColor(
                                  transformer.status
                                )}`}
                              >
                                {transformer.status}
                              </span>
                            </div>
                            <div className="space-y-2 text-sm text-gray-500">
                              <div className="flex justify-between">
                                <span>Modelo:</span>
                                <span className="font-medium">
                                  {transformer.model}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span>Horizonte:</span>
                                <span className="font-medium">
                                  {transformer.predictionHorizon}
                                </span>
                              </div>
                            </div>
                            <div className="mt-4 pt-4 border-t border-gray-100">
                              <div className="flex items-center text-blue-600 text-sm font-medium">
                                <svg
                                  className="h-4 w-4 mr-2"
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M9 5l7 7-7 7"
                                  />
                                </svg>
                                Ver Detalles
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              // Details View
              <div>
                {/* Back Button */}
                <div className="mb-6">
                  <button
                    onClick={handleBackToList}
                    className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors duration-200"
                  >
                    <svg
                      className="h-4 w-4 mr-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 19l-7-7 7-7"
                      />
                    </svg>
                    Volver a la Lista
                  </button>
                </div>

                {selectedTransformer && (
                  <div className="space-y-6">
                    {/* Simplified Transformer Info */}
                    <div className="bg-white shadow rounded-lg">
                      <div className="px-6 py-4 border-b border-gray-200">
                        <div className="flex items-center justify-between">
                          <div>
                            <h2 className="text-lg font-medium text-gray-900">
                              {selectedTransformer.name}
                            </h2>
                            <p className="text-sm text-gray-600 mt-1">
                              {selectedTransformer.location} •{" "}
                              {selectedTransformer.id}
                            </p>
                          </div>
                          <span
                            className={`px-3 py-1 text-sm font-medium rounded-full ${getStatusColor(
                              selectedTransformer.status
                            )}`}
                          >
                            {selectedTransformer.status}
                          </span>
                        </div>
                      </div>
                      <div className="p-4">
                        <div className="flex items-center justify-between text-sm text-gray-600">
                          <span>Modelo: {selectedTransformer.model}</span>
                          <span>
                            Predicción: {selectedTransformer.predictionHorizon}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Historical Data */}
                    <div className="bg-white shadow rounded-lg">
                      <div className="px-6 py-4 border-b border-gray-200">
                        <h2 className="text-lg font-medium text-gray-900">
                          Datos Históricos
                        </h2>
                      </div>
                      <div className="p-4">
                        {loadingHistory ? (
                          <div className="text-center py-8">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                            <p className="mt-2 text-gray-600">
                              Cargando historial...
                            </p>
                          </div>
                        ) : history ? (
                          <div className="space-y-6">
                            {/* Compact Data Summary */}
                            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-100">
                              <div className="flex items-center justify-between mb-4">
                                <h3 className="text-sm font-medium text-gray-900">
                                  Resumen de Datos
                                </h3>
                                <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded-full">
                                  {history.data_range.total_records.toLocaleString()}{" "}
                                  registros
                                </span>
                              </div>

                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                                {/* Date Range */}
                                <div className="space-y-1">
                                  <span className="text-gray-500 text-xs">
                                    Período:
                                  </span>
                                  <p className="font-medium text-gray-900">
                                    {new Date(
                                      history.data_range.start
                                    ).toLocaleDateString()}{" "}
                                    -{" "}
                                    {new Date(
                                      history.data_range.end
                                    ).toLocaleDateString()}
                                  </p>
                                </div>

                                {/* Current Status */}
                                <div className="space-y-1">
                                  <span className="text-gray-500 text-xs">
                                    Estado Actual:
                                  </span>
                                  <div className="flex items-center space-x-4">
                                    <span className="font-medium text-gray-900">
                                      {history.current_measurements.temp_oil?.toFixed(
                                        1
                                      )}
                                      °C
                                    </span>
                                    <span className="text-gray-400">•</span>
                                    <span className="font-medium text-gray-900">
                                      {history.current_measurements.current_load?.toFixed(
                                        1
                                      )}
                                      A
                                    </span>
                                    <span className="text-gray-400">•</span>
                                    <span className="font-medium text-gray-900">
                                      {history.current_measurements.power_apparent?.toFixed(
                                        1
                                      )}
                                      kVA
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* Filters Section */}
                            <div className="mb-6">
                              <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center space-x-3">
                                  <h3 className="text-sm font-medium text-gray-900">
                                    Datos Históricos
                                  </h3>
                                  <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
                                    {
                                      applyFilters(history.historical_data)
                                        .length
                                    }{" "}
                                    de {history.historical_data.length}
                                  </span>
                                </div>
                                <button
                                  onClick={() => setShowFilters(!showFilters)}
                                  className={`inline-flex items-center px-3 py-1.5 text-xs font-medium rounded-md transition-colors duration-200 ${
                                    hasActiveFilters()
                                      ? "bg-blue-100 text-blue-700 border border-blue-200"
                                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                                  }`}
                                >
                                  <svg
                                    className="h-3 w-3 mr-1.5"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth={2}
                                      d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.707A1 1 0 013 7V4z"
                                    />
                                  </svg>
                                  Filtros
                                  {hasActiveFilters() && (
                                    <span className="ml-1.5 inline-flex items-center justify-center w-4 h-4 text-xs font-medium text-blue-600 bg-blue-200 rounded-full">
                                      !
                                    </span>
                                  )}
                                </button>
                              </div>

                              {/* Filters Panel */}
                              {showFilters && (
                                <div className="bg-gray-50 rounded-lg p-3 mb-4 border border-gray-200">
                                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                                    {/* Date Range Filter */}
                                    <div>
                                      <label className="block text-xs font-medium text-gray-700 mb-2">
                                        Rango de Fechas
                                      </label>
                                      <div className="space-y-1.5">
                                        <input
                                          type="date"
                                          value={filters.dateRange.from}
                                          onChange={(e) =>
                                            handleFilterChange(
                                              "dateRange",
                                              "from",
                                              e.target.value
                                            )
                                          }
                                          className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                                          placeholder="Desde"
                                        />
                                        <input
                                          type="date"
                                          value={filters.dateRange.to}
                                          onChange={(e) =>
                                            handleFilterChange(
                                              "dateRange",
                                              "to",
                                              e.target.value
                                            )
                                          }
                                          className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                                          placeholder="Hasta"
                                        />
                                      </div>
                                    </div>

                                    {/* Temperature Range Filter */}
                                    <div>
                                      <label className="block text-xs font-medium text-gray-700 mb-2">
                                        Temperatura Aceite (°C)
                                      </label>
                                      <div className="space-y-1.5">
                                        <input
                                          type="number"
                                          value={filters.temperatureRange.min}
                                          onChange={(e) =>
                                            handleFilterChange(
                                              "temperatureRange",
                                              "min",
                                              e.target.value
                                            )
                                          }
                                          className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                                          placeholder="Mín"
                                          step="0.1"
                                        />
                                        <input
                                          type="number"
                                          value={filters.temperatureRange.max}
                                          onChange={(e) =>
                                            handleFilterChange(
                                              "temperatureRange",
                                              "max",
                                              e.target.value
                                            )
                                          }
                                          className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                                          placeholder="Máx"
                                          step="0.1"
                                        />
                                      </div>
                                    </div>

                                    {/* Current Range Filter */}
                                    <div>
                                      <label className="block text-xs font-medium text-gray-700 mb-2">
                                        Corriente (A)
                                      </label>
                                      <div className="space-y-1.5">
                                        <input
                                          type="number"
                                          value={filters.currentRange.min}
                                          onChange={(e) =>
                                            handleFilterChange(
                                              "currentRange",
                                              "min",
                                              e.target.value
                                            )
                                          }
                                          className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                                          placeholder="Mín"
                                          step="0.1"
                                        />
                                        <input
                                          type="number"
                                          value={filters.currentRange.max}
                                          onChange={(e) =>
                                            handleFilterChange(
                                              "currentRange",
                                              "max",
                                              e.target.value
                                            )
                                          }
                                          className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                                          placeholder="Máx"
                                          step="0.1"
                                        />
                                      </div>
                                    </div>

                                    {/* Power Range Filter */}
                                    <div>
                                      <label className="block text-xs font-medium text-gray-700 mb-2">
                                        Potencia (kVA)
                                      </label>
                                      <div className="space-y-1.5">
                                        <input
                                          type="number"
                                          value={filters.powerRange.min}
                                          onChange={(e) =>
                                            handleFilterChange(
                                              "powerRange",
                                              "min",
                                              e.target.value
                                            )
                                          }
                                          className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                                          placeholder="Mín"
                                          step="0.1"
                                        />
                                        <input
                                          type="number"
                                          value={filters.powerRange.max}
                                          onChange={(e) =>
                                            handleFilterChange(
                                              "powerRange",
                                              "max",
                                              e.target.value
                                            )
                                          }
                                          className="w-full px-2 py-1.5 text-xs border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                                          placeholder="Máx"
                                          step="0.1"
                                        />
                                      </div>
                                    </div>
                                  </div>

                                  {/* Filter Actions */}
                                  <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-200">
                                    <div className="text-xs text-gray-500">
                                      {hasActiveFilters()
                                        ? "Filtros activos"
                                        : "Sin filtros aplicados"}
                                    </div>
                                    <div className="flex space-x-2">
                                      <button
                                        onClick={clearFilters}
                                        className="px-2 py-1 text-xs font-medium text-gray-600 bg-white border border-gray-300 rounded hover:bg-gray-50 transition-colors duration-200"
                                      >
                                        Limpiar
                                      </button>
                                      <button
                                        onClick={() => setShowFilters(false)}
                                        className="px-2 py-1 text-xs font-medium text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors duration-200"
                                      >
                                        Aplicar
                                      </button>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>

                            {/* Historical Data Table */}
                            <div>
                              <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200">
                                  <thead className="bg-gray-50">
                                    <tr>
                                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Fecha
                                      </th>
                                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Temp. (°C)
                                      </th>
                                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Voltaje (V)
                                      </th>
                                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Potencia (kVA)
                                      </th>
                                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        P. Caliente (°C)
                                      </th>
                                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Corriente (A)
                                      </th>
                                    </tr>
                                  </thead>
                                  <tbody className="bg-white divide-y divide-gray-200">
                                    {applyFilters(history.historical_data)
                                      .slice(0, 25)
                                      .map((data, index) => (
                                        <tr
                                          key={index}
                                          className="hover:bg-gray-50"
                                        >
                                          <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-900">
                                            {new Date(
                                              data.timestamp
                                            ).toLocaleDateString()}
                                          </td>
                                          <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-900">
                                            {data.temp_oil.toFixed(1)}
                                          </td>
                                          <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-900">
                                            {data.voltage.toFixed(1)}
                                          </td>
                                          <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-900">
                                            {data.power_apparent.toFixed(1)}
                                          </td>
                                          <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-900">
                                            {data.temp_hot_spot.toFixed(1)}
                                          </td>
                                          <td className="px-3 py-2 whitespace-nowrap text-xs text-gray-900">
                                            {data.current_load.toFixed(1)}
                                          </td>
                                        </tr>
                                      ))}
                                  </tbody>
                                </table>
                              </div>
                              {applyFilters(history.historical_data).length >
                                25 && (
                                <p className="mt-2 text-sm text-gray-500 text-center">
                                  Mostrando los primeros 25 registros de{" "}
                                  {applyFilters(history.historical_data).length}{" "}
                                  filtrados
                                </p>
                              )}
                            </div>
                          </div>
                        ) : (
                          <p className="text-gray-500 text-center py-8">
                            No hay datos históricos disponibles
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </Layout>
    </ProtectedRoute>
  );
}
