"use client";

import React, { useState, useEffect } from "react";
import ProtectedRoute from "../../components/ProtectedRoute";
import Layout from "../../components/Layout";
import {
  getAvailableTransformers,
  downloadHistoricalDataCSV,
  downloadHistoricalDataPDF,
  downloadPredictionsCSV,
  Transformer,
  ReportFilters,
} from "../../lib/api";
import toast from "react-hot-toast";

export default function ReportesPage() {
  const [transformers, setTransformers] = useState<Transformer[]>([]);
  const [selectedTransformer, setSelectedTransformer] =
    useState<string>("TR01");
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [sampleHours, setSampleHours] = useState<number>(6);
  const [isLoading, setIsLoading] = useState(false);
  const [reportType, setReportType] = useState<"historical" | "predictions">(
    "historical"
  );
  const [formatType, setFormatType] = useState<"csv" | "pdf">("csv");

  useEffect(() => {
    loadTransformers();
  }, []);

  const loadTransformers = async () => {
    try {
      const data = await getAvailableTransformers();
      setTransformers(data.transformers);
    } catch (error: any) {
      toast.error("Error cargando transformadores");
      console.error("Error:", error);
    }
  };

  const handleDownload = async () => {
    if (!selectedTransformer && reportType === "historical") {
      toast.error("Selecciona un transformador");
      return;
    }

    setIsLoading(true);
    try {
      let blob: Blob;
      let filename: string;

      if (reportType === "historical") {
        const filters: ReportFilters = {
          transformer_id: selectedTransformer,
          start_date: startDate || undefined,
          end_date: endDate || undefined,
          sample_hours: sampleHours,
        };

        if (formatType === "csv") {
          blob = await downloadHistoricalDataCSV(filters);
          filename = `datos_historicos_${selectedTransformer}_${
            new Date().toISOString().split("T")[0]
          }.csv`;
        } else {
          blob = await downloadHistoricalDataPDF(filters);
          filename = `reporte_historico_${selectedTransformer}_${
            new Date().toISOString().split("T")[0]
          }.pdf`;
        }
      } else {
        // Predictions report
        blob = await downloadPredictionsCSV(
          startDate || undefined,
          endDate || undefined
        );
        filename = `predicciones_${new Date().toISOString().split("T")[0]}.csv`;
      }

      // Crear enlace de descarga
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      toast.success(
        `Reporte ${formatType.toUpperCase()} descargado exitosamente`
      );
    } catch (error: any) {
      toast.error(`Error descargando reporte: ${error.message}`);
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const clearFilters = () => {
    setStartDate("");
    setEndDate("");
    setSampleHours(6);
  };

  return (
    <ProtectedRoute>
      <Layout>
        <div className="w-full">
          {/* Header */}
          <div className="py-6">
            <h1 className="text-2xl font-semibold text-gray-900">
              Generar Reportes
            </h1>
            <p className="text-sm text-gray-600 mt-2">
              Exporta datos históricos y predicciones en formato PDF o CSV
            </p>
          </div>

          {/* Main Content */}
          <div className="bg-white rounded-2xl shadow-lg p-8">
            {/* Report Type Selection */}
            <div className="mb-8">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Tipo de Reporte
              </h2>
              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={() => setReportType("historical")}
                  className={`p-4 rounded-xl border-2 transition-all duration-200 ${
                    reportType === "historical"
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                        />
                      </svg>
                    </div>
                    <div>
                      <h3 className="font-medium">Datos Históricos</h3>
                      <p className="text-sm text-gray-500">
                        Sensores y mediciones
                      </p>
                    </div>
                  </div>
                </button>

                <button
                  onClick={() => setReportType("predictions")}
                  className={`p-4 rounded-xl border-2 transition-all duration-200 ${
                    reportType === "predictions"
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 10V3L4 14h7v7l9-11h-7z"
                        />
                      </svg>
                    </div>
                    <div>
                      <h3 className="font-medium">Predicciones</h3>
                      <p className="text-sm text-gray-500">
                        Análisis predictivo
                      </p>
                    </div>
                  </div>
                </button>
              </div>
            </div>

            {/* Format Selection */}
            <div className="mb-8">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Formato de Exportación
              </h2>
              <div className="flex space-x-4">
                <button
                  onClick={() => setFormatType("csv")}
                  className={`px-6 py-3 rounded-lg border-2 transition-all duration-200 ${
                    formatType === "csv"
                      ? "border-green-500 bg-green-50 text-green-700"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    <span className="font-medium">CSV</span>
                  </div>
                </button>

                <button
                  onClick={() => setFormatType("pdf")}
                  disabled={reportType === "predictions"}
                  className={`px-6 py-3 rounded-lg border-2 transition-all duration-200 ${
                    formatType === "pdf" && reportType !== "predictions"
                      ? "border-red-500 bg-red-50 text-red-700"
                      : reportType === "predictions"
                      ? "border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                      />
                    </svg>
                    <span className="font-medium">PDF</span>
                  </div>
                </button>
              </div>
              {reportType === "predictions" && (
                <p className="text-sm text-gray-500 mt-2">
                  Los reportes de predicciones solo están disponibles en formato
                  CSV
                </p>
              )}
            </div>

            {/* Filters */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">
                  Filtros y Configuración
                </h2>
                <button
                  onClick={clearFilters}
                  className="text-sm text-gray-500 hover:text-gray-700"
                >
                  Limpiar filtros
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* Transformer Selection (only for historical data) */}
                {reportType === "historical" && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Transformador
                    </label>
                    <select
                      value={selectedTransformer}
                      onChange={(e) => setSelectedTransformer(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      {transformers.map((transformer) => (
                        <option key={transformer.id} value={transformer.id}>
                          {transformer.name} ({transformer.id})
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Date Filters */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Fecha Inicio
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Fecha Fin
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                {/* Sample Hours (only for historical data) */}
                {reportType === "historical" && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Intervalo de Muestreo (horas)
                    </label>
                    <select
                      value={sampleHours}
                      onChange={(e) => setSampleHours(Number(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value={1}>Cada hora</option>
                      <option value={6}>Cada 6 horas</option>
                      <option value={12}>Cada 12 horas</option>
                      <option value={24}>Cada día</option>
                    </select>
                  </div>
                )}
              </div>
            </div>

            {/* Preview Info */}
            <div className="mb-8 p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-2">
                Resumen del Reporte
              </h3>
              <div className="text-sm text-gray-600 space-y-1">
                <p>
                  <span className="font-medium">Tipo:</span>{" "}
                  {reportType === "historical"
                    ? "Datos Históricos"
                    : "Predicciones"}
                </p>
                <p>
                  <span className="font-medium">Formato:</span>{" "}
                  {formatType.toUpperCase()}
                </p>
                {reportType === "historical" && (
                  <p>
                    <span className="font-medium">Transformador:</span>{" "}
                    {selectedTransformer}
                  </p>
                )}
                {startDate && (
                  <p>
                    <span className="font-medium">Desde:</span> {startDate}
                  </p>
                )}
                {endDate && (
                  <p>
                    <span className="font-medium">Hasta:</span> {endDate}
                  </p>
                )}
                {reportType === "historical" && (
                  <p>
                    <span className="font-medium">Muestreo:</span> Cada{" "}
                    {sampleHours} horas
                  </p>
                )}
              </div>
            </div>

            {/* Download Button */}
            <div className="flex justify-center">
              <button
                onClick={handleDownload}
                disabled={isLoading}
                className="px-8 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white font-medium rounded-lg hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl flex items-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <svg
                      className="animate-spin h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    <span>Generando...</span>
                  </>
                ) : (
                  <>
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    <span>Descargar Reporte</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </Layout>
    </ProtectedRoute>
  );
}
