"use client";

import { useEffect, useMemo, useState } from "react";
import {
  fetchFeatures,
  predictFromRecords,
  getMaintenanceResults,
  PredictItem,
} from "@/lib/api";

type Row = Record<string, number>;

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [featureOrder, setFeatureOrder] = useState<string[]>([]);
  const [modelVersion, setModelVersion] = useState<string>("unknown");
  const [rows, setRows] = useState<Row[]>([]);
  const [results, setResults] = useState<PredictItem[]>([]);
  const ready = featureOrder.length > 0;

  // Load features on mount
  useEffect(() => {
    (async () => {
      try {
        const f = await fetchFeatures();
        setFeatureOrder(f.feature_order);
        setModelVersion(f.model_version);
        // initialize with 1 empty row
        const initial: Row = {};
        f.feature_order.forEach((k) => (initial[k] = 0));
        setRows([initial]);
      } catch (e) {
        console.error(e);
        alert("No se pudo cargar /features del backend.");
      }
    })();
  }, []);

  // Handlers
  const setCell = (rIdx: number, key: string, value: number) => {
    setRows((prev) => {
      const next = prev.map((r) => ({ ...r }));
      next[rIdx][key] = value;
      return next;
    });
  };

  const addRow = () => {
    const blank: Row = {};
    featureOrder.forEach((k) => (blank[k] = 0));
    setRows((prev) => [...prev, blank]);
  };

  const randomize = () => {
    setRows((prev) =>
      prev.map((r) => {
        const nr: Row = {};
        featureOrder.forEach((k) => {
          // Basic randomization (adjust range if you know realistic bounds)
          nr[k] = Number((Math.random() * 100).toFixed(3));
        });
        return nr;
      })
    );
  };

  const predict = async () => {
    setLoading(true);
    setResults([]);
    try {
      const data = await predictFromRecords(rows);
      setResults(data.results);
    } catch (e: any) {
      console.error(e);
      alert(`Error al predecir: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  };

  const getETLResults = async () => {
    setLoading(true);
    setResults([]);
    try {
      const data = await getMaintenanceResults();
      setResults(data.results);
      console.log("📊 Datos del ETL:", data);
    } catch (e: any) {
      console.error(e);
      alert(`Error al obtener resultados del ETL: ${e?.message || e}`);
    } finally {
      setLoading(false);
    }
  };

  const hasResults = results.length > 0;

  return (
    <main className="min-h-screen p-6">
      <h1 className="text-2xl font-bold">Predictive Maintenance Dashboard</h1>
      <p className="text-sm text-gray-500">Model version: {modelVersion}</p>

      {!ready && <p className="mt-4">Cargando features…</p>}

      {ready && (
        <>
          <div className="mt-6 overflow-auto border rounded-xl">
            <table className="min-w-[900px] w-full text-sm">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left">#</th>
                  {featureOrder.map((f) => (
                    <th key={f} className="px-3 py-2 text-left">
                      {f}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rIdx) => (
                  <tr key={rIdx} className="odd:bg-white even:bg-gray-50">
                    <td className="px-3 py-2">{rIdx + 1}</td>
                    {featureOrder.map((f) => (
                      <td key={f} className="px-3 py-2">
                        <input
                          type="number"
                          step="any"
                          className="w-32 border rounded-md px-2 py-1"
                          value={row[f]}
                          onChange={(e) =>
                            setCell(rIdx, f, Number(e.target.value))
                          }
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex gap-2 flex-wrap">
            <button onClick={addRow} className="px-3 py-2 rounded-md border">
              + Add row
            </button>
            <button onClick={randomize} className="px-3 py-2 rounded-md border">
              Randomize
            </button>
            <button
              onClick={predict}
              disabled={loading}
              className="px-4 py-2 rounded-md bg-black text-white disabled:opacity-50"
            >
              {loading ? "Predicting…" : "Predict (Manual)"}
            </button>
            <button
              onClick={getETLResults}
              disabled={loading}
              className="px-4 py-2 rounded-md bg-blue-600 text-white disabled:opacity-50"
            >
              {loading ? "Loading…" : "🔄 Get ETL Results"}
            </button>
          </div>

          <div className="mt-2 text-sm text-gray-600">
            <p>
              <strong>Manual Predict:</strong> Usa los datos que ingreses en la
              tabla
            </p>
            <p>
              <strong>Get ETL Results:</strong> Lee los últimos datos procesados
              por tu ETL y hace inferencia
            </p>
          </div>

          {hasResults && (
            <div className="mt-6">
              <h2 className="text-xl font-semibold">Results</h2>
              <div className="mt-2 overflow-auto border rounded-xl">
                <table className="min-w-[400px] w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left">Row</th>
                      <th className="px-3 py-2 text-left">Score</th>
                      <th className="px-3 py-2 text-left">Label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr key={i} className="odd:bg-white even:bg-gray-50">
                        <td className="px-3 py-2">{r.index}</td>
                        <td className="px-3 py-2">{r.score.toFixed(3)}</td>
                        <td className="px-3 py-2">
                          <span
                            className={
                              "px-2 py-1 rounded-md " +
                              (r.label === "ANOMALY"
                                ? "bg-red-100 text-red-700"
                                : "bg-green-100 text-green-700")
                            }
                          >
                            {r.label}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </main>
  );
}
