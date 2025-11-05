// src/lib/api.ts
import axios from "axios";
import Cookies from "js-cookie";

export type FeaturesResponse = {
  feature_order: string[];
  model_version: string;
};

export type PredictItem = {
  index: number;
  score: number;
  label: string;
  horizon_shift?: number;
  predicted_future_state?: string;
  prediction_time?: string;
};

export type PredictResponse = {
  model_version: string;
  feature_order: string[];
  results: PredictItem[];
  data_info?: {
    total_rows?: number;
    file_source?: string;
    threshold_used?: number;
    lookback_window?: number;
    horizon_shift?: number;
    ae_only_mode?: boolean;
    prediction_type?: string;
  };
};

export type Transformer = {
  id: string;
  name: string;
  location: string;
};

export type ReportFilters = {
  transformer_id: string;
  start_date?: string;
  end_date?: string;
  sample_hours: number;
};

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Create axios instance
const api = axios.create({
  baseURL: BASE,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = Cookies.get("token");
    console.log("API Request interceptor - Token:", token ? "exists" : "none");
    console.log("API Request interceptor - URL:", config.url);
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
      console.log("API Request interceptor - Authorization header set");
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid, clear cookies and redirect to login
      Cookies.remove("token");
      Cookies.remove("user");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

export default api;

export async function fetchFeatures(): Promise<FeaturesResponse> {
  const res = await fetch(`${BASE}/simple/features`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function predictFromRecords(
  records: Array<Record<string, number>>
): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // El backend acepta `records` o `gold_parquet_path`. Aquí usamos records.
    body: JSON.stringify({ records }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Nueva función para obtener resultados del ETL
export async function getMaintenanceResults(): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/simple/predictions`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Funciones para reportes
export async function getAvailableTransformers(): Promise<{
  transformers: Transformer[];
}> {
  const res = await fetch(`${BASE}/reports/available-transformers`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function downloadHistoricalDataCSV(
  filters: ReportFilters
): Promise<Blob> {
  const params = new URLSearchParams({
    transformer_id: filters.transformer_id,
    sample_hours: filters.sample_hours.toString(),
  });

  if (filters.start_date) params.append("start_date", filters.start_date);
  if (filters.end_date) params.append("end_date", filters.end_date);

  const res = await fetch(`${BASE}/reports/historical-data/csv?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

export async function downloadHistoricalDataPDF(
  filters: ReportFilters
): Promise<Blob> {
  const params = new URLSearchParams({
    transformer_id: filters.transformer_id,
    sample_hours: filters.sample_hours.toString(),
  });

  if (filters.start_date) params.append("start_date", filters.start_date);
  if (filters.end_date) params.append("end_date", filters.end_date);

  const res = await fetch(`${BASE}/reports/historical-data/pdf?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

export async function downloadPredictionsCSV(
  start_date?: string,
  end_date?: string
): Promise<Blob> {
  const params = new URLSearchParams();
  if (start_date) params.append("start_date", start_date);
  if (end_date) params.append("end_date", end_date);

  const res = await fetch(`${BASE}/reports/predictions/csv?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}
