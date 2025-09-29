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
