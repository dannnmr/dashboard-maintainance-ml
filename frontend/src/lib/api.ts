// src/lib/api.ts
export type FeaturesResponse = {
  feature_order: string[];
  model_version: string;
};

export type PredictItem = {
  index: number;
  score: number;
  label: string;
};

export type PredictResponse = {
  model_version: string;
  feature_order: string[];
  results: PredictItem[];
};

const BASE = process.env.NEXT_PUBLIC_API_URL!;

export async function fetchFeatures(): Promise<FeaturesResponse> {
  const res = await fetch(`${BASE}/features`, { cache: "no-store" });
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
  const res = await fetch(`${BASE}/maintenance/results`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
