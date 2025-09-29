"use client";

import React, { useState, useEffect } from "react";
import Cookies from "js-cookie";

export default function TestPredictionsPage() {
  const [predictions, setPredictions] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    testPredictions();
  }, []);

  const testPredictions = async () => {
    try {
      setLoading(true);
      setError(null);

      const token = Cookies.get("token");
      console.log("Token from cookies:", token ? "exists" : "none");

      if (!token) {
        setError("No token found in cookies");
        setLoading(false);
        return;
      }

      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      console.log("API URL:", API_URL);

      // Test predictions
      console.log("Fetching predictions...");
      const predictionsResponse = await fetch(`${API_URL}/predictions/`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      console.log("Predictions response status:", predictionsResponse.status);

      if (predictionsResponse.ok) {
        const predictionsData = await predictionsResponse.json();
        console.log("Predictions data:", predictionsData);
        setPredictions(predictionsData);
      } else {
        const errorText = await predictionsResponse.text();
        console.error("Predictions error:", errorText);
        setError(
          `Predictions error: ${predictionsResponse.status} - ${errorText}`
        );
      }

      // Test stats
      console.log("Fetching stats...");
      const statsResponse = await fetch(`${API_URL}/predictions/stats`, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      console.log("Stats response status:", statsResponse.status);

      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        console.log("Stats data:", statsData);
        setStats(statsData);
      } else {
        const errorText = await statsResponse.text();
        console.error("Stats error:", errorText);
        setError(`Stats error: ${statsResponse.status} - ${errorText}`);
      }
    } catch (err) {
      console.error("Test error:", err);
      setError(`Test error: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="p-8">
        <h1 className="text-2xl font-bold mb-4">Test Predictions</h1>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">Test Predictions</h1>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Stats</h2>
        {stats ? (
          <div className="bg-gray-100 p-4 rounded">
            <pre>{JSON.stringify(stats, null, 2)}</pre>
          </div>
        ) : (
          <p>No stats available</p>
        )}
      </div>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">
          Predictions ({predictions.length})
        </h2>
        {predictions.length > 0 ? (
          <div className="bg-gray-100 p-4 rounded">
            <pre>{JSON.stringify(predictions, null, 2)}</pre>
          </div>
        ) : (
          <p>No predictions available</p>
        )}
      </div>

      <button
        onClick={testPredictions}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
      >
        Test Again
      </button>
    </div>
  );
}
