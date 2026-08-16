import type { HealthResponse, MetricsResponse, PredictResponse } from './types'

// Local dev uses the Vite proxy at /api (see vite.config.ts).
// In production (Vercel) the SPA calls the deployed backend directly via
// VITE_API_BASE, e.g. https://dr-grade.onrender.com (no trailing slash).
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* keep generic message */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return fetch(`${API_BASE}/health`).then((r) => json<HealthResponse>(r))
}

export function getMetrics(): Promise<MetricsResponse> {
  return fetch(`${API_BASE}/metrics`).then((r) => json<MetricsResponse>(r))
}

export function predictImage(file: File, tta: boolean): Promise<PredictResponse> {
  const body = new FormData()
  body.append('file', file)
  return fetch(`${API_BASE}/predict?tta=${tta}`, { method: 'POST', body }).then((r) =>
    json<PredictResponse>(r),
  )
}
