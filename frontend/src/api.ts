export interface IngestionReport {
  total_rows: number;
  success_rows: number;
  failed_rows: number;
  errors: string[];
}

export async function uploadCsv(path: string, file: File): Promise<IngestionReport> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(path, { method: 'POST', body: form });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }
  return (await res.json()) as IngestionReport;
}

export interface TrainResponse {
  run_id: string;
  metrics: Record<string, number>;
}

export async function triggerTrain(): Promise<TrainResponse> {
  const res = await fetch('/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
  if (!res.ok) {
    throw new Error(`Train failed: ${res.status}`);
  }
  return (await res.json()) as TrainResponse;
}

export interface RunInfo {
  run_id: string;
  created_at: string;
  metrics: Record<string, number>;
}

export async function listRuns(): Promise<RunInfo[]> {
  const res = await fetch('/runs');
  if (!res.ok) {
    throw new Error(`List runs failed: ${res.status}`);
  }
  return (await res.json()) as RunInfo[];
}

export interface EntityPrediction {
  entity_id: string;
  regression: number;
  probability: number;
  ranking_score: number;
}

export interface PredictResponse {
  run_id: string;
  predictions: EntityPrediction[];
}

export async function predict(entityIds: string[]): Promise<PredictResponse> {
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entity_ids: entityIds, explanations: true }),
  });
  if (!res.ok) {
    throw new Error(`Predict failed: ${res.status}`);
  }
  return (await res.json()) as PredictResponse;
}
